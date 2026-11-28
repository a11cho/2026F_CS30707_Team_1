"""경기 단위로 분리한 점령 지도에서 설정에 따라 전체 칸 또는 가려진 칸을 복원합니다."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from arena import Arena, bot, opening_row, ownership, seed_for
from models import MapEncoder
from storage import allocate, save_pt, write_csv, write_json

STYLES = ("random", "expand", "aggressive", "defend")


def collect(cfg, size, count, phase):
    game, opt = cfg["game"], cfg["ssl"]
    base = seed_for(opt["seed"], phase)
    frames = []
    for episode in range(count):
        seed = seed_for(base, episode)
        order = np.random.default_rng(seed_for(base, episode // 16, 2)).permutation(16)
        pair = int(order[episode % 16])
        arena = Arena(game, size, seed, opening_row(base, episode, game["height"]))
        bots = [[bot(STYLES[style], seed_for(seed, side, i)) for i in range(size)]
                for side, style in enumerate(divmod(pair, 4))]
        boards = []
        while not arena.state.terminated:
            boards.append([ownership(arena.state, side) for side in range(2)])
            actions = {aid: bots[side][i].act(arena.observation(side, i))
                       for side, team in enumerate(arena.teams) for i, aid in enumerate(team)}
            arena.step(actions, 0)
        boards.append([ownership(arena.state, side) for side in range(2)])
        # 짧게 끝난 경기도 같은 수로 표집하며 두 진영의 좌표계를 모두 봅니다.
        idx = np.linspace(0, len(boards)-1, opt["frames_per_episode"]).round().astype(int)
        frames.extend(np.asarray(boards)[idx].reshape(-1, game["height"] * game["width"]))
        if (episode+1) % 32 == 0:
            print(f"{size}vs{size} 지도 수집 {episode+1}/{count}", flush=True)
    return torch.from_numpy(np.stack(frames))


def sample_mask(board, fraction, generator=None):
    if fraction == 0:
        return torch.zeros_like(board, dtype=torch.bool)
    k = max(1, min(board.shape[1]-1, round(board.shape[1]*fraction)))
    indices = torch.rand(board.shape, generator=generator).argsort(1)[:, :k]
    return torch.zeros_like(board, dtype=torch.bool).scatter_(1, indices, True)


def validation(encoder, decoder, board, mask, game):
    h, w = game["height"], game["width"]
    logits = decoder(encoder(board, mask))
    cells = board.reshape(-1, 1, h, w)
    seen = (~mask).reshape_as(cells).float()
    kernel = torch.tensor([[0.,1.,0.],[1.,0.,1.],[0.,1.,0.]]).reshape(1,1,3,3)
    def neighbours(x):
        return F.conv2d(x, kernel, padding=1).flatten(1)
    starting = (torch.arange(w).repeat(h) < w//2).expand_as(board)
    vote = 2*neighbours(cells*seen) - neighbours(seen)
    majority = torch.where(vote == 0, starting, vote > 0)
    # 실제 지도에서 상하좌우 하나라도 반대 진영 소유면 전선입니다.
    boundary = torch.where(board.bool(), neighbours(1-cells)>0, neighbours(cells)>0)
    scored = mask if mask.any() else torch.ones_like(mask)
    metrics = {"validation_loss": F.binary_cross_entropy_with_logits(logits[scored], board[scored]).item()}
    for region, keep in [("all", scored), ("frontier", scored & boundary)]:
        metrics[region+"_cells"] = int(keep.sum())
        for name, pred in [("encoder", logits>=0), ("initial", starting), ("neighbour", majority)]:
            metrics[name+"_"+region] = (pred[keep] == board[keep]).float().mean().item() if keep.any() else None
    return metrics


def pretrain(path, cfg, size):
    opt, game = cfg["ssl"], cfg["game"]
    n = opt["episodes"]
    if n < 2 or not 0 < opt["validation_fraction"] < 1 or not 0 <= opt["mask_fraction"] < 1:
        raise ValueError("사전학습에는 두 경기 이상과 0~1 사이의 검증·마스크 비율이 필요합니다.")
    valid_n = min(n-1, max(1, round(n * opt["validation_fraction"])))
    out = allocate(path / "pretrain" / f"{size}v{size}", "fit")
    write_json(out/"settings.json", cfg)
    torch.manual_seed(opt["seed"])
    train = collect(cfg, size, n-valid_n, 11)
    val = collect(cfg, size, valid_n, 12)
    fixed = sample_mask(val, opt["mask_fraction"], torch.Generator().manual_seed(seed_for(opt["seed"], 13)))
    cells = train.shape[1]
    encoder = MapEncoder(cells, opt["hidden"], opt["latent"])
    decoder = nn.Sequential(nn.Linear(opt["latent"], opt["hidden"]), nn.ReLU(), nn.Linear(opt["hidden"], cells))
    optim = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=opt["learning_rate"])
    best, history = float("inf"), []
    for epoch in range(1, opt["epochs"]+1):
        total = 0.
        for ids in torch.randperm(len(train)).split(opt["batch_size"]):
            batch = train[ids]
            mask = sample_mask(batch, opt["mask_fraction"])
            logits = decoder(encoder(batch, mask))
            if opt["mask_fraction"] == 0:
                loss = F.binary_cross_entropy_with_logits(logits, batch)
            else:
                loss = F.binary_cross_entropy_with_logits(logits[mask], batch[mask])
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item() * len(batch)
        with torch.no_grad():
            stats = validation(encoder, decoder, val, fixed, game)
        history.append(dict(epoch=epoch, train_loss=total/len(train), **stats))
        if stats["validation_loss"] < best:
            best = stats["validation_loss"]
            save_pt(out/"encoder.pt", dict(size=size, config=cfg, weights=encoder.state_dict(), epoch=epoch))
        write_csv(out/"metrics.csv", history)
    print(out, flush=True)
    return out
