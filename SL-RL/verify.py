"""작은 예산으로 규칙, 재개 재현성, 고정 상대와 복제 평가를 검증합니다."""
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import run  # 원본 환경 패키지를 실행 경로에 추가합니다.
import torch
from arena import Action, Arena, SIDES, inputs, opening_row, reflect
from storage import EXPERIMENTS, ROOT, load_pt


def check_rules():
    game = dict(height=3, width=10, max_ticks=4, respawn_delay=1)
    for start in range(0, 12, 3):
        assert sorted(opening_row(19, i, 3) for i in range(start,start+3)) == [0,1,2]
    for size in (1,3):
        # 본 실험의 64틱 시간 제한을 원본 및 두 명 환경 모두에서 확인합니다.
        idle = Arena(dict(game, max_ticks=64), size, 19, 1)
        stopped = {aid:Action.NOOP for team in idle.teams for aid in team}
        for i in range(64):
            idle.step(stopped, 1)
            assert idle.state.terminated == (i == 63)
        assert idle.state.areas == (15,15)
        arena = Arena(game, size, 19, 1)
        opening = arena.state
        assert len(opening.agents) == size*2
        for side in range(2):
            board, own, entities, mask = inputs(arena, side)
            assert entities.shape == (size,2*size-1,6) and board.shape == (size,30)
            for i, aid in enumerate(arena.teams[side]):
                for action in Action:
                    assert mask[i,action.index] == arena.env.action_mask(aid)[reflect(action,side).index]
        actions = {a.agent_id:Action.NOOP for a in opening.agents}
        actions["red_0"] = Action.ATTACK_RIGHT
        rewards = arena.step(actions, 1)
        assert arena.state.areas == (16,14)
        assert abs(rewards["red_0"]-1-16/30) < 1e-6
        assert abs(rewards["blue_0"]-14/30) < 1e-6
        assert not arena.state.terminated and not any(a.alive for a in arena.state.agents if a.agent_id=="blue_0")
        assert sum(arena.env.action_mask("blue_0")) == 1
        arena.step(dict.fromkeys(actions, Action.NOOP), 1)
        assert next(a for a in arena.state.agents if a.agent_id=="blue_0").alive


def smoke():
    check_rules()
    # 임시 실험은 지정된 SL-RL/experiment 내부에서만 생성하고 회수합니다.
    with tempfile.TemporaryDirectory(prefix=".sanity", dir=EXPERIMENTS) as tmp:
        path = Path(tmp)
        assert path.resolve().parent == EXPERIMENTS.resolve()
        cfg = (EXPERIMENTS/"exp1"/"config.toml").read_text(encoding="utf-8")
        # 사용자가 변경한 exp1 설정과 독립적으로 일정한 작은 예산을 보장합니다.
        overrides = {"game":{"height":3,"width":10,"max_ticks":4,"respawn_delay":1},
                     "ssl":{"episodes":10,"epochs":2,"frames_per_episode":3,"batch_size":8},
                     "ppo":{"updates":2,"episodes_per_update":2,"epochs":1,"checkpoint_every":1}}
        import re
        for section, values in overrides.items():
            section_re = r"(?ms)^\["+section+r"\]\s*\n(.*?)(?=^\[|\Z)"
            def replace(m):
                block = m.group()
                for key, value in values.items():
                    block = re.sub(r"(?m)^"+key+r"\s*=.*$", f"{key} = {value}", block)
                return block
            cfg = re.sub(section_re, replace, cfg)
        (path/"config.toml").write_text(cfg,encoding="utf-8")
        def cli(command, *args):
            subprocess.run([sys.executable, "-B", str(ROOT/"run.py"),command,"--experiment",path.name,*args],
                           cwd=ROOT,check=True)
        for size in ("1","3"):
            cli("pretrain","--size",size)
            cli("train","--size",size,"--seed","7","--updates","1")
        cli("resume","--run","run1","--updates","2")
        cli("train","--size","1","--seed","7","--updates","2","--keep-last-and-best","--validation-games","2")
        assert not list((path/"run3").glob("update_*.pt"))
        assert (path/"run3/best.pt").is_file()
        first = path/"run1"
        assert {p.name for p in first.glob("update_*.pt")} == {"update_0001.pt", "update_0002.pt"}
        from checkpoint_selection import keep_last_and_best
        from unittest.mock import patch
        from storage import run_lock
        def score(player, games):
            return {"update":player.update, "score":1.0/player.update}
        with run_lock(first):
            with patch("checkpoint_selection.validation_score", side_effect=RuntimeError("검증 중단")):
                try:
                    keep_last_and_best(first, 2, 2)
                except RuntimeError as exc:
                    assert str(exc) == "검증 중단"
                else:
                    raise AssertionError("검증 실패가 전달되어야 합니다.")
            assert len(list(first.glob("update_*.pt"))) == 2
            with patch("checkpoint_selection.validation_score", side_effect=score):
                keep_last_and_best(first, 2, 2)
        assert load_pt(first/"best.pt")["update"] == 1
        assert not list(first.glob("update_*.pt"))
        from evaluation import Player
        assert Player(first, "update_0001.pt").update == 1
        assert Player(first, "update_0002.pt").update == 2
        cli("prune-checkpoints","--run","run3","--at-update","2","--validation-games","2")
        a, b = (load_pt(path/r/"latest.pt") for r in ("run1","run3"))
        for left,right in zip(a["policies"],b["policies"]):
            assert all(torch.equal(left[k],right[k]) for k in left)
        for r in ("run1","run2"):
            cli("evaluate","--run",r,"--games","2","--replay")
            with (path/r/"evaluation/eval1/summary.csv").open(encoding="utf-8-sig") as f:
                rows=list(csv.DictReader(f))
            assert len(rows)==10 and all(int(x["wins"])+int(x["draws"])+int(x["losses"])==2 for x in rows)
        cli("evaluate","--run","run3","--checkpoint","best.pt","--games","2","--opponents","expand")
        cli("resume","--run","run1","--updates","3","--keep-last-and-best","--validation-games","2")
        assert load_pt(first/"latest.pt")["update"] == 3 and not list(first.glob("update_*.pt"))
        cli("duel","--run","run1","--versus","run2","--games","2")
        with (path/"comparison/eval1/games.csv").open(encoding="utf-8-sig") as f:
            assert len(list(csv.DictReader(f)))==8
    print("검증 통과: 환경 규칙, 사전학습 2종, PPO 2종, 재개 일치, 고정 상대 5종, 복제 맞대결")


if __name__ == "__main__":
    smoke()
