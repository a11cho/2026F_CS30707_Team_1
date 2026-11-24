"""고정 검증 경기로 팀 정책 쌍을 비교한 뒤 선택한 run의 보관 파일을 정리합니다."""
import os
import re
from pathlib import Path

from arena import opening_row, seed_for
from evaluation import Player, match
from storage import load_pt, save_pt, write_json

OPPONENTS = ("random", "expand", "aggressive", "defend")
DEFAULT_GAMES = 4


def validation_score(player, games, size=3):
    """양 학습 정책·상대별로 진영을 교대하고 같은 환경·봇 시드를 사용합니다."""
    wins = draws = 0
    for trained in range(2):
        for opponent in OPPONENTS:
            for pair in range(games // 2):
                # 본 평가의 난수 스트림(71)과 분리합니다.
                seed = seed_for(player.cfg["evaluation"]["seed"], pair, 72)
                row = opening_row(seed_for(player.cfg["evaluation"]["seed"], 72), pair, player.cfg["game"]["height"])
                for field in range(2):
                    result, _ = match(player, trained, field, opponent, None, seed, False, size=size, row=row if size == 1 else 0)
                    wins += result["win"]
                    draws += result["draw"]
    total = 2 * len(OPPONENTS) * games
    return dict(update=player.update, wins=wins, draws=draws, losses=total-wins-draws,
                games=total, score=(wins+0.5*draws)/total)


def keep_last_and_best(run: Path, expected_update: int, games: int):
    """호출자는 run 잠금을 보유합니다. 최종 파일과 검증 결과를 확보한 뒤 정리합니다."""
    if games < 2 or games % 2:
        raise ValueError("검증 경기 수는 2 이상의 짝수여야 합니다.")
    latest = Player(run, "latest.pt")
    if "online_validation" in latest.cfg:
        raise ValueError("이 run은 학습 중에 best를 선정합니다. validation.csv를 확인하세요.")
    if latest.update != expected_update:
        raise ValueError(f"{run.name}: latest.pt가 {latest.update}회입니다. 요청한 {expected_update}회와 다릅니다.")

    numbered = {}
    for path in run.glob("update_*.pt"):
        m = re.fullmatch(r"update_([0-9]+)\.pt", path.name)
        if not m:
            continue
        step = int(m[1])
        if path.name != f"update_{step:04d}.pt" or not path.is_file():
            continue
        if step > latest.update:
            raise ValueError(f"{path.name}이 latest.pt보다 앞서 있습니다. 체크포인트를 확인하세요.")
        numbered[step] = path

    candidates = dict(numbered)
    best_path = run/"best.pt"
    if best_path.exists():
        previous = Player(run, best_path.name)
        if previous.update > latest.update:
            raise ValueError("best.pt의 업데이트가 latest.pt보다 큽니다.")
        candidates.setdefault(previous.update, best_path)
    candidates[latest.update] = run/"latest.pt"

    results = []
    for step, path in sorted(candidates.items()):
        player = latest if step == latest.update else Player(run, path.name)
        if player.update != step or player.seed != latest.seed or player.size != latest.size:
            raise ValueError(f"{path.name}의 실행 정보가 일치하지 않습니다.")
        row = validation_score(player, games)
        results.append(row)
        print(f"{run.name} | update {step} | 검증 점수 {row['score']:.3f}", flush=True)

    winner = max(results, key=lambda r: (r["score"], r["update"]))
    source = candidates[winner["update"]]
    if source != best_path:
        save_pt(best_path, load_pt(source))
    report = dict(last_update=latest.update, best_update=winner["update"],
                  validation_seed=latest.cfg["evaluation"]["seed"], seed_stream=72,
                  games_per_policy_opponent=games, opponents=list(OPPONENTS),
                  results=results)
    # 동일한 완료 시점의 재실행도 이전 전체 후보의 선정 기록을 보존합니다.
    index = 1
    while (run/f"selection_{latest.update:04d}_{index}.json").exists():
        index += 1
    report_path = run/f"selection_{latest.update:04d}_{index}.json"
    temp = report_path.with_suffix(".tmp")
    write_json(temp, report)
    os.replace(temp, report_path)
    for path in numbered.values():
        path.unlink()
    print(f"{run.name}: latest.pt={latest.update}, best.pt={winner['update']}, 번호 파일 {len(numbered)}개 정리", flush=True)
