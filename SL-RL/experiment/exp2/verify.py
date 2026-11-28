"""임시 초소형 실험에서 실제 실행기, 저장·재개 및 3vs3 평가 경로를 검사합니다."""
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import launch
import torch
from analysis import collect, find_runs, read_rows
from checkpoint_selection import validation_score
from evaluation import Player, match
from pretrain import sample_mask
from storage import EXPERIMENTS, experiment, load_pt
from training import train


def main():
    torch.set_num_threads(1)
    config = (Path(__file__).parent/"config.toml").read_text(encoding="utf-8")
    import re
    # 학습률·네트워크 구조는 유지하고 경기 길이와 반복 횟수만 줄입니다.
    budgets = {"episodes": 6, "max_ticks": 4, "frames_per_episode": 3, "epochs": 1,
               "updates": 2, "episodes_per_update": 1, "checkpoint_every": 1, "games": 2}
    for key, value in budgets.items():
        config = re.sub(rf"(?m)^{key} = \d+$", f"{key} = {value}", config)
    with tempfile.TemporaryDirectory(dir=EXPERIMENTS, prefix=".sanity_exp2_") as temp:
        root = Path(temp)
        (root/"config.toml").write_text(config, encoding="utf-8")
        root, cfg = experiment(root.name)
        subprocess.run([sys.executable, "-B", str(launch.HERE), "--experiment", root.name], check=True)
        runs = find_runs(root, None)
        keys = {(r.settings["config"]["representation"], r.size, r.seed) for r in runs}
        assert len(runs) == len(keys) == 12
        assert len(list(root.glob("pretrain/*/fit*/encoder.pt"))) == 2
        assert not sample_mask(torch.ones(3, 30), 0).any()
        for r in runs:
            assert set(p.name for p in r.path.glob("*.pt")) == {"best.pt", "latest.pt"}
            latest = load_pt(r.path/"latest.pt")
            best = load_pt(r.path/"best.pt")
            rows = latest["validation_history"]
            assert latest["update"] == 2 and [x["update"] for x in rows] == [1, 2]
            assert all(x["games"] == 16 and x["size"] == r.size for x in rows)
            assert best["update"] == max(rows, key=lambda x: (x["score"], x["update"]))["update"]
            assert len(read_rows(r.path/"validation.csv")) == len(rows)
            raw = r.settings["config"]["representation"] == "raw"
            assert (latest["fit"] is None) == (r.settings["encoder"] is None) == raw
            player = Player(r.path, "best.pt")
            assert player.encoder(torch.zeros(1, 30)).shape == (1, 30 if raw else 16)
        for metric_file in root.glob("pretrain/*/fit*/metrics.csv"):
            assert int(read_rows(metric_file)[0]["all_cells"]) == 2 * 3 * 30
        # 서로 다른 학습 인원 모두 실제로 해당 크기의 경기와 고정된 세 시작 행을 사용하는지 검사합니다.
        for size in (1, 3):
            chosen = next(r for r in runs if r.size == size)
            with patch("checkpoint_selection.match", wraps=match) as observed:
                validation_score(Player(chosen.path, "best.pt"), 6, size=size)
                assert {c.kwargs["size"] for c in observed.call_args_list} == {size}
                if size == 1:
                    assert {c.kwargs["row"] for c in observed.call_args_list} == {0, 1, 2}
        # 완료된 실행을 다시 찾고, 같은 시드의 연속 실행과 재개 실행의 가중치를 대조합니다.
        chosen = next(r for r in runs if r.settings["config"]["representation"] == "raw" and r.size == 1)
        raw_cfg = chosen.settings["config"]
        assert launch.existing_run(root, raw_cfg, 1, chosen.seed, None) == chosen.path
        clone = train(root, raw_cfg, 1, chosen.seed, updates=1)
        train(root, cfg, None, None, resume=clone.name, updates=2)
        a, b = (load_pt(p/"latest.pt") for p in (chosen.path, clone))
        assert a["validation_history"] == b["validation_history"]
        for first, second in zip(a["policies"], b["policies"]):
            assert all(torch.equal(first[k], second[k]) for k in first)
        # 네 조건의 고정 상대 평가, 같은 입력끼리의 복제 맞대결 및 보고서 생성을 확인합니다.
        from analysis_plots import report
        selected = [r for r in runs if r.seed == 7]
        data = collect(root, cfg, selected, 2, 2, execute=True)
        assert len(data) == 6 and sum(e.rival is not None for e in data) == 2
        for mode in ("ae", "raw"):
            group = [r for r in selected if r.settings["config"]["representation"] == mode]
            dest = root/"result"/mode
            dest.mkdir(parents=True)
            report(root, dest, dest, group, 2, [e for e in data if e.run in group], cfg["evaluation"], 2)
            assert (dest/"REPORT.md").is_file() and (dest/"ssl.png").is_file() == (mode == "ae")
    print("exp2 검증 통과 · 세 터미널, 12개 run, 인원별 검증, 두 체크포인트, 재개 일치, 복제 평가 및 보고서")


if __name__ == "__main__":
    main()
