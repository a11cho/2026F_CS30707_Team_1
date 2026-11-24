"""4틱 임시 실험에서 신규 조건별 3콘솔 학습, 재개 및 맞대결 조합을 검사합니다."""
import csv
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import launch
import torch
from analysis import collect, find_runs
from experiment.exp3.grid import SIZES, conditions, key_of
from pretrain import sample_mask
from storage import EXPERIMENTS, ROOT, experiment, load_pt
from training import train


def main():
    torch.set_num_threads(1)
    for count in (0, 9):
        masks = sample_mask(torch.ones(100, 30), count/30)
        assert (masks.sum(1) == count).all()
    base = (Path(__file__).parent/"config.toml").read_text(encoding="utf-8")
    budgets = {"episodes": 6, "max_ticks": 4, "frames_per_episode": 2, "epochs": 1, "games": 2,
               "updates": 2, "episodes_per_update": 1, "checkpoint_every": 1}
    for key, value in budgets.items():
        base = re.sub(rf"(?m)^{key} = \d+$", f"{key} = {value}", base)
    base = base.replace("latent_sizes = [8, 16, 32]", "latent_sizes = [8, 32]").replace(
                  "reuse_latent16 = true", "reuse_latent16 = false")
    with tempfile.TemporaryDirectory(dir=EXPERIMENTS, prefix=".sanity3_") as tmp:
        root = Path(tmp)
        (root/"config.toml").write_text(base, encoding="utf-8")
        _, cfg = experiment(root.name)
        variants = conditions(cfg)
        fits = launch.prepare(root, cfg, variants)
        # 한 run을 첫 저장까지 진행하고 일괄 실행에서 같은 폴더로 재개하게 합니다.
        first = train(root, variants[9, 8], 1, 7, fit_name=fits["mask09_z8_1"], updates=1)
        before = load_pt(first/"latest.pt")
        cmd = [sys.executable, "-B", str(launch.HERE), "--experiment", root.name]
        subprocess.run(cmd, check=True)
        runs = find_runs(root, None)
        seeds = cfg["ae_grid"]["seeds"]
        expected = {(key, size, seed) for key in variants for size in SIZES for seed in seeds}
        assert len(runs) == len(expected) == 24
        assert {(key_of(r.settings["config"]), r.size, r.seed) for r in runs} == expected
        assert len(list(root.glob("pretrain/*/fit*/encoder.pt"))) == 8
        for r in runs:
            state = load_pt(r.path/"latest.pt")
            assert state["update"] == 2 and len(state["validation_history"]) == 2
            assert all(row["size"] == r.size for row in state["validation_history"])
            assert state["fit"]["config"]["ssl"] == r.settings["config"]["ssl"]
            assert {p.name for p in r.path.glob("*.pt")} == {"best.pt", "latest.pt"}
        after = load_pt(first/"latest.pt")
        assert after["history"][:len(before["history"])] == before["history"]
        # 완료된 worker를 다시 실행해도 학습 로그나 가중치가 추가되지 않습니다.
        launch.worker(root, cfg, variants, 7, fits)
        assert len(find_runs(root, None)) == 24 and load_pt(first/"latest.pt")["history"] == after["history"]
        for stage in ("evaluate", "report"):
            subprocess.run([sys.executable, "-B", str(ROOT/"analysis.py"), stage, "--experiment", root.name], check=True)
        for key in variants:
            subset = [r for r in runs if key_of(r.settings["config"]) == key]
            data = collect(root, cfg, subset, 2, 2)
            assert len(data) == 9
            assert all(not e.rival or (key_of(e.rival.settings["config"]) == key and e.run.seed == e.rival.seed) for e in data)
        summary = next((root/"result/analysis_u0002").glob("grid_*.csv"))
        with summary.open(encoding="utf-8-sig") as file:
            rows = list(csv.DictReader(file))
        assert len({r["condition"] for r in rows}) == 4
        assert all(int(r["games"]) == 24 for r in rows if r["opponent"] == "duel_1v3" and r["seed"] == "all")
    print("exp3 검증 통과 · 정확히 0/9칸, 신규 24개 조합, 세 콘솔, 체크포인트 재개, 조건별 평가와 비교표")


if __name__ == "__main__":
    main()
