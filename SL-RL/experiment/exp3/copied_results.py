"""16차원 기준 결과를 이전 실험의 상대경로 구조 그대로 복사하고 해시를 확인합니다."""
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tempfile

from analysis import collect, find_runs, metadata
from experiment.exp3.grid import SIZES, compatible, label
from storage import experiment, load_pt, write_json


def digest(file):
    with file.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_copy(target, name, variant, seeds):
    manifest = metadata(target/"ORIGIN.json")
    expected = {(s, seed) for s in SIZES for seed in seeds}
    runs = find_runs(target, None)
    if (manifest["source_experiment"] != name or len(runs) != len(expected)
            or {(r.size, r.seed) for r in runs} != expected
            or any(not compatible(r.settings["config"], variant) for r in runs)):
        raise ValueError(f"복사된 실행 조건과 현재 그리드가 다릅니다: {target}")
    for rel, checksum in manifest["sha256"].items():
        if digest(target/rel) != checksum:
            raise ValueError(f"복사본의 해시가 달라졌습니다: {target/rel}")
    return runs


def import_baseline(root, name, key, variant, seeds):
    parent = root/"copied"
    target = parent/name
    if target.exists():
        check_copy(target, name, variant, seeds)
        return
    src, _ = experiment(name)
    runs = [r for r in find_runs(src, None) if r.seed in seeds and compatible(r.settings["config"], variant)]
    expected = {(size, seed) for size in SIZES for seed in seeds}
    if len(runs) != len(expected) or {(r.size, r.seed) for r in runs} != expected:
        raise ValueError(f"{name}의 학습 조건과 시드가 일치하지 않습니다. 새 설정에는 reuse_latent16=false를 사용하세요.")
    update = variant["ppo"]["updates"]
    files = {src/"config.toml"}
    trees = set()
    for r in runs:
        state = load_pt(r.path/"latest.pt")
        if (state["update"], state["size"], state["seed"]) != (update, r.size, r.seed) or not compatible(state["config"], variant):
            raise ValueError(f"{r.path}의 최종 체크포인트를 확인하세요.")
        files.update(r.path/f for f in ("settings.json", "training.csv", "latest.pt"))
        if "online_validation" in r.settings["config"]:
            files.update(r.path/f for f in ("best.pt", "validation.csv"))
        elif (r.path/"best.pt").is_file():
            files.add(r.path/"best.pt")
        trees.add((src/r.settings["encoder"]).parent)
    # 이미 완료된 평가의 시드·진영·상대 조합을 검사하고 해당 평가만 복사합니다.
    evaluations = collect(src, variant, runs, update, variant["evaluation"]["games"])
    trees.update(e.path for e in evaluations)
    figures = src/"result"/f"analysis_u{update:04d}"
    if name == "exp2":
        figures /= "ae"
    if not list(figures.glob("eval_*/REPORT.md")) or not list(figures.glob("eval_*/replays/index.csv")):
        raise FileNotFoundError(f"복사할 보고서 또는 리플레이를 찾지 못했습니다: {figures}")
    trees.add(figures)
    for tree in trees:
        files.update(p for p in tree.rglob("*") if p.is_file() and p.name != ".training.lock")
    parent.mkdir(parents=True, exist_ok=True)
    # 완성 전 중단된 폴더가 정상 복사본으로 인식되지 않도록 같은 디스크에서 공개합니다.
    with tempfile.TemporaryDirectory(dir=parent, prefix=".copy_") as tmp:
        bundle = Path(tmp)/name
        hashes = {}
        for file in sorted(files):
            rel = file.resolve().relative_to(src.resolve())
            dest = bundle/rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, dest)
            hashes[rel.as_posix()] = digest(file)
        write_json(bundle/"ORIGIN.json", dict(source_experiment=name, condition=label(key),
                   copied_utc=datetime.now(timezone.utc).isoformat(), update=update,
                   runs={r.name: dict(size=r.size, seed=r.seed) for r in runs}, sha256=hashes))
        check_copy(bundle, name, variant, seeds)
        if bundle.resolve().parent != Path(tmp).resolve() or target.resolve().parent != parent.resolve():
            raise ValueError("복사 대상 경로를 확인하세요.")
        bundle.rename(target)
    print(f"{label(key)} · {name}에서 {len(runs)}개 run, {len(files)}개 파일 복사·검증", flush=True)
