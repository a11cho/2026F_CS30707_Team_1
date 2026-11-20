"""대표 후보를 별도 프로세스에서 재학습하여 Windows 자원 사용량을 측정합니다.

기존 전체 실행의 측정값을 소급해서 만들어내지 않습니다. 결과를 출력할 뿐
기존 체크포인트·로그를 바꾸지 않으며, 비교 대상 Q-table의 재현 여부도 검사합니다.
"""
import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import subprocess
import sys
import time


def memory():
    """현재 프로세스의 working set·최대 working set·private commit을 바이트로 조회합니다."""
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
            "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
            "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    counter = Counters()
    counter.cb = ctypes.sizeof(counter)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counter), counter.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return dict(working_set_bytes=counter.WorkingSetSize, peak_working_set_bytes=counter.PeakWorkingSetSize,
                private_commit_bytes=counter.PrivateUsage)


def child(run_dir, checkpoint):
    """독립 프로세스 한 개에서 import 후 기준점과 학습 직후 자원을 측정합니다."""
    repo = run_dir.parents[2]
    sys.path.insert(0, str(repo))
    import numpy as np
    from territory_rl import ExperimentConfig
    from smoke_test_mc_td.config import RunConfig
    from smoke_test_mc_td.learning import load_models
    from smoke_test_mc_td.runner import train
    reference, metadata = load_models(run_dir / checkpoint)
    cfg = RunConfig(**metadata["config"])
    experiment = ExperimentConfig(**metadata["experiment"])
    # 기준 모델은 비교용이며, 크기가 작은 실제 후보 객체를 기준 메모리에 포함합니다.
    before = memory()
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    models, rows = train(cfg, experiment, metadata["algorithm"], metadata["seed"])
    cpu_seconds, wall_seconds = time.process_time() - cpu_start, time.perf_counter() - wall_start
    after = memory()
    same = len(models) == len(reference)
    for actual, expected in zip(models, reference):
        same = same and actual.updates == expected.updates and actual.q.keys() == expected.q.keys()
        same = same and all(np.array_equal(value, expected.q[key]) for key, value in actual.q.items())
    return dict(checkpoint=checkpoint, algorithm=metadata["algorithm"], seed=metadata["seed"],
                experiment=metadata["experiment"], training_episodes=len(rows), ticks=cfg.episodes*cfg.ticks,
                wall_seconds=wall_seconds, cpu_seconds=cpu_seconds,
                one_core_cpu_percent=cpu_seconds / wall_seconds * 100,
                before=before, after=after, q_states=sum(len(m.q) for m in models),
                q_array_bytes=sum(v.nbytes for m in models for v in m.q.values()), exact_q_reproduction=bool(same))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--child-checkpoint")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if args.child_checkpoint:
        print(json.dumps(child(run_dir, args.child_checkpoint), ensure_ascii=False))
        return
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    profiles = []
    for obs in ("local", "aware"):
        for algorithm in ("mc", "sarsa"):
            best = max((s for s in results["selected"] if s["condition"].startswith(obs+"_") and s["algorithm"] == algorithm),
                       key=lambda s: (s["validation"]["score"], s["validation"]["area_fraction"]))
            completed = subprocess.run([sys.executable, "-B", "-X", "utf8", str(Path(__file__).resolve()),
                                        str(run_dir), "--child-checkpoint", best["checkpoint"]],
                                       text=True, encoding="utf-8", capture_output=True, check=True)
            profiles.append(json.loads(completed.stdout))
    print(json.dumps(dict(kind="별도 대표 후보 재학습 측정", profiles=profiles), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
