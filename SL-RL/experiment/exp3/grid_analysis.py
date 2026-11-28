"""복사한 16차원과 신규 실행을 같은 평가 조건으로 묶어 시드별 비교표를 만듭니다."""
from collections import defaultdict

from analysis import analyze_group, find_runs
from experiment.exp3.grid import SIZES, conditions, copied_source, key_of, label
from storage import write_csv


def grid_analysis(parser, args, root, cfg, update):
    variants = conditions(cfg)
    labels = {label(key) for key in variants}
    if args.condition and args.condition not in labels:
        parser.error(f"조건을 선택하세요: {sorted(labels)}")
    if args.runs and not args.condition:
        parser.error("exp3에서는 run 번호와 함께 --condition을 지정하세요.")
    out = root/"result"/f"analysis_u{update:04d}"
    summary = []
    for key in variants:
        condition = label(key)
        if args.condition and args.condition != condition:
            continue
        origin = copied_source(cfg, key)
        source = root/"copied"/origin if origin else root
        if origin and not (source/"ORIGIN.json").is_file():
            raise ValueError("launch.py --import-only로 복사본을 먼저 준비하세요.")
        runs = [r for r in find_runs(source, args.runs) if key_of(r.settings["config"]) == key]
        if args.runs:
            if set(args.runs) != {r.name for r in runs}:
                parser.error("요청한 run의 마스킹 칸 수와 차원을 확인하세요.")
        else:
            expected = {(s,t) for s in SIZES for t in cfg["ae_grid"]["seeds"]}
            if {(r.size,r.seed) for r in runs} != expected or len(runs) != len(expected):
                raise ValueError(f"{condition}의 전체 run이 준비되지 않았습니다. 일부 분석에는 --condition과 --runs를 사용하세요.")
        dest = out/condition
        if args.runs:
            dest /= "_".join(r.name for r in runs)
        result = analyze_group(parser, args, source, cfg, runs, update, dest)
        if args.stage == "report":
            from analysis_plots import aggregate
            data, report_dir = result
            with (report_dir/"REPORT.md").open("a", encoding="utf-8") as stream:
                stream.write(f"\n조건 {condition} · {origin+'에서 복사한 기준 결과' if origin else '이 실험에서 학습한 결과'}\n")
            groups = defaultdict(list)
            for e in data:
                for row in e.games:
                    opponent = "duel_1v3" if e.rival else row["opponent"]
                    for seed in (e.run.seed, "all"):
                        groups[e.run.size, opponent, seed].append(row)
            for (size, opponent, seed), rows in groups.items():
                summary.append(dict(condition=condition, masked_cells=key[0], latent=key[1],
                                    origin=origin or root.name, training_size=size, seed=seed,
                                    opponent=opponent, **aggregate(rows)))
    if summary:
        folder = dest if args.condition else out
        opts = cfg["evaluation"]
        games = opts["games"] if args.games is None else args.games
        filename = f"grid_{opts['seed']}_{games}_{'-'.join(opts['opponents'])}.csv"
        write_csv(folder/filename, summary)
        print(folder/filename, flush=True)
