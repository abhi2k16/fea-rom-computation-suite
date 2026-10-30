"""Worker-count scaling sweep for Section 7.5. Run on the many-core machine:

    python multicore_scaling.py            # sweeps 1 .. os.cpu_count() workers
    python multicore_scaling.py 1 2 4 8    # or a chosen list

Writes experiments/multicore_scaling.jsonl (one line per worker count, plus a header line with the machine)."""
import sys, os, json, time, platform, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "fea_engine", "src"))
N_ITEMS = 48          # divisible by 1,2,3,4,6,8,12 so every worker count gets equal work


def child(nj):
    sys.path.insert(0, SRC); sys.path.insert(0, HERE)
    from multicore import plate_job
    from fea_engine.batch import map as fe_map
    items = [150e9 + 5e9 * i for i in range(N_ITEMS)]
    fe_map(plate_job, items[:2], n_jobs=1)            # warm-up
    best = 1e9
    for _ in range(3):
        t = time.perf_counter(); r = fe_map(plate_job, items, n_jobs=nj, backend="process")
        best = min(best, time.perf_counter() - t)
    print(json.dumps(dict(case="plate", n_jobs=nj, backend="process", blas="1", time=best, n_items=N_ITEMS, check=r[0])))


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--child":
        child(int(sys.argv[2])); sys.exit()
    ncpu = os.cpu_count()
    counts = [int(a) for a in sys.argv[1:]] or list(range(1, ncpu + 1))
    env = dict(os.environ, **{k: "1" for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")})
    out = os.path.join(HERE, "multicore_scaling.jsonl")
    with open(out, "w") as f:
        f.write(json.dumps(dict(machine=platform.processor() or platform.machine(), logical_cpus=ncpu,
                                system=platform.platform(), python=platform.python_version())) + "\n")
        for nj in counts:
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", str(nj)], env=env,
                               capture_output=True, text=True)
            line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else None
            if line is None:
                print("failed for", nj, p.stderr[-500:]); continue
            f.write(line + "\n"); f.flush(); print(line)
    print("wrote", out)
