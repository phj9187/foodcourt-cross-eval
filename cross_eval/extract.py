"""Step 1: pull per-day components out of every finished job's paired_test.json.

Each job folder <experiment>/<objective id>/rep_<n>/ holds summary.json and the
large paired_test.json. Only what re-scoring needs is kept (a few hundred KB per
job), saved as <cache>/<objective id>__rep_<n>.npz. Cached jobs are skipped, so a
second run only reads newly finished jobs.
"""
import argparse
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:
    import orjson
    _loads = orjson.loads
except ImportError:  # pragma: no cover
    _loads = json.loads

from .scoring import COMPONENTS, period_scores


def _arrays(rows):
    opened = np.array([bool(r['opened']) for r in rows])
    comp = np.array([[float((r.get('components') or {}).get(k, 0.0)) for k in COMPONENTS] for r in rows])
    profit = np.array([float(r.get('contribution_profit', 0.0)) for r in rows])
    return opened, comp, profit


def extract_job(job_dir, cache_dir):
    job_dir, cache_dir = Path(job_dir), Path(cache_dir)
    name = f'{job_dir.parent.name}__{job_dir.name}'
    out = cache_dir / f'{name}.npz'
    if out.exists():
        return name, 'cached', 0.0
    t0 = time.time()
    summary = json.loads((job_dir / 'summary.json').read_text(encoding='utf-8'))
    objective = summary['objective']
    pairs = _loads((job_dir / 'paired_test.json').read_bytes())
    base = [_arrays(p['baseline']) for p in pairs]
    ai = [_arrays(p['ai']) for p in pairs]
    days = max(len(b[0]) for b in base)
    if any(len(b[0]) != days for b in base + ai):
        raise ValueError(f'{name}: unequal period lengths')
    arr = dict(
        base_opened=np.stack([b[0] for b in base]), base_comp=np.stack([b[1] for b in base]),
        base_profit=np.stack([b[2] for b in base]),
        ai_opened=np.stack([a[0] for a in ai]), ai_comp=np.stack([a[1] for a in ai]),
        ai_profit=np.stack([a[2] for a in ai]),
        court=np.array([str(p['court']) for p in pairs]),
        repeat=np.array([int(p.get('repeat', 0)) for p in pairs]),
        changes=np.array([bool(p['changes']) for p in pairs]),
        world=np.array([str(p.get('behaviour_world', '')) for p in pairs]),
        stored_delta=np.array([float(p['delta']) for p in pairs]),
    )
    # Exactness check: own objective re-scored from components must equal the stored delta.
    own = (period_scores(arr['ai_comp'], arr['ai_opened'], [objective])[:, 0]
           - period_scores(arr['base_comp'], arr['base_opened'], [objective])[:, 0])
    arr['recheck_max_abs_err'] = np.array(float(np.max(np.abs(own - arr['stored_delta']))))
    arr['objective_json'] = np.array(json.dumps(objective))
    arr['replica'] = np.array(int(summary.get('replica', -1)))
    res = summary.get('results', {})
    arr['summary_mean_delta_stable'] = np.array(float(res.get('stable', {}).get('mean_delta', np.nan)))
    arr['summary_mean_delta_changed'] = np.array(float(res.get('changed', {}).get('mean_delta', np.nan)))
    tmp = cache_dir / f'.{name}.tmp.npz'
    np.savez_compressed(tmp, **arr)
    os.replace(tmp, out)
    return name, f'ok err={float(arr["recheck_max_abs_err"]):.2e}', time.time() - t0


def finished_jobs(experiment):
    experiment = Path(experiment)
    jobs = []
    for obj in sorted(p for p in experiment.iterdir() if p.is_dir()):
        for rep in sorted(obj.glob('rep_*')):
            # Drive makes 'rep_2 (1)' when two runtimes wrote the same job; keep the canonical folder only.
            if not re.fullmatch(r'rep_\d+', rep.name):
                continue
            if (rep / 'paired_test.json').exists() and (rep / 'summary.json').exists():
                jobs.append(rep)
    return jobs


def default_workers():
    cpus = os.cpu_count() or 2
    try:
        mem_gb = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2**30
    except (ValueError, OSError):  # pragma: no cover
        mem_gb = 12
    # A 166 MB paired_test.json takes roughly 2-3 GB of Python objects while parsed.
    return max(1, min(cpus, int(mem_gb // 3.5)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('experiment', help='experiment folder, e.g. .../extended_main_20261108_001')
    ap.add_argument('cache', help='folder for the small per-job .npz files')
    ap.add_argument('--workers', type=int, default=0)
    a = ap.parse_args(argv)
    cache = Path(a.cache); cache.mkdir(parents=True, exist_ok=True)
    jobs = finished_jobs(a.experiment)
    todo = [j for j in jobs if not (cache / f'{j.parent.name}__{j.name}.npz').exists()]
    workers = a.workers or default_workers()
    print(f'{len(jobs)} finished jobs, {len(jobs) - len(todo)} cached, {len(todo)} to extract with {workers} workers', flush=True)
    t0 = time.time(); errors = []
    with ProcessPoolExecutor(workers) as pool:
        futures = {pool.submit(extract_job, j, cache): j for j in todo}
        for i, f in enumerate(as_completed(futures), 1):
            try:
                name, status, sec = f.result()
                print(f'[{i}/{len(todo)}] {name} {status} {sec:.0f}s (elapsed {time.time() - t0:.0f}s)', flush=True)
            except Exception as e:  # keep going; report at the end
                errors.append((str(futures[f]), repr(e)))
                print(f'[{i}/{len(todo)}] FAILED {futures[f]}: {e!r}', flush=True)
    if errors:
        print(f'{len(errors)} jobs failed:', *errors, sep='\n')
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
