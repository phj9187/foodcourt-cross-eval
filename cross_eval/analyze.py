"""Step 2: score every trained AI under every objective and summarise.

Unit of uncertainty: a replica (independently trained AI). For each job the
value under objective k is the court-level mean of AI - no-recommendation, the
same quantity the main study reports, recomputed from daily components.

Outputs (in --out):
  cross_matrix_<split>.csv     trained objective x evaluation objective, mean dF over replicas
  cross_se_<split>.csv         standard error over replicas
  self_vs_best_<split>.csv     per evaluation objective: is the AI trained on it the best?
  generalist_<split>.csv       per trained objective: how strong across all evaluation objectives
  common_metrics_<split>.csv   per trained objective: component and won deltas, Pareto flag
  checks.json                  recheck errors, job counts
  heatmap_<split>.png          column-normalised cross matrix
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .scoring import COMPONENTS, period_component_means, period_scores

SPLITS = {'stable': False, 'changed': True}


def load(cache):
    jobs = []
    for f in sorted(Path(cache).glob('*.npz')):
        z = np.load(f, allow_pickle=False)
        d = {k: z[k] for k in z.files}
        d['objective'] = json.loads(str(d.pop('objective_json')))
        d['name'] = f.stem
        jobs.append(d)
    if not jobs:
        raise SystemExit(f'no .npz files in {cache}')
    return jobs


def objectives_of(jobs):
    seen = {}
    for j in jobs:
        seen.setdefault(j['objective']['id'], j['objective'])
    def key(i):  # lhs first in order, boundary objectives last
        return (not i.startswith('lhs_'), i)
    return [seen[i] for i in sorted(seen, key=key)]


def court_mean(values, courts):
    """values (P, ...) -> mean over courts of the per-court mean."""
    uniq = np.unique(courts)
    return np.mean([values[courts == c].mean(0) for c in uniq], axis=0)


def job_values(job, objectives, changes):
    sel = job['changes'] == changes
    if not sel.any():
        return None
    ai = period_scores(job['ai_comp'][sel], job['ai_opened'][sel], objectives)
    base = period_scores(job['base_comp'][sel], job['base_opened'][sel], objectives)
    courts = job['court'][sel]
    delta = court_mean(ai - base, courts)                                          # K
    comp_delta = court_mean(period_component_means(job['ai_comp'][sel], job['ai_opened'][sel])
                            - period_component_means(job['base_comp'][sel], job['base_opened'][sel]), courts)
    om = job['base_opened'][sel]
    won = np.where(om.sum(1) > 0, ((job['ai_profit'][sel] - job['base_profit'][sel]) * om).sum(1) / np.maximum(om.sum(1), 1), 0.0)
    return delta, comp_delta, float(court_mean(won, courts))


def pareto(points):
    """points (n, 2), both to maximise -> boolean mask of non-dominated rows."""
    keep = np.ones(len(points), bool)
    for i, p in enumerate(points):
        others = np.delete(points, i, 0)
        if np.any(np.all(others >= p, 1) & np.any(others > p, 1)):
            keep[i] = False
    return keep


def write_csv(path, header, rows):
    with open(path, 'w', encoding='utf-8-sig') as f:
        f.write(','.join(header) + '\n')
        for r in rows:
            f.write(','.join(f'{v:.6g}' if isinstance(v, float) else str(v) for v in r) + '\n')


def analyse(cache, out, split):
    jobs = load(cache)
    objs = objectives_of(jobs)
    ids = [o['id'] for o in objs]
    K = len(objs)
    per = {i: [] for i in ids}
    for j in jobs:
        v = job_values(j, objs, SPLITS[split])
        if v is not None:
            per[j['objective']['id']].append(v)
    # Second check: diagonal court-mean dF must equal summary.json's mean_delta.
    diag_err = []
    for j in jobs:
        v = job_values(j, objs, SPLITS[split])
        ref = float(j[f'summary_mean_delta_{split}'])
        if v is not None and np.isfinite(ref):
            diag_err.append(abs(float(v[0][ids.index(j['objective']['id'])]) - ref))
    trained = [i for i in ids if per[i]]
    M = np.array([np.mean([v[0] for v in per[i]], 0) for i in trained])            # T,K
    S = np.array([np.std([v[0] for v in per[i]], 0, ddof=1) / np.sqrt(len(per[i])) if len(per[i]) > 1
                  else np.full(K, np.nan) for i in trained])
    n = {i: len(per[i]) for i in trained}
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    write_csv(out / f'cross_matrix_{split}.csv', ['trained\\evaluated', *ids], [[t, *map(float, M[a])] for a, t in enumerate(trained)])
    write_csv(out / f'cross_se_{split}.csv', ['trained\\evaluated', *ids], [[t, *map(float, S[a])] for a, t in enumerate(trained)])

    # 1) For each evaluation objective: is the self-trained AI the best?
    rows = []
    for k, eid in enumerate(ids):
        if eid not in trained:
            continue
        col = M[:, k]; order = np.argsort(-col)
        self_row = trained.index(eid)
        best = order[0] if order[0] != self_row else order[1]
        gap = col[self_row] - col[best]
        se = np.sqrt(S[self_row, k] ** 2 + S[best, k] ** 2)
        rows.append([eid, int(np.where(order == self_row)[0][0]) + 1, len(trained), float(col[self_row]),
                     trained[best], float(col[best]), float(gap), float(gap / se) if se > 0 else float('nan'),
                     'self_best' if order[0] == self_row else ('other_better_2se' if gap < -2 * se else 'other_ahead_within_noise')])
    write_csv(out / f'self_vs_best_{split}.csv',
              ['evaluated', 'self_rank', 'n_trained', 'self_dF', 'best_other', 'best_other_dF', 'self_minus_other', 'z', 'verdict'], rows)

    # 2) Generalist strength: z-score each column across trained AIs, then average.
    Z = (M - M.mean(0)) / np.where(M.std(0) > 0, M.std(0), 1)
    top3 = (np.argsort(np.argsort(-M, 0), 0) < 3).sum(1)
    beats_self = []
    for a, t in enumerate(trained):
        cnt = 0
        for k, eid in enumerate(ids):
            if eid in trained and eid != t:
                s = trained.index(eid)
                se = np.sqrt(S[a, k] ** 2 + S[s, k] ** 2)
                cnt += bool(M[a, k] - M[s, k] > 2 * se)
        beats_self.append(cnt)
    order = np.argsort(-Z.mean(1))
    write_csv(out / f'generalist_{split}.csv',
              ['trained', 'replicas', 'mean_z_over_all_objectives', 'top3_count', 'beats_self_trained_by_2se_count', 'positive_dF_count'],
              [[trained[a], n[trained[a]], float(Z[a].mean()), int(top3[a]), beats_self[a], int((M[a] > 0).sum())] for a in order])

    # 3) Common metrics and Pareto front on (customer with equal weights, profit won).
    comp = np.array([np.mean([v[1] for v in per[i]], 0) for i in trained])
    won = np.array([np.mean([v[2] for v in per[i]]) for i in trained])
    won_se = np.array([np.std([v[2] for v in per[i]], ddof=1) / np.sqrt(len(per[i])) if len(per[i]) > 1 else np.nan for i in trained])
    cust_eq = comp[:, :3].mean(1)
    front = pareto(np.column_stack([cust_eq, won]))
    obj_by = {o['id']: o for o in objs}
    write_csv(out / f'common_metrics_{split}.csv',
              ['trained', 'lambda', 'q', 'a', 'b', 'c', *[f'd_{c}' for c in COMPONENTS], 'd_customer_equal_weights', 'd_profit_won_per_court_day', 'se_profit_won', 'pareto'],
              [[t, obj_by[t]['customer_share'], obj_by[t]['profit_share'], obj_by[t]['taste'], obj_by[t]['affordability'], obj_by[t]['waiting'],
                *map(float, comp[a]), float(cust_eq[a]), float(won[a]), float(won_se[a]), bool(front[a])] for a, t in enumerate(trained)])
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        C = M / np.where(np.abs(M).max(0) > 0, np.abs(M).max(0), 1)
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(C, cmap='RdBu', vmin=-1, vmax=1)
        ax.set_xticks(range(K)); ax.set_xticklabels(ids, rotation=90, fontsize=7)
        ax.set_yticks(range(len(trained))); ax.set_yticklabels(trained, fontsize=7)
        for k, eid in enumerate(ids):
            if eid in trained:
                ax.add_patch(plt.Rectangle((k - .5, trained.index(eid) - .5), 1, 1, fill=False, lw=1.2))
        ax.set_xlabel('evaluated with objective'); ax.set_ylabel('trained with objective')
        ax.set_title(f'Cross-evaluation ({split}): AI - none, each column scaled by its max |dF|; boxes = self')
        fig.colorbar(im, shrink=.6); fig.tight_layout(); fig.savefig(out / f'heatmap_{split}.png', dpi=130); plt.close(fig)
    except Exception as e:  # plotting is optional
        print('heatmap skipped:', e)
    return dict(split=split, trained=len(trained), jobs=sum(n.values()),
                summary_mean_delta_max_abs_err=max(diag_err) if diag_err else None,
                self_best=sum(r[-1] == 'self_best' for r in rows),
                other_better_2se=sum(r[-1] == 'other_better_2se' for r in rows),
                top_generalists=[trained[a] for a in order[:5]],
                pareto=[t for a, t in enumerate(trained) if front[a]])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('cache'); ap.add_argument('out')
    a = ap.parse_args(argv)
    jobs = load(a.cache)
    errs = {j['name']: float(j['recheck_max_abs_err']) for j in jobs}
    report = dict(jobs=len(jobs), recheck_max_abs_err=max(errs.values()),
                  recheck_failed=[k for k, v in errs.items() if v > 1e-9],
                  splits=[analyse(a.cache, a.out, s) for s in SPLITS])
    Path(a.out, 'checks.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
