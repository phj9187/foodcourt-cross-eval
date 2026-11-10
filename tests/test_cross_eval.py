"""Synthetic end-to-end test: build fake job folders shaped like study.py output, run both steps."""
import json, sys, tempfile, unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cross_eval import extract, analyze
from cross_eval.scoring import COMPONENTS


def objective_score(o, c):  # literal copy of kiosk_adaptive.design.Objective.score (642cf41)
    cust = sum(w * max(0., c[k]) ** o['power'] for w, k in ((o['taste'], 'taste'), (o['affordability'], 'affordability'), (o['waiting'], 'waiting')))
    store = o['profit_share'] * c['margin'] - (1 - o['profit_share']) * c['waste']
    return o['customer_share'] * cust + (1 - o['customer_share']) * store


def make_rows(rng, days, shift):
    rows = []
    for d in range(days):
        opened = d % 7 != 6
        comp = {k: float(rng.normal(.3 + shift, .2)) for k in COMPONENTS} if opened else {k: 0. for k in COMPONENTS}
        rows.append(dict(day=d, opened=opened, components=comp, contribution_profit=float(rng.normal(1e5, 1e4)) if opened else 0.))
    return rows


def make_experiment(root, objectives, reps=3, courts=4, days=14):
    rng = np.random.default_rng(0)
    for oi, o in enumerate(objectives):
        for rep in range(reps):
            job = root / o['id'] / f'rep_{rep}'; job.mkdir(parents=True)
            pairs = []
            for c in range(courts):
                for changes in (False, True):
                    b = make_rows(rng, days, 0.); a = make_rows(rng, days, .01 * oi)
                    ps = lambda rows: float(np.mean([objective_score(o, r['components']) for r in rows if r['opened']]))
                    pairs.append(dict(court=f'test_{c:04d}', repeat=0, changes=changes, behaviour_world='w',
                                      delta=ps(a) - ps(b), baseline=b, ai=a))
            md = {ch: float(np.mean([np.mean([p['delta'] for p in pairs if p['court'] == f'test_{c:04d}' and p['changes'] == ch]) for c in range(courts)])) for ch in (False, True)}
            (job / 'paired_test.json').write_text(json.dumps(pairs))
            (job / 'summary.json').write_text(json.dumps(dict(objective=o, replica=rep,
                results=dict(stable=dict(mean_delta=md[False]), changed=dict(mean_delta=md[True])))))


class CrossEvalTest(unittest.TestCase):
    def test_end_to_end(self):
        objs = [dict(id=f'lhs_{i:04d}', taste=t, affordability=a, waiting=1 - t - a, customer_share=l, profit_share=q, power=1.0)
                for i, (t, a, l, q) in enumerate([(.2, .3, .1, .9), (.5, .1, .8, .3), (.3, .3, .5, .5)])]
        objs.append(dict(id='profit_only', taste=1/3, affordability=1/3, waiting=1/3, customer_share=0., profit_share=1., power=1.0))
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); make_experiment(d / 'exp', objs)
            (d / 'exp' / 'lhs_0000' / 'rep_9').mkdir()  # unfinished job must be skipped
            import shutil; shutil.copytree(d / 'exp' / 'lhs_0001' / 'rep_0', d / 'exp' / 'lhs_0001' / 'rep_0 (1)')  # Drive duplicate
            self.assertEqual(extract.main([str(d / 'exp'), str(d / 'cache'), '--workers', '2']), 0)
            self.assertEqual(len(list((d / 'cache').glob('*.npz'))), 12)
            self.assertEqual(extract.main([str(d / 'exp'), str(d / 'cache'), '--workers', '2']), 0)  # all cached
            analyze.main([str(d / 'cache'), str(d / 'out')])
            rep = json.loads((d / 'out' / 'checks.json').read_text())
            self.assertLess(rep['recheck_max_abs_err'], 1e-9)
            for s in rep['splits']:
                self.assertLess(s['summary_mean_delta_max_abs_err'], 1e-9)
            for name in ('cross_matrix_stable.csv', 'self_vs_best_changed.csv', 'generalist_stable.csv', 'common_metrics_stable.csv', 'heatmap_stable.png'):
                self.assertTrue((d / 'out' / name).exists(), name)


if __name__ == '__main__':
    unittest.main()
