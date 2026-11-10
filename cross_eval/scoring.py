"""Re-scoring that mirrors kiosk_adaptive.design.Objective.score and study.period_score.

Daily score F = lambda * sum_k w_k * max(0, c_k)**power + (1 - lambda) * (q * margin - (1 - q) * waste)
Period score  = mean of F over opened days (0 when no day opened).
"""
import numpy as np

COMPONENTS = ('taste', 'affordability', 'waiting', 'margin', 'waste')


def daily_scores(comp, objectives):
    """comp (..., 5) -> scores (..., K) for K objectives (list of dicts from summary.json)."""
    w = np.array([[o['taste'], o['affordability'], o['waiting']] for o in objectives])          # K,3
    lam = np.array([o['customer_share'] for o in objectives])                                   # K
    q = np.array([o['profit_share'] for o in objectives])                                       # K
    power = np.array([o.get('power', 1.0) for o in objectives])                                 # K
    cust_parts = np.maximum(comp[..., None, :3], 0.0) ** power[:, None]                         # ...,K,3
    customer = (cust_parts * w).sum(-1)                                                         # ...,K
    store = q * comp[..., 3:4] - (1 - q) * comp[..., 4:5]                                       # ...,K
    return lam * customer + (1 - lam) * store


def period_scores(comp, opened, objectives):
    """comp (P, D, 5), opened (P, D) -> (P, K) mean over opened days."""
    s = daily_scores(comp, objectives)                                                          # P,D,K
    m = opened[..., None].astype(float)
    n = m.sum(1)                                                                                # P,1
    total = (s * m).sum(1)
    return np.where(n > 0, total / np.maximum(n, 1), 0.0)


def period_component_means(comp, opened):
    """comp (P, D, 5) -> (P, 5) mean over opened days."""
    m = opened[..., None].astype(float)
    n = m.sum(1)
    return np.where(n > 0, (comp * m).sum(1) / np.maximum(n, 1), 0.0)
