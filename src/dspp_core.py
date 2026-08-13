"""
DSPP (Dynamic Streaming Pareto Pruning) — core pipeline.
Implements DSPP-Lite, DSPP-Adaptive, ablations (StaticSPP, StaticSPPFullSearch),
and 4 baselines (OB, LB, ARF, SRP), evaluated over multiple streams/seeds with
accuracy, Cohen's Kappa, latency, memory, plus Friedman/Wilcoxon tests.
"""
import itertools
import math
import pickle
import random
import time
from collections import deque

import numpy as np
from river import ensemble, forest, tree, drift, metrics as river_metrics
from river.datasets import synth

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WINDOW_SIZE = 200
MAX_POOL = 12
EXHAUSTIVE_THRESHOLD = 12
RANDOM_SUBSET_SAMPLES = 200
GAMMA = 0.01          # recency-weight decay
EVICT_THRESHOLD = 0.5  # weighted-error eviction threshold
N_INJECT = 2           # new base learners injected on drift
GRACE_INSTANCES = 30   # min instances a model needs before scoring counts fully


# ---------------------------------------------------------------------------
# Recency-weighted tracking of error / agreement within a window
# ---------------------------------------------------------------------------
class WindowBuffer:
    """Buffers (x, y, {model_id: pred}) for the current window."""

    def __init__(self):
        self.records = []  # list of (x, y, preds_dict)

    def add(self, x, y, preds):
        self.records.append((x, y, preds))

    def clear(self):
        self.records = []

    def __len__(self):
        return len(self.records)

    def weighted_error(self, model_id):
        n = len(self.records)
        if n == 0:
            return 1.0
        num, den = 0.0, 0.0
        for k, (x, y, preds) in enumerate(self.records):
            w = math.exp(-GAMMA * (n - 1 - k))
            den += w
            p = preds.get(model_id)
            if p is None or p != y:
                num += w
        return num / den if den > 0 else 1.0

    def pairwise_agreement(self, model_id, other_ids):
        if not other_ids:
            return 0.0
        n = len(self.records)
        if n == 0:
            return 0.0
        agree_sum = 0.0
        count = 0
        for oid in other_ids:
            agree = 0
            for (x, y, preds) in self.records:
                pi, pj = preds.get(model_id), preds.get(oid)
                if pi is not None and pj is not None and pi == pj:
                    agree += 1
            agree_sum += agree / n
            count += 1
        return agree_sum / count if count else 0.0


# ---------------------------------------------------------------------------
# Pareto front + knee point
# ---------------------------------------------------------------------------
def pareto_front(candidates):
    """candidates: list of (id, f1, f2). Returns list of ids on the non-dominated
    front (minimizing both f1 and f2), via O(n log n) skyline sweep."""
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda c: c[1])
    front = []
    running_min_f2 = float("inf")
    for cid, f1, f2 in ordered:
        if f2 < running_min_f2:
            front.append((cid, f1, f2))
            running_min_f2 = f2
    return front


def knee_point(front):
    """Pick the point on the front with max perpendicular distance to the chord
    connecting the two extreme anchor points (min-f1, min-f2)."""
    if len(front) == 1:
        return front[0][0]
    pts = sorted(front, key=lambda c: c[1])
    p1 = pts[0]   # min f1
    p2 = pts[-1]  # max f1 (== min f2 among front, since front sorted by f1)
    x1, y1 = p1[1], p1[2]
    x2, y2 = p2[1], p2[2]
    dx, dy = x2 - x1, y2 - y1
    norm = math.hypot(dx, dy)
    if norm == 0:
        return p1[0]
    best_id, best_dist = p1[0], -1.0
    for cid, f1, f2 in pts:
        dist = abs(dy * f1 - dx * f2 + x2 * y1 - y2 * x1) / norm
        if dist > best_dist:
            best_dist, best_id = dist, cid
    return best_id


def generate_candidate_subsets(model_ids, rng):
    """Enumerate all subsets of size >=2 if |pool| <= threshold, else sample."""
    m = len(model_ids)
    if m < 2:
        return [tuple(model_ids)] if m else []
    if m <= EXHAUSTIVE_THRESHOLD:
        subsets = []
        for r in range(2, m + 1):
            subsets.extend(itertools.combinations(model_ids, r))
        return subsets
    subsets = []
    for _ in range(RANDOM_SUBSET_SAMPLES):
        size = rng.randint(2, m)
        subsets.append(tuple(rng.sample(model_ids, size)))
    return subsets


def eleven_point_scalarization_subset(model_ids, buf):
    """Static baseline (StaticSPP): grid of 11 scalarization weights
    lambda in {0, 0.1, ..., 1.0}, pick best full-pool-derived subset per lambda,
    then knee-select among those 11 (mirrors the original static formulation)."""
    if len(model_ids) < 2:
        return tuple(model_ids)
    subsets = generate_candidate_subsets(model_ids, random.Random(0))
    scored = []
    for s in subsets:
        f1 = float(np.mean([buf.weighted_error(mid) for mid in s]))
        f2 = float(np.mean([buf.pairwise_agreement(mid, [o for o in s if o != mid]) for mid in s]))
        scored.append((s, f1, f2))
    lambdas = [round(i * 0.1, 1) for i in range(11)]
    picks = []
    for lam in lambdas:
        best = min(scored, key=lambda t: lam * t[1] + (1 - lam) * t[2])
        picks.append(best)
    front_candidates = [(s, f1, f2) for (s, f1, f2) in picks]
    front = pareto_front([(s, f1, f2) for (s, f1, f2) in front_candidates])
    if not front:
        front = [(picks[0][0], picks[0][1], picks[0][2])]
    return knee_point(front)


def full_search_subset(model_ids, buf, rng):
    subsets = generate_candidate_subsets(model_ids, rng)
    if not subsets:
        return tuple(model_ids)
    scored = []
    for s in subsets:
        f1 = float(np.mean([buf.weighted_error(mid) for mid in s]))
        f2 = float(np.mean([buf.pairwise_agreement(mid, [o for o in s if o != mid]) for mid in s]))
        scored.append((s, f1, f2))
    front = pareto_front(scored)
    if not front:
        front = [scored[0]]
    return knee_point(front)


# ---------------------------------------------------------------------------
# Base-learner factories
# ---------------------------------------------------------------------------
def make_hoeffding_tree(seed):
    # HoeffdingTreeClassifier has no internal randomness; the per-model seed
    # instead drives that model's private Poisson(1) online-bagging RNG
    # (see DSPPRunner._add_model / _learn_pool for DSPP-Lite diversity injection).
    return tree.HoeffdingTreeClassifier()


def make_hat(seed):
    return tree.HoeffdingAdaptiveTreeClassifier(seed=seed, bootstrap_sampling=True)


# ---------------------------------------------------------------------------
# DSPP engine (covers DSPP-Lite / DSPP-Adaptive / StaticSPP / StaticSPPFullSearch)
# ---------------------------------------------------------------------------
class DSPPRunner:
    """
    variant: 'lite' | 'adaptive'
    mode:    'dynamic' (full DSPP), 'static' (StaticSPP, fixed pool + 11pt wrapper),
             'static_fullsearch' (StaticSPPFullSearch, fixed pool + combinatorial search)
    """

    def __init__(self, variant, mode, seed, init_pool_size=6):
        assert variant in ("lite", "adaptive")
        assert mode in ("dynamic", "static", "static_fullsearch")
        self.variant = variant
        self.mode = mode
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

        self.pool = {}          # model_id -> classifier
        self.pool_seen = {}     # model_id -> instances seen (grace tracking)
        self.pool_rng = {}      # model_id -> private np.random.RandomState for Poisson(1) bagging (lite only)
        self.next_id = 0
        for _ in range(init_pool_size):
            self._add_model()

        self.active = tuple(self.pool.keys())
        self.buf = WindowBuffer()
        self.adwin = drift.ADWIN()
        self.pending_drift = False

        # metrics
        self.acc_metric = river_metrics.Accuracy()
        self.kappa_metric = river_metrics.CohenKappa()
        self.n_seen = 0
        self.latencies = []  # seconds per prediction
        self.drift_count = 0
        self.active_size_history = []

    def _add_model(self):
        mid = self.next_id
        self.next_id += 1
        seed_i = 100 * self.seed + mid
        if self.variant == "lite":
            self.pool[mid] = make_hoeffding_tree(seed_i)
            self.pool_rng[mid] = np.random.RandomState(seed_i)
        else:
            self.pool[mid] = make_hat(seed_i)
        self.pool_seen[mid] = 0

    def _predict_pool(self, x):
        preds = {}
        for mid, model in self.pool.items():
            try:
                preds[mid] = model.predict_one(x)
            except Exception:
                preds[mid] = None
        return preds

    def _vote(self, preds_active):
        preds_active = {k: v for k, v in preds_active.items() if v is not None}
        if not preds_active:
            return None
        if self.variant == "lite":
            vals, counts = np.unique(list(preds_active.values()), return_counts=True)
            return vals[np.argmax(counts)]
        # adaptive: error-weighted vote
        weights = {}
        for mid in preds_active:
            e = self.buf.weighted_error(mid) if len(self.buf) else 0.5
            weights[mid] = max(1e-3, 1 - e)
        tally = {}
        for mid, p in preds_active.items():
            tally[p] = tally.get(p, 0.0) + weights[mid]
        return max(tally, key=tally.get)

    def _select_active_subset(self):
        model_ids = list(self.pool.keys())
        if len(model_ids) < 2 or len(self.buf) == 0:
            return tuple(model_ids)
        if self.mode == "static":
            return eleven_point_scalarization_subset(model_ids, self.buf)
        # dynamic and static_fullsearch both use combinatorial search + knee
        return full_search_subset(model_ids, self.buf, self.rng)

    def _adapt_pool(self):
        if self.mode != "dynamic":
            return
        if not self.pending_drift:
            return
        self.pending_drift = False
        self.drift_count += 1
        for _ in range(N_INJECT):
            self._add_model()
        # eviction by weighted error threshold
        evictable = [mid for mid in self.pool if self.buf.weighted_error(mid) > EVICT_THRESHOLD]
        for mid in evictable:
            if len(self.pool) > 2:
                del self.pool[mid]
                self.pool_seen.pop(mid, None)
                self.pool_rng.pop(mid, None)
        # hard cap
        if len(self.pool) > MAX_POOL:
            ranked = sorted(self.pool.keys(), key=lambda mid: -self.buf.weighted_error(mid))
            n_excess = len(self.pool) - MAX_POOL
            for mid in ranked[:n_excess]:
                del self.pool[mid]
                self.pool_seen.pop(mid, None)
                self.pool_rng.pop(mid, None)

    def run(self, stream_iter, max_instances=None):
        for i, (x, y) in enumerate(stream_iter):
            if max_instances is not None and i >= max_instances:
                break
            t0 = time.perf_counter()
            preds_all = self._predict_pool(x)
            preds_active = {mid: preds_all.get(mid) for mid in self.active}
            y_pred = self._vote(preds_active)
            t1 = time.perf_counter()
            self.latencies.append(t1 - t0)

            if y_pred is not None:
                self.acc_metric.update(y, y_pred)
                self.kappa_metric.update(y, y_pred)

            self.buf.add(x, y, preds_all)
            for mid, model in self.pool.items():
                if self.variant == "lite":
                    # DSPP-Lite diversity injection: per-model online (Poisson-1)
                    # bagging, each model's own RNG (Oza & Russell, 2001).
                    k = self.pool_rng[mid].poisson(1)
                    for _ in range(k):
                        model.learn_one(x, y)
                else:
                    # DSPP-Adaptive: no external bagging (HAT already
                    # bootstrap-samples internally; stacking degrades accuracy,
                    # see paper Section 6.2).
                    model.learn_one(x, y)
                self.pool_seen[mid] += 1

            correct = 1 if y_pred == y else 0
            self.adwin.update(correct)
            if self.adwin.drift_detected:
                self.pending_drift = True

            self.n_seen += 1
            if self.n_seen % WINDOW_SIZE == 0:
                self._adapt_pool()
                self.active = self._select_active_subset()
                self.active_size_history.append(len(self.active))
                self.buf.clear()

        return self.results()

    def pool_memory_bytes(self):
        total = 0
        for m in self.pool.values():
            try:
                total += len(pickle.dumps(m))
            except Exception:
                pass
        return total

    def active_memory_bytes(self):
        total = 0
        for mid in self.active:
            m = self.pool.get(mid)
            if m is None:
                continue
            try:
                total += len(pickle.dumps(m))
            except Exception:
                pass
        return total

    def results(self):
        lat_us = float(np.mean(self.latencies) * 1e6) if self.latencies else float("nan")
        return {
            "accuracy": self.acc_metric.get(),
            "kappa": self.kappa_metric.get(),
            "latency_us": lat_us,
            "pool_memory_kb": self.pool_memory_bytes() / 1024.0,
            "active_memory_kb": self.active_memory_bytes() / 1024.0,
            "final_pool_size": len(self.pool),
            "mean_active_size": float(np.mean(self.active_size_history)) if self.active_size_history else len(self.active),
            "drift_count": self.drift_count,
        }


# ---------------------------------------------------------------------------
# Baseline wrapper
# ---------------------------------------------------------------------------
def make_baseline(name, seed):
    if name == "OB":
        return ensemble.ADWINBaggingClassifier(model=tree.HoeffdingTreeClassifier(), n_models=10, seed=seed)
    if name == "LB":
        return ensemble.LeveragingBaggingClassifier(model=tree.HoeffdingTreeClassifier(), n_models=10, seed=seed)
    if name == "ARF":
        return forest.ARFClassifier(n_models=10, seed=seed)
    if name == "SRP":
        return ensemble.SRPClassifier(model=tree.HoeffdingTreeClassifier(), n_models=10, seed=seed)
    raise ValueError(name)


def run_baseline(name, stream_iter, seed, max_instances=None):
    model = make_baseline(name, seed)
    acc = river_metrics.Accuracy()
    kappa = river_metrics.CohenKappa()
    latencies = []
    for i, (x, y) in enumerate(stream_iter):
        if max_instances is not None and i >= max_instances:
            break
        t0 = time.perf_counter()
        y_pred = model.predict_one(x)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)
        if y_pred is not None:
            acc.update(y, y_pred)
            kappa.update(y, y_pred)
        model.learn_one(x, y)
    try:
        mem_kb = len(pickle.dumps(model)) / 1024.0
    except Exception:
        mem_kb = float("nan")
    return {
        "accuracy": acc.get(),
        "kappa": kappa.get(),
        "latency_us": float(np.mean(latencies) * 1e6) if latencies else float("nan"),
        "pool_memory_kb": mem_kb,
        "active_memory_kb": mem_kb,
        "final_pool_size": 10,
        "mean_active_size": 10,
        "drift_count": None,
    }
