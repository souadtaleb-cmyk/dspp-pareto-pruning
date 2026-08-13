"""
Stream registry + experiment orchestration for the DSPP pipeline.
Builds the 8 streams (5 original + 3 real-world additions), runs all
methods x seeds, and assembles the paper-style result tables + stats tests.
"""
import itertools
import time

import numpy as np
import pandas as pd
from scipy import stats as sps

from river import datasets
from river.datasets import synth

from .dspp_core import DSPPRunner, run_baseline

# ---------------------------------------------------------------------------
# Stream builders
# ---------------------------------------------------------------------------
def sea_recurring(seed, n_switches=6, segment_len=800):
    """Alternates SEA variant 0 / variant 1 every `segment_len` instances,
    `n_switches` times total -- a controlled synthetic analogue of recurring drift."""
    rng_seed = seed
    for i in range(n_switches):
        variant = i % 2
        gen = synth.SEA(variant=variant, seed=rng_seed + i)
        it = iter(gen)
        for _ in range(segment_len):
            yield next(it)


def sea_abrupt(seed):
    base = synth.SEA(variant=0, seed=seed)
    drift = synth.SEA(variant=1, seed=seed)
    # width=50 relative to position=2500 gives a sharp (near-abrupt) transition
    # while avoiding the float-overflow river hits at width=1.
    return iter(synth.ConceptDriftStream(stream=base, drift_stream=drift,
                                          position=2500, width=50, seed=seed))


def hyperplane_gradual(seed):
    return iter(synth.Hyperplane(seed=seed, n_features=10, n_drift_features=2,
                                  mag_change=0.001, noise_percentage=0.05, sigma=0.1))


def rbf_incremental(seed):
    return iter(synth.RandomRBFDrift(seed_model=seed, seed_sample=seed,
                                      n_classes=2, n_features=10, n_centroids=30,
                                      change_speed=0.05, n_drift_centroids=10))


def electricity(seed):
    # real, order-fixed -> seed only affects model-side randomness, not instance order
    return iter(datasets.Elec2())


def creditcard(seed):
    return iter(datasets.CreditCard())


def http_kdd(seed):
    return iter(datasets.HTTP())


def phishing(seed):
    return iter(datasets.Phishing())


STREAM_REGISTRY = {
    # name: (builder(seed) -> iterator, max_instances or None, is_real)
    "Electricity":        (electricity,        45312,  True),
    "SEA-abrupt":         (sea_abrupt,         5000,   False),
    "SEA-recurring":      (sea_recurring,      4800,   False),
    "Hyperplane-gradual": (hyperplane_gradual, 5000,   False),
    "RBF-incremental":    (rbf_incremental,    5000,   False),
    "CreditCard":         (creditcard,         284807, True),
    "HTTP":               (http_kdd,           567498, True),
    "Phishing":           (phishing,           1250,   True),
}

DSPP_CONFIGS = [
    # (label, variant, mode)
    ("DSPP-Lite",              "lite",     "dynamic"),
    ("DSPP-Adaptive",          "adaptive", "dynamic"),
    ("StaticSPP-Lite",         "lite",     "static"),
    ("StaticSPP-Adaptive",     "adaptive", "static"),
    ("StaticSPPFullSearch-Lite",     "lite",     "static_fullsearch"),
    ("StaticSPPFullSearch-Adaptive", "adaptive", "static_fullsearch"),
]
BASELINES = ["OB", "LB", "ARF", "SRP"]


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------
def run_all(streams=None, seeds=range(10), max_instances_cap=None, verbose=True):
    """Runs every DSPP config + every baseline on every stream x seed.
    Returns a tidy pandas.DataFrame, one row per (stream, method, seed)."""
    streams = streams or list(STREAM_REGISTRY.keys())
    rows = []
    for stream_name in streams:
        builder, default_cap, is_real = STREAM_REGISTRY[stream_name]
        cap = max_instances_cap if max_instances_cap is not None else default_cap
        for seed in seeds:
            for label, variant, mode in DSPP_CONFIGS:
                t0 = time.time()
                runner = DSPPRunner(variant=variant, mode=mode, seed=seed)
                res = runner.run(builder(seed), max_instances=cap)
                res.update(stream=stream_name, method=label, seed=seed,
                            wall_time_s=time.time() - t0)
                rows.append(res)
                if verbose:
                    print(f"[{stream_name}] {label} seed={seed} acc={res['accuracy']:.4f} "
                          f"({res['wall_time_s']:.1f}s)")
            for name in BASELINES:
                t0 = time.time()
                res = run_baseline(name, builder(seed), seed=seed, max_instances=cap)
                res.update(stream=stream_name, method=name, seed=seed,
                            wall_time_s=time.time() - t0)
                rows.append(res)
                if verbose:
                    print(f"[{stream_name}] {name} seed={seed} acc={res['accuracy']:.4f} "
                          f"({res['wall_time_s']:.1f}s)")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table 1: mean prequential accuracy by stream x method
# ---------------------------------------------------------------------------
def table1_accuracy(df):
    return df.pivot_table(index="stream", columns="method", values="accuracy", aggfunc="mean")


def table1_kappa(df):
    return df.pivot_table(index="stream", columns="method", values="kappa", aggfunc="mean")


# ---------------------------------------------------------------------------
# Friedman test across methods (streams as blocks, mean accuracy per stream)
# ---------------------------------------------------------------------------
def friedman_test(df, methods):
    pivot = df[df.method.isin(methods)].pivot_table(
        index="stream", columns="method", values="accuracy", aggfunc="mean")
    pivot = pivot.dropna()
    stat, p = sps.friedmanchisquare(*[pivot[m].values for m in methods])
    return {"chi2": stat, "p": p, "n_blocks": len(pivot), "methods": methods}


# ---------------------------------------------------------------------------
# Table 2: paired Wilcoxon tests (Test1 / Test2 / Test3), dataset x seed pairs
# ---------------------------------------------------------------------------
def _paired_values(df, method_a, method_b):
    a = df[df.method == method_a].set_index(["stream", "seed"])["accuracy"]
    b = df[df.method == method_b].set_index(["stream", "seed"])["accuracy"]
    common = a.index.intersection(b.index)
    return a.loc[common].values, b.loc[common].values, common


def wilcoxon_paired(df, method_a, method_b, alternative="greater"):
    a, b, idx = _paired_values(df, method_a, method_b)
    if len(a) == 0:
        return None
    diff = a - b
    wins = int((diff > 0).sum())
    n = len(diff)
    try:
        w_stat, p = sps.wilcoxon(a, b, alternative=alternative, zero_method="wilcox")
    except ValueError:
        w_stat, p = float("nan"), float("nan")
    return {
        "comparison": f"{method_a} vs {method_b}",
        "n_pairs": n,
        "wins": wins,
        "mean_gain": float(diff.mean()),
        "W": w_stat,
        "p": p,
    }


def table2_paired_tests(df):
    results = []
    for variant, dspp, static, staticfs in [
        ("Lite", "DSPP-Lite", "StaticSPP-Lite", "StaticSPPFullSearch-Lite"),
        ("Adaptive", "DSPP-Adaptive", "StaticSPP-Adaptive", "StaticSPPFullSearch-Adaptive"),
    ]:
        r1 = wilcoxon_paired(df, dspp, static)
        r1["test"] = "Test1 (vs StaticSPP)"
        r1["variant"] = variant
        r2 = wilcoxon_paired(df, dspp, staticfs)
        r2["test"] = "Test2 (vs StaticSPPFullSearch)"
        r2["variant"] = variant
        r3 = wilcoxon_paired(df, dspp, "SRP")
        r3["test"] = "Test3 (vs SRP)"
        r3["variant"] = variant
        results.extend([r1, r2, r3])
    out = pd.DataFrame(results)
    return out[["test", "variant", "comparison", "n_pairs", "wins", "mean_gain", "W", "p"]]


# ---------------------------------------------------------------------------
# Table 3: per-stream Test2 gain decomposition
# ---------------------------------------------------------------------------
def table3_per_stream_gain(df, variant="Lite"):
    dspp = f"DSPP-{variant}"
    staticfs = f"StaticSPPFullSearch-{variant}"
    a = df[df.method == dspp].groupby("stream")["accuracy"].mean()
    b = df[df.method == staticfs].groupby("stream")["accuracy"].mean()
    return (a - b).rename(f"gain_{variant}").to_frame()


# ---------------------------------------------------------------------------
# Table 4: efficiency (macro-averaged over streams, mean over seeds)
# ---------------------------------------------------------------------------
def table4_efficiency(df):
    return df.groupby("method").agg(
        accuracy=("accuracy", "mean"),
        latency_us=("latency_us", "mean"),
        pool_memory_kb=("pool_memory_kb", "mean"),
        active_memory_kb=("active_memory_kb", "mean"),
        mean_active_size=("mean_active_size", "mean"),
    ).round(3)
