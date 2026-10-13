# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
contact_comparison.py -- one call that runs several reduced contact models on the same held-out inputs and reports
displacement, gap, contact-force and complementarity errors, size, selected elements, online time and the
effective speed-up that charges offline cost against the number of queries. It extends the idea of
`validation.compare_bases` to models whose output is a (displacement, multiplier, gap) triple.

    table = compare_contact_models(fom, {"svd dual": dict(predict=f1, offline_time=t1, size=(p, pl)),
                                         "nmf dual": dict(predict=f2, offline_time=t2, size=(p, pl))}, test_inputs)

`fom(x)` and each `predict(x)` return (u, lam, gap). Errors are the maximum over the test inputs of the
max-normalised errors of `contact_validation.contact_errors`. Nothing is ranked: read the rows.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import time
import numpy as np

from .contact_validation import contact_errors
from .validation import effective_speedup, break_even_queries

__all__ = ["compare_contact_models", "format_contact_comparison"]


def _timed(f, x):
    t0 = time.perf_counter()
    out = f(x)
    return out, time.perf_counter() - t0


def compare_contact_models(fom, variants, test_inputs, n_queries=None, repeats=1):
    """Run `fom` and every variant on `test_inputs`. variants : dict name -> dict(predict=callable, offline_time=0.0,
    size=None, n_elements=None). Returns dict name -> row with keys displacement, gap, force, complementarity,
    active_agreement (worst case over inputs; active_agreement is the minimum), t_online, t_fom (mean seconds per
    query), speedup, effective_speedup, break_even, size, n_elements, n_converged_inputs."""
    xs = list(test_inputs)
    refs, t_fom = [], []
    for x in xs:
        best = np.inf
        for _ in range(max(1, repeats)):
            ref, dt = _timed(fom, x)
            best = min(best, dt)
        refs.append(ref)
        t_fom.append(best)
    t_fom_mean = float(np.mean(t_fom))
    nq = len(xs) if n_queries is None else int(n_queries)
    table = {}
    for name, v in variants.items():
        errs, times = [], []
        for x, ref in zip(xs, refs):
            best, out = np.inf, None
            for _ in range(max(1, repeats)):
                out, dt = _timed(v["predict"], x)
                best = min(best, dt)
            times.append(best)
            errs.append(contact_errors(out[0], out[1], out[2], ref[0], ref[1], ref[2]))
        t_on = float(np.mean(times))
        off = float(v.get("offline_time", 0.0))
        table[name] = dict(
            displacement=max(e.displacement for e in errs), gap=max(e.gap for e in errs),
            force=max(e.force for e in errs), complementarity=max(e.complementarity for e in errs),
            active_agreement=min(e.active_agreement for e in errs),
            t_online=t_on, t_fom=t_fom_mean, speedup=t_fom_mean / t_on if t_on > 0 else float("inf"),
            effective_speedup=effective_speedup(nq, t_on, off, t_fom_mean),
            break_even=break_even_queries(t_on, off, t_fom_mean),
            size=v.get("size"), n_elements=v.get("n_elements"), offline_time=off)
    return table


def format_contact_comparison(table):
    """Plain-text table of `compare_contact_models` output."""
    head = (f"{'model':<18}{'size':>10}{'disp err':>11}{'gap err':>11}{'force err':>11}{'compl.':>10}"
            f"{'active':>8}{'t_on [ms]':>11}{'speedup':>9}{'eff.':>8}")
    rows = [head, "-" * len(head)]
    for name, r in table.items():
        size = "-" if r["size"] is None else (str(r["size"]) if np.isscalar(r["size"]) else "/".join(map(str, r["size"])))
        rows.append(f"{name:<18}{size:>10}{r['displacement']:>11.2e}{r['gap']:>11.2e}{r['force']:>11.2e}"
                    f"{r['complementarity']:>10.1e}{r['active_agreement']:>8.2f}{1e3 * r['t_online']:>11.3f}"
                    f"{r['speedup']:>9.1f}{r['effective_speedup']:>8.2f}")
    return "\n".join(rows)
