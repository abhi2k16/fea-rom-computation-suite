# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
ensemble_uq.py -- ensemble uncertainty quantification for trainable corrections: train an NN or RBF correction N
times from different random seeds or mini-batch orderings and report the ensemble mean and standard deviation of its
predictions, a cheap, well-understood epistemic-uncertainty proxy. The ensemble spread tends to correlate spatially
with regions of higher true prediction error; this module provides the aggregation machinery, not the correlation
check itself (which is problem-specific).

A thin wrapper around any training loop, not a new algorithm. Works with ANY model exposing `fit(X, y) -> self` /
`predict(X) -> array` (the duck-typed `ReducedForceModel` protocol of nonlinear_rom.py) via a caller-supplied
`model_factory(seed) -> fresh model instance`. A fresh instance is created per seed, rather than re-fitting one
shared instance, because a real ensemble member (for example an NN with random weight initialisation) generally
needs a FRESH object per seed, not a re-run of `.fit()` on the same one.

Independent of torch at the module level: `EnsembleUQ` never imports torch; only the convenience constructor
`neural_surrogate_ensemble()` touches `NeuralSurrogate`, which is itself torch-optional at import time
(nonlinear_rom.py's `_HAS_TORCH` pattern). Constructing a `NeuralSurrogate` is what requires torch, not importing
the module that defines it."""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

from .nonlinear_rom import NeuralSurrogate


class EnsembleUQ:
    """Trains `n_members` independently-seeded copies of the same
    model family and reports mean/std across the ensemble at predict
    time -- a cheap, well-understood epistemic-uncertainty proxy (see
    this module's own docstring).

    Parameters
    ----------
    model_factory : callable, seed -> a fresh model instance exposing
        fit(X, y, **kwargs) -> self and predict(X) -> array.
    n_members : int, default 5
    seeds : list of int, optional
        Defaults to range(n_members). Must have length n_members if
        supplied explicitly.
    """

    def __init__(self, model_factory, n_members=5, seeds=None):
        if seeds is None:
            seeds = list(range(n_members))
        elif len(seeds) != n_members:
            raise ValueError(
                f"seeds has {len(seeds)} entries, expected n_members={n_members}")
        self.model_factory = model_factory
        self.n_members = n_members
        self.seeds = seeds
        self.members = []

    def fit(self, X, y, **fit_kwargs):
        """Fits `n_members` fresh model instances, one per seed, each
        on the SAME (X, y) training data -- the ensemble's spread then
        reflects only the model family's own training stochasticity
        (random init, mini-batch order, etc.), not a data-sampling
        effect (bootstrap/bagging ensembles are a different, also
        legitimate, technique -- out of scope here, which is
        specifically about seed variation)."""
        self.members = []
        for seed in self.seeds:
            model = self.model_factory(seed)
            model.fit(X, y, **fit_kwargs)
            self.members.append(model)
        return self

    def predict_all(self, X):
        """Returns (n_members, n_points, n_out) -- every member's own
        prediction, un-aggregated (a caller wanting the raw per-member
        spread rather than just mean/std uses this directly)."""
        if not self.members:
            raise RuntimeError("EnsembleUQ.predict_all() called before fit()")
        preds = [np.atleast_2d(np.asarray(m.predict(X), dtype=float)) for m in self.members]
        return np.stack(preds, axis=0)

    def predict_mean_std(self, X):
        """Returns (mean, std), each (n_points, n_out) -- the ensemble
        mean prediction and its per-point, per-output-component
        standard deviation across members (the epistemic-uncertainty
        proxy itself)."""
        preds = self.predict_all(X)
        return preds.mean(axis=0), preds.std(axis=0)


def neural_surrogate_ensemble(n_modes, q_samples, F_samples, n_members=5, seeds=None,
                               **surrogate_kwargs):
    """Convenience constructor: builds and fits an EnsembleUQ of
    `NeuralSurrogate` instances (rom_engine's own trainable, seed-
    sensitive `ReducedForceModel`) -- the concrete use
    case for an ensemble (any trainable NN correction
    is seed-dependent). Requires torch
    (NeuralSurrogate's own constructor does; this function itself adds
    no additional torch dependency beyond that).

    surrogate_kwargs are forwarded to every NeuralSurrogate(...)
    construction (hidden_sizes, activation, lr, n_epochs, ...) except
    `seed`, which this function supplies itself, one per ensemble
    member.

    Returns a fitted EnsembleUQ."""
    def factory(seed):
        return NeuralSurrogate(n_modes=n_modes, seed=seed, **surrogate_kwargs)

    ensemble = EnsembleUQ(factory, n_members=n_members, seeds=seeds)
    ensemble.fit(q_samples, F_samples)
    return ensemble
