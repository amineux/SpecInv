"""SpecInv -- a reference implementation of SC-Net (arXiv:2603.20602).

SC-Net solves an ill-posed linear inverse problem :math:`y = \\mathcal{K}f + \\varepsilon` by
learning a *spectral filter*: it projects the data onto the singular system of
:math:`\\mathcal{K}`, rescales each coefficient by a learned, bounded, pointwise factor
:math:`\\Psi_\\theta(y_n,\\sigma_n) \\in [0, C_\\Psi]`, and synthesises the result,

.. math::
    \\mathcal{R}_\\theta(y^\\delta) = \\sum_{n=1}^{N}
        \\Psi_\\theta(y_n, \\sigma_n)\\, \\frac{\\langle y^\\delta, u_n\\rangle}{\\sigma_n}\\, v_n .

Because the only learned object is a scalar function of a single spectral mode, the model
is interpretable in the same units as classical regularisation -- the learned
:math:`\\Psi_\\theta` overlays directly on the Tikhonov and TSVD filter functions -- and it
is defined independently of any grid.

Quick start
-----------
>>> import numpy as np
>>> from specinv import (
...     InverseProblemSuite, SCNet, SCNetConfig, SobolevPrior, TrainConfig,
...     power_law_operator, summarise_errors, train_scnet,
... )
>>> suite = InverseProblemSuite(power_law_operator(256, 1.5), SobolevPrior(1.5))
>>> net = SCNet(SCNetConfig())
>>> history = train_scnet(net, suite, TrainConfig(epochs=5, n_train=256))
>>> batch = suite.sample(64, 0.1, np.random.default_rng(0))
>>> rec = net.reconstruct(batch.noisy_data, suite.operator.singular_values)
>>> bool(summarise_errors(rec, batch.true_coefficients).mean_relative < 1.0)
True
"""

from __future__ import annotations

from .basis import SineBasis
from .baselines import SpatialCNN, SpatialCNNConfig, train_spatial_cnn
from .filters import (
    OracleSelection,
    landweber_damping,
    oracle_spectral_bound,
    oracle_tikhonov,
    oracle_tsvd,
    prior_wiener,
    tikhonov_damping,
    tsvd_damping,
    wiener_damping,
)
from .metrics import ErrorSummary, RateFit, fit_rate, relative_errors, summarise_errors
from .operators import (
    DiagonalSpectralOperator,
    dense_kernel_matrix,
    power_law_operator,
)
from .problems import (
    InverseProblemSuite,
    NoiseModel,
    ProblemBatch,
    SobolevPrior,
)
from .scnet import SCNet, SCNetConfig
from .theory import (
    deterministic_rate,
    optimal_truncation_index,
    predicted_error_terms,
    statistical_rate,
)
from .training import (
    TrainConfig,
    TrainHistory,
    evaluate_on_resolutions,
    sobolev_loss,
    train_scnet,
)

__version__ = "0.1.0"

PAPER_CITATION = (
    "H.-C. Dong, P. Cheng, S. Li. Interpretable Operator Learning for Inverse Problems "
    "via Adaptive Spectral Filtering: Convergence and Discretization Invariance. "
    "arXiv:2603.20602."
)

__all__ = [
    "PAPER_CITATION",
    "DiagonalSpectralOperator",
    "ErrorSummary",
    "InverseProblemSuite",
    "NoiseModel",
    "OracleSelection",
    "ProblemBatch",
    "RateFit",
    "SCNet",
    "SCNetConfig",
    "SineBasis",
    "SobolevPrior",
    "SpatialCNN",
    "SpatialCNNConfig",
    "TrainConfig",
    "TrainHistory",
    "__version__",
    "dense_kernel_matrix",
    "deterministic_rate",
    "evaluate_on_resolutions",
    "fit_rate",
    "landweber_damping",
    "optimal_truncation_index",
    "oracle_spectral_bound",
    "oracle_tikhonov",
    "oracle_tsvd",
    "power_law_operator",
    "predicted_error_terms",
    "prior_wiener",
    "relative_errors",
    "sobolev_loss",
    "statistical_rate",
    "summarise_errors",
    "tikhonov_damping",
    "train_scnet",
    "train_spatial_cnn",
    "tsvd_damping",
    "wiener_damping",
]