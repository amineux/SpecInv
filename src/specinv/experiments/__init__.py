"""Reproduction scripts for the numerical experiments of arXiv:2603.20602.

Each module corresponds to one section of the paper and writes machine-readable output
(JSON plus CSV) into a results directory:

=====================  ==================  =============================================
Module                 Paper section       Output
=====================  ==================  =============================================
:mod:`convergence`     §5.2, Figure 1      ``convergence.json`` / ``.csv``
:mod:`filters`         §5.3, Figure 2      ``filters.json``, ``filter_profiles.csv``
:mod:`zero_shot`       §5.4, Figure 3      ``zero_shot.json`` / ``.csv``
:mod:`ablations`       --                  ``ablations.json`` / ``.csv``
:mod:`run_all`         all                 ``summary.json`` with pass/fail criteria
=====================  ==================  =============================================
"""

from __future__ import annotations

__all__ = ["ablations", "convergence", "filters", "run_all", "zero_shot"]