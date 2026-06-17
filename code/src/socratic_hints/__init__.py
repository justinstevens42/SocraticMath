"""Bayesian analysis of Socratic mathematics hint-type transitions.

This package parses the chains of Socratic questions stored in the
``hint_<domain>/hint_<n>.json`` files, classifies each question into one of
four pedagogical hint types, learns a Dirichlet-Multinomial transition matrix
per domain, and compares domains via row-wise KL divergence.
"""

from .taxonomy import HINT_TYPES, HINT_NAMES, HintType

__all__ = ["HINT_TYPES", "HINT_NAMES", "HintType"]
