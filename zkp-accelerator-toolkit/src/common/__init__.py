"""
Common utilities for ZKP Accelerator Toolkit.

This module provides:
    - Finite field arithmetic (PrimeField)
    - Polynomial representations (Polynomial, Term)
    - Utility functions for ZKP computations
"""

from .field import PrimeField, FieldElement
from .polynomial import Polynomial, Term, PolynomialTerm

__all__ = [
    "PrimeField",
    "FieldElement", 
    "Polynomial",
    "Term",
    "PolynomialTerm",
]
