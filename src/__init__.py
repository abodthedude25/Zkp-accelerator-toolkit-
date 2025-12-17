"""
ZKP Accelerator Toolkit
=======================

A comprehensive educational toolkit for understanding Zero-Knowledge Proof (ZKP)
accelerators, specifically focused on the SumCheck protocol used in HyperPlonk.

Modules:
    - visualizer: Interactive SumCheck visualization
    - simulator: Hardware performance modeling
    - optimizer: Custom gate design exploration
    - common: Shared utilities (field arithmetic, polynomials)

Quick Start:
    >>> from src.visualizer import SumCheckVisualizer, MLETable
    >>> from src.common.field import PrimeField
    >>> field = PrimeField(97)
    >>> # ... see README.md for full examples
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from . import common
from . import visualizer
from . import simulator
from . import optimizer
