"""
SumCheck Protocol Visualizer (Project 1)

This module provides interactive visualization of the SumCheck protocol,
helping understand how the algorithm works step by step.

Key Components:
    - MLETable: Represents a multilinear extension as a lookup table
    - SumCheckVisualizer: Runs and visualizes the full protocol
    - RoundData: Captures state at each round for analysis

Usage:
    >>> from src.visualizer import SumCheckVisualizer, MLETable
    >>> from src.common.field import PrimeField
    >>> 
    >>> field = PrimeField(97)
    >>> a = MLETable("a", [3, 7, 2, 5, 1, 8, 4, 6], field)
    >>> b = MLETable("b", [1, 4, 6, 2, 8, 3, 5, 7], field)
    >>> 
    >>> viz = SumCheckVisualizer([a, b], field)
    >>> viz.run_full_protocol(verbose=True)
"""

from .mle import MLETable
from .core import SumCheckVisualizer, RoundData

__all__ = [
    "MLETable",
    "SumCheckVisualizer",
    "RoundData",
]
