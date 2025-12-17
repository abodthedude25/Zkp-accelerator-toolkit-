"""
Custom Gate Design Optimizer (Project 3)

This module provides tools for exploring custom gate configurations
and understanding the tradeoffs between vanilla and high-degree gates.

Key Components:
    - GateType: Define gate characteristics
    - Computation: Represent computations to be proven
    - GateOptimizer: Find optimal gate configurations
    - Analysis utilities for pattern recognition

The key insight from zkPHIRE is that custom gates like Jellyfish can
reduce gate count by 8-32x, even though each gate is more complex.
This module helps explore those tradeoffs.

Usage:
    >>> from src.optimizer import GateOptimizer, Computation, Operation, OpType
    >>> 
    >>> # Define a computation: x^5
    >>> comp = Computation(
    ...     operations=[
    ...         Operation(OpType.MUL, ["x", "x"], "x2"),
    ...         Operation(OpType.MUL, ["x2", "x2"], "x4"),
    ...         Operation(OpType.MUL, ["x4", "x"], "x5"),
    ...     ],
    ...     inputs={"x"},
    ...     outputs={"x5"}
    ... )
    >>> 
    >>> optimizer = GateOptimizer()
    >>> best, analysis = optimizer.optimize(comp)
    >>> print(f"Recommended: {analysis['recommendation']}")
"""

from .gates import GateType, VANILLA_GATE, JELLYFISH_GATE, create_custom_gate
from .computation import Computation, Operation, OpType
from .core import GateOptimizer, GateMapping, OptimizationResult

__all__ = [
    "GateType",
    "VANILLA_GATE",
    "JELLYFISH_GATE",
    "create_custom_gate",
    "Computation",
    "Operation",
    "OpType",
    "GateOptimizer",
    "GateMapping",
    "OptimizationResult",
]
