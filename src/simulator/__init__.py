"""
SumCheck Performance Simulator (Project 2)

This module provides performance modeling for SumCheck hardware accelerators,
helping understand the tradeoffs between compute and memory resources.

Key Components:
    - HardwareConfig: Define accelerator parameters
    - SumCheckSimulator: Cycle-accurate performance estimation
    - PerformanceMetrics: Results and analysis
    - Predefined polynomials: Vanilla, Jellyfish, etc.

The simulator models:
    - Extension computation cycles
    - Product computation cycles
    - MLE update cycles
    - Memory bandwidth constraints
    - Pipeline utilization

Usage:
    >>> from src.simulator import SumCheckSimulator, HardwareConfig
    >>> from src.simulator import VANILLA_ZEROCHECK
    >>> 
    >>> config = HardwareConfig(num_pes=4, hbm_bandwidth_gb_s=2000)
    >>> sim = SumCheckSimulator(config)
    >>> metrics = sim.simulate(VANILLA_ZEROCHECK, problem_size=2**20)
    >>> print(f"Runtime: {metrics.runtime_ms:.2f} ms")
"""

from .hardware import HardwareConfig
from .core import SumCheckSimulator, PerformanceMetrics
from .polynomials import (
    VANILLA_ZEROCHECK,
    VANILLA_PERMCHECK,
    VANILLA_OPENCHECK,
    JELLYFISH_ZEROCHECK,
    SimPolynomial,
)

__all__ = [
    "HardwareConfig",
    "SumCheckSimulator",
    "PerformanceMetrics",
    "SimPolynomial",
    "VANILLA_ZEROCHECK",
    "VANILLA_PERMCHECK",
    "VANILLA_OPENCHECK",
    "JELLYFISH_ZEROCHECK",
]
