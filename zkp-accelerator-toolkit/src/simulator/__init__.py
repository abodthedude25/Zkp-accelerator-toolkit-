"""
SumCheck Performance Simulator (Project 2)

This module provides performance modeling for SumCheck hardware accelerators,
helping understand the tradeoffs between compute and memory resources.

Key Components:
    - HardwareConfig: Define accelerator parameters
    - SumCheckSimulator: Cycle-accurate performance estimation
    - PerformanceMetrics: Results and analysis
    - Predefined polynomials: Vanilla, Jellyfish, etc.
    - Workload comparison: Account for gate reduction

The simulator models:
    - Extension computation cycles
    - Product computation cycles
    - MLE update cycles
    - Memory bandwidth constraints
    - Pipeline utilization

Usage:
    >>> from src.simulator import SumCheckSimulator, HardwareConfig
    >>> from src.simulator import VANILLA_ZEROCHECK, JELLYFISH_ZEROCHECK
    >>> from src.simulator import compare_workload
    >>> 
    >>> config = HardwareConfig(num_pes=4, hbm_bandwidth_gb_s=2000)
    >>> sim = SumCheckSimulator(config)
    >>> 
    >>> # Simple simulation (same gate count)
    >>> metrics = sim.simulate(VANILLA_ZEROCHECK, problem_size=2**20)
    >>> 
    >>> # Workload comparison (accounts for gate reduction!)
    >>> result = compare_workload(sim, VANILLA_ZEROCHECK, JELLYFISH_ZEROCHECK,
    ...                           base_gates=2**20, workload_type="hash")
    >>> print(f"Net speedup: {result.net_speedup:.2f}x")
"""

from .hardware import HardwareConfig
from .core import (
    SumCheckSimulator, 
    PerformanceMetrics,
    WorkloadComparisonResult,
    compare_workload,
    compare_all_workloads,
    print_workload_comparison_table,
    WORKLOAD_GATE_REDUCTION,
)
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
    # Workload comparison
    "WorkloadComparisonResult",
    "compare_workload",
    "compare_all_workloads",
    "print_workload_comparison_table",
    "WORKLOAD_GATE_REDUCTION",
]
