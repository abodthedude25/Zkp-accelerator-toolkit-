"""
SumCheck Performance Simulator Core Implementation.

This module provides cycle-accurate performance modeling for SumCheck
hardware accelerators. It models:
    - Extension computation (per MLE, per pair)
    - Product computation (per term, per extension point)
    - MLE updates (between rounds)
    - Memory bandwidth constraints

The key insight from zkSpeed/zkPHIRE is that SumCheck performance depends
critically on the balance between:
    - Compute resources (PEs, EEs, PLs)
    - Memory bandwidth (HBM throughput)

For simple polynomials: MEMORY-BOUND (must read all MLE entries each round)
For complex polynomials: COMPUTE-BOUND (many multiplications per entry)

The crossover point depends on polynomial degree and hardware configuration.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import math

from .hardware import HardwareConfig
from .polynomials import SimPolynomial


@dataclass
class PerformanceMetrics:
    """
    Performance metrics from simulation.
    
    Captures both aggregate and per-round metrics for analysis.
    
    Attributes:
        total_cycles: Total execution cycles
        compute_cycles: Cycles spent on computation
        memory_cycles: Cycles spent on memory access
        cycles_per_round: Breakdown by round
        compute_utilization: Fraction of time compute is active
        memory_utilization: Fraction of time memory is active
    """
    total_cycles: int
    compute_cycles: int
    memory_cycles: int
    cycles_per_round: List[int] = field(default_factory=list)
    compute_utilization: float = 0.0
    memory_utilization: float = 0.0
    
    # Detailed breakdown
    extension_cycles: int = 0
    product_cycles: int = 0
    update_cycles: int = 0
    
    @property
    def is_compute_bound(self) -> bool:
        """Check if simulation was compute-bound."""
        return self.compute_cycles >= self.memory_cycles
    
    @property
    def is_memory_bound(self) -> bool:
        """Check if simulation was memory-bound."""
        return self.memory_cycles > self.compute_cycles
    
    @property
    def bottleneck(self) -> str:
        """Return the bottleneck type."""
        if self.is_compute_bound:
            return "COMPUTE"
        return "MEMORY"
    
    @property
    def runtime_ms(self) -> float:
        """Runtime in milliseconds (at 1 GHz)."""
        return self.total_cycles / 1e6
    
    def runtime_ms_at_freq(self, freq_ghz: float) -> float:
        """Runtime in milliseconds at given frequency."""
        return self.total_cycles / (freq_ghz * 1e6)
    
    def summary(self) -> str:
        """Return summary string."""
        return (
            f"PerformanceMetrics:\n"
            f"  Total cycles: {self.total_cycles:,}\n"
            f"  Runtime (1 GHz): {self.runtime_ms:.2f} ms\n"
            f"  Compute cycles: {self.compute_cycles:,}\n"
            f"  Memory cycles: {self.memory_cycles:,}\n"
            f"  Bottleneck: {self.bottleneck}\n"
            f"  Compute utilization: {self.compute_utilization:.1%}\n"
            f"  Memory utilization: {self.memory_utilization:.1%}\n"
            f"  Rounds: {len(self.cycles_per_round)}"
        )
    
    def __repr__(self) -> str:
        return (f"PerformanceMetrics(cycles={self.total_cycles:,}, "
                f"runtime={self.runtime_ms:.2f}ms, bottleneck={self.bottleneck})")


@dataclass
class RoundMetrics:
    """Metrics for a single SumCheck round."""
    round_num: int
    table_size: int
    compute_cycles: int
    memory_cycles: int
    extension_cycles: int
    product_cycles: int
    update_cycles: int
    
    @property
    def total_cycles(self) -> int:
        """Round takes max of compute and memory (they overlap)."""
        return max(self.compute_cycles, self.memory_cycles)
    
    @property
    def bottleneck(self) -> str:
        return "COMPUTE" if self.compute_cycles >= self.memory_cycles else "MEMORY"


class SumCheckSimulator:
    """
    Cycle-accurate SumCheck performance simulator.
    
    Models the execution of SumCheck on a configurable hardware accelerator.
    The simulation accounts for:
        - Parallel processing across PEs
        - Pipeline latencies
        - Memory bandwidth constraints
        - Data reuse patterns
    
    Usage:
        >>> config = HardwareConfig(num_pes=4, hbm_bandwidth_gb_s=2000)
        >>> sim = SumCheckSimulator(config)
        >>> metrics = sim.simulate(VANILLA_ZEROCHECK, problem_size=2**20)
        >>> print(metrics.summary())
    """
    
    def __init__(self, config: HardwareConfig):
        """
        Initialize simulator with hardware configuration.
        
        Args:
            config: Hardware configuration
        """
        self.config = config
    
    def simulate(self, polynomial: SimPolynomial, 
                 problem_size: int) -> PerformanceMetrics:
        """
        Simulate SumCheck execution.
        
        Args:
            polynomial: The polynomial structure to process
            problem_size: Number of gates (2^μ)
            
        Returns:
            PerformanceMetrics with cycle counts and utilization
        """
        num_rounds = int(math.log2(problem_size))
        
        round_metrics: List[RoundMetrics] = []
        current_size = problem_size
        
        for round_idx in range(num_rounds):
            is_first = (round_idx == 0)
            rm = self._simulate_round(polynomial, current_size, is_first, round_idx + 1)
            round_metrics.append(rm)
            current_size //= 2
        
        # Aggregate metrics
        total_compute = sum(rm.compute_cycles for rm in round_metrics)
        total_memory = sum(rm.memory_cycles for rm in round_metrics)
        total_cycles = sum(rm.total_cycles for rm in round_metrics)
        
        total_extension = sum(rm.extension_cycles for rm in round_metrics)
        total_product = sum(rm.product_cycles for rm in round_metrics)
        total_update = sum(rm.update_cycles for rm in round_metrics)
        
        # Compute utilization
        effective_time = total_cycles * self.config.num_pes
        compute_util = min(1.0, total_compute / effective_time) if effective_time > 0 else 0
        memory_util = min(1.0, total_memory / total_cycles) if total_cycles > 0 else 0
        
        return PerformanceMetrics(
            total_cycles=total_cycles,
            compute_cycles=total_compute,
            memory_cycles=total_memory,
            cycles_per_round=[rm.total_cycles for rm in round_metrics],
            compute_utilization=compute_util,
            memory_utilization=memory_util,
            extension_cycles=total_extension,
            product_cycles=total_product,
            update_cycles=total_update,
        )
    
    def _simulate_round(self, polynomial: SimPolynomial, table_size: int,
                        is_first_round: bool, round_num: int) -> RoundMetrics:
        """
        Simulate one SumCheck round.
        
        Returns RoundMetrics with cycle breakdown.
        """
        num_pairs = table_size // 2
        num_mles = polynomial.num_mles
        num_extensions = polynomial.num_extensions
        num_terms = polynomial.num_terms
        
        # =====================================================================
        # COMPUTE CYCLES
        # =====================================================================
        
        # 1. Extension computation
        # Per pair, per MLE: need to compute 'num_extensions' values
        # Each extension value: 1 sub + 1 mul + 1 add ≈ 1 mul equivalent
        ops_per_extension = num_extensions
        extension_ops_per_pair = num_mles * ops_per_extension
        
        # 2. Product computation
        # Per pair, per extension point, per term
        # Each term with k MLEs needs (k-1) multiplications
        muls_per_term = polynomial.multiplications_per_evaluation()
        product_ops_per_pair = muls_per_term * num_extensions
        
        # Total compute ops per pair
        total_ops_per_pair = extension_ops_per_pair + product_ops_per_pair
        
        # Distribute across PEs
        pairs_per_pe = math.ceil(num_pairs / self.config.num_pes)
        
        # With pipelining, throughput is limited by:
        # - Number of EEs (for extensions)
        # - Number of PLs (for products)
        
        # Extension cycles (per PE)
        ext_cycles_per_pair = math.ceil(
            extension_ops_per_pair / self.config.extension_engines_per_pe
        )
        extension_cycles = pairs_per_pe * ext_cycles_per_pair + self.config.modmul_latency
        
        # Product cycles (per PE)
        prod_cycles_per_pair = math.ceil(
            product_ops_per_pair / self.config.product_lanes_per_pe
        )
        product_cycles = pairs_per_pe * prod_cycles_per_pair + self.config.modmul_latency
        
        # Extension and product can often overlap, but we'll be conservative
        compute_cycles = extension_cycles + product_cycles
        
        # =====================================================================
        # MEMORY CYCLES
        # =====================================================================
        
        # Read all MLEs
        bytes_per_element = self.config.bytes_per_element
        
        if is_first_round:
            # First round: some MLEs are sparse (selectors like qL, qR)
            # Assume 50% sparse (1-byte), 50% dense (32-byte)
            sparse_ratio = 0.5
            sparse_mles = int(num_mles * sparse_ratio)
            dense_mles = num_mles - sparse_mles
            
            read_bytes = (sparse_mles * table_size * 1 +
                         dense_mles * table_size * bytes_per_element)
        else:
            # After first round, all MLEs are dense
            read_bytes = num_mles * table_size * bytes_per_element
        
        # Write updated MLEs (half size)
        write_bytes = num_mles * (table_size // 2) * bytes_per_element
        
        total_bytes = read_bytes + write_bytes
        
        # Memory cycles (accounting for bandwidth and latency)
        transfer_cycles = int(total_bytes / self.config.bandwidth_bytes_per_cycle)
        memory_cycles = transfer_cycles + self.config.memory_latency
        
        # MLE Update cycles (between rounds)
        # This is usually hidden by memory transfer time
        update_ops = num_mles * num_pairs * 3  # 1 sub + 1 mul + 1 add per entry
        update_cycles = math.ceil(update_ops / self.config.total_product_lanes)
        
        return RoundMetrics(
            round_num=round_num,
            table_size=table_size,
            compute_cycles=compute_cycles,
            memory_cycles=memory_cycles,
            extension_cycles=extension_cycles,
            product_cycles=product_cycles,
            update_cycles=update_cycles,
        )
    
    def sweep_bandwidth(self, polynomial: SimPolynomial, problem_size: int,
                        bandwidths: List[float]) -> Dict[float, PerformanceMetrics]:
        """
        Sweep bandwidth to analyze memory sensitivity.
        
        Args:
            polynomial: Polynomial to simulate
            problem_size: Number of gates
            bandwidths: List of bandwidth values (GB/s)
            
        Returns:
            Dict mapping bandwidth to performance metrics
        """
        results = {}
        original_bw = self.config.hbm_bandwidth_gb_s
        
        for bw in bandwidths:
            self.config.hbm_bandwidth_gb_s = bw
            results[bw] = self.simulate(polynomial, problem_size)
        
        self.config.hbm_bandwidth_gb_s = original_bw
        return results
    
    def sweep_pes(self, polynomial: SimPolynomial, problem_size: int,
                  pe_counts: List[int]) -> Dict[int, PerformanceMetrics]:
        """
        Sweep PE count to analyze compute scaling.
        
        Args:
            polynomial: Polynomial to simulate
            problem_size: Number of gates
            pe_counts: List of PE counts
            
        Returns:
            Dict mapping PE count to performance metrics
        """
        results = {}
        original_pes = self.config.num_pes
        
        for pes in pe_counts:
            self.config.num_pes = pes
            results[pes] = self.simulate(polynomial, problem_size)
        
        self.config.num_pes = original_pes
        return results
    
    def sweep_problem_size(self, polynomial: SimPolynomial,
                           sizes: List[int]) -> Dict[int, PerformanceMetrics]:
        """
        Sweep problem size to analyze scaling.
        
        Args:
            polynomial: Polynomial to simulate
            sizes: List of problem sizes (must be powers of 2)
            
        Returns:
            Dict mapping size to performance metrics
        """
        results = {}
        for size in sizes:
            results[size] = self.simulate(polynomial, size)
        return results
    
    def compare_polynomials(self, polynomials: List[SimPolynomial],
                            problem_size: int) -> Dict[str, PerformanceMetrics]:
        """
        Compare performance across different polynomials.
        
        Useful for understanding vanilla vs. Jellyfish tradeoffs.
        """
        results = {}
        for poly in polynomials:
            results[poly.name] = self.simulate(poly, problem_size)
        return results
    
    def find_crossover_degree(self, base_polynomial: SimPolynomial,
                              problem_size: int,
                              max_degree: int = 20) -> int:
        """
        Find the degree at which compute starts dominating memory.
        
        This is the "crossover point" mentioned in zkPHIRE where
        higher-degree gates start seeing diminishing returns.
        
        Returns:
            Degree at which bottleneck switches from memory to compute
        """
        from .polynomials import create_polynomial_with_degree
        
        for degree in range(2, max_degree + 1):
            poly = create_polynomial_with_degree(degree, base_polynomial.num_terms)
            metrics = self.simulate(poly, problem_size)
            
            if metrics.is_compute_bound:
                return degree
        
        return max_degree  # Still memory-bound at max degree


def analyze_design_space(polynomial: SimPolynomial, problem_size: int,
                         pe_range: Tuple[int, int] = (1, 16),
                         bw_range: Tuple[float, float] = (256, 4096)
                         ) -> List[Dict]:
    """
    Explore the design space of PE count vs. bandwidth.
    
    Returns a list of configurations with their metrics,
    useful for plotting Pareto frontiers.
    """
    results = []
    
    pe_values = [2**i for i in range(int(math.log2(pe_range[0])), 
                                      int(math.log2(pe_range[1])) + 1)]
    bw_values = [256 * (2**i) for i in range(int(math.log2(bw_range[0]/256)),
                                              int(math.log2(bw_range[1]/256)) + 1)]
    
    for num_pes in pe_values:
        for bandwidth in bw_values:
            config = HardwareConfig(
                num_pes=num_pes,
                hbm_bandwidth_gb_s=bandwidth
            )
            sim = SumCheckSimulator(config)
            metrics = sim.simulate(polynomial, problem_size)
            
            results.append({
                "num_pes": num_pes,
                "bandwidth": bandwidth,
                "cycles": metrics.total_cycles,
                "runtime_ms": metrics.runtime_ms,
                "bottleneck": metrics.bottleneck,
                "compute_util": metrics.compute_utilization,
                "memory_util": metrics.memory_utilization,
            })
    
    return results


# Example usage
if __name__ == "__main__":
    from .polynomials import VANILLA_ZEROCHECK, JELLYFISH_ZEROCHECK
    from .hardware import create_zkphire_config
    
    print("=" * 70)
    print("SUMCHECK SIMULATOR DEMO")
    print("=" * 70)
    
    config = create_zkphire_config()
    sim = SumCheckSimulator(config)
    
    print(f"\n{config.summary()}")
    
    # Simulate vanilla ZeroCheck
    print("\n" + "-" * 70)
    print("VANILLA ZEROCHECK SIMULATION")
    print("-" * 70)
    
    problem_size = 2**20
    metrics = sim.simulate(VANILLA_ZEROCHECK, problem_size)
    
    print(f"\nProblem size: 2^20 = {problem_size:,} gates")
    print(f"\n{metrics.summary()}")
    
    # Compare vanilla vs jellyfish
    print("\n" + "-" * 70)
    print("VANILLA vs JELLYFISH COMPARISON")
    print("-" * 70)
    
    comparison = sim.compare_polynomials(
        [VANILLA_ZEROCHECK, JELLYFISH_ZEROCHECK],
        problem_size
    )
    
    for name, m in comparison.items():
        print(f"\n{name}:")
        print(f"  Cycles: {m.total_cycles:,}")
        print(f"  Runtime: {m.runtime_ms:.2f} ms")
        print(f"  Bottleneck: {m.bottleneck}")
