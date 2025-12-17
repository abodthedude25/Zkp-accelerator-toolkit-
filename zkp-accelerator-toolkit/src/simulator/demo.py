"""
SumCheck Performance Simulator Demo

This script demonstrates the performance simulator with various
hardware configurations and polynomial types.

Run with:
    python -m src.simulator.demo
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.simulator.hardware import (
    HardwareConfig,
    create_zkspeed_config,
    create_zkphire_config,
    create_minimal_config,
)
from src.simulator.polynomials import (
    VANILLA_ZEROCHECK,
    VANILLA_PERMCHECK,
    JELLYFISH_ZEROCHECK,
    JELLYFISH_SIMPLE,
    SimPolynomial,
    create_polynomial_with_degree,
)
from src.simulator.core import (
    SumCheckSimulator,
    analyze_design_space,
    compare_workload,
    print_workload_comparison_table,
    WORKLOAD_GATE_REDUCTION,
)


def demo_basic_simulation():
    """Basic simulation demonstrating core functionality."""
    print("\n" + "=" * 70)
    print("DEMO 1: BASIC SIMULATION")
    print("=" * 70)
    
    # Create simulator with zkPHIRE-like config
    config = create_zkphire_config()
    sim = SumCheckSimulator(config)
    
    print(f"\nHardware Configuration:")
    print(f"  {config}")
    print(f"  Bandwidth: {config.hbm_bandwidth_gb_s} GB/s")
    print(f"  Total EEs: {config.total_extension_engines}")
    print(f"  Total PLs: {config.total_product_lanes}")
    
    # Simulate on Vanilla ZeroCheck
    problem_size = 2**20  # 1 million gates
    
    print(f"\nPolynomial: {VANILLA_ZEROCHECK.name}")
    print(f"  Terms: {VANILLA_ZEROCHECK.num_terms}")
    print(f"  Degree: {VANILLA_ZEROCHECK.max_degree}")
    print(f"  MLEs: {VANILLA_ZEROCHECK.num_mles}")
    
    print(f"\nProblem size: 2^20 = {problem_size:,} gates")
    
    metrics = sim.simulate(VANILLA_ZEROCHECK, problem_size)
    
    print(f"\nResults:")
    print(f"  Total cycles: {metrics.total_cycles:,}")
    print(f"  Runtime: {metrics.runtime_ms:.2f} ms")
    print(f"  Bottleneck: {metrics.bottleneck}")
    print(f"  Compute utilization: {metrics.compute_utilization:.1%}")
    print(f"  Memory utilization: {metrics.memory_utilization:.1%}")


def demo_workload_comparison():
    """Compare Vanilla vs Jellyfish accounting for gate reduction."""
    print("\n" + "=" * 70)
    print("DEMO 2: WORKLOAD COMPARISON (KEY INSIGHT!)")
    print("=" * 70)
    
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│  WHY THIS MATTERS                                                   │
├─────────────────────────────────────────────────────────────────────┤
│  A naive comparison of Vanilla vs Jellyfish at the SAME gate count  │
│  makes Jellyfish look slower (2.4x more compute per gate).          │
│                                                                     │
│  But the whole point of Jellyfish is GATE REDUCTION!                │
│  - Hash workloads: 8x fewer gates                                   │
│  - EC workloads: 4x fewer gates                                     │
│  - Mixed workloads: 5x fewer gates                                  │
│                                                                     │
│  A fair comparison uses different gate counts for each approach.    │
└─────────────────────────────────────────────────────────────────────┘
""")
    
    config = create_zkphire_config()
    sim = SumCheckSimulator(config)
    
    problem_size = 2**20
    
    # First show the misleading same-gate comparison
    print("MISLEADING: Same gate count comparison")
    print("-" * 50)
    print(f"Problem size: 2^20 = {problem_size:,} gates for BOTH")
    
    v_metrics = sim.simulate(VANILLA_ZEROCHECK, problem_size)
    j_metrics = sim.simulate(JELLYFISH_ZEROCHECK, problem_size)
    
    print(f"\n  Vanilla:   {v_metrics.runtime_ms:>8.2f} ms")
    print(f"  Jellyfish: {j_metrics.runtime_ms:>8.2f} ms")
    print(f"  Ratio: {j_metrics.runtime_ms / v_metrics.runtime_ms:.1f}x slower 😞")
    print("\n⚠️  This ignores gate reduction!")
    
    # Now show the correct comparison
    print("\n" + "=" * 50)
    print("CORRECT: Accounting for gate reduction")
    print("=" * 50)
    
    print_workload_comparison_table(sim, VANILLA_ZEROCHECK, JELLYFISH_ZEROCHECK, problem_size)
    
    print("\nAnalysis:")
    print("  - Hash workloads (8x reduction): ~3x net speedup!")
    print("  - EC workloads (4x reduction): ~1.5x net speedup")
    print("  - Mixed workloads (5x reduction): ~2x net speedup")
    print("\nThis is why zkPHIRE excels on hash-heavy applications like rollups!")


def demo_polynomial_comparison():
    """Compare different polynomial structures."""
    print("\n" + "=" * 70)
    print("DEMO 3: POLYNOMIAL COMPARISON (same gate count)")
    print("=" * 70)
    
    config = create_zkphire_config()
    sim = SumCheckSimulator(config)
    
    problem_size = 2**20
    
    polynomials = [
        VANILLA_ZEROCHECK,
        VANILLA_PERMCHECK,
        JELLYFISH_SIMPLE,
        JELLYFISH_ZEROCHECK,
    ]
    
    print(f"\nProblem size: 2^20 = {problem_size:,} gates")
    print(f"\n{'Polynomial':<25} {'Terms':>6} {'Degree':>7} {'MLEs':>5} "
          f"{'Cycles':>12} {'Runtime':>10} {'Bottleneck':>12}")
    print("-" * 90)
    
    for poly in polynomials:
        metrics = sim.simulate(poly, problem_size)
        
        print(f"{poly.name:<25} {poly.num_terms:>6} {poly.max_degree:>7} "
              f"{poly.num_mles:>5} {metrics.total_cycles:>12,} "
              f"{metrics.runtime_ms:>9.2f}ms {metrics.bottleneck:>12}")
    
    # Explain the results
    print("\nAnalysis:")
    print("  - PermCheck has highest degree (5) due to fraction polynomials")
    print("  - Jellyfish ZeroCheck has many terms (13) but degree 7")
    print("  - Higher degree → more extensions → more compute per pair")
    print("  - More terms → more products → more compute per pair")


def demo_bandwidth_sensitivity():
    """Analyze how performance varies with memory bandwidth."""
    print("\n" + "=" * 70)
    print("DEMO 3: BANDWIDTH SENSITIVITY ANALYSIS")
    print("=" * 70)
    
    config = create_zkphire_config()
    sim = SumCheckSimulator(config)
    
    problem_size = 2**20
    bandwidths = [256, 512, 1024, 2048, 4096]  # GB/s
    
    print(f"\nPolynomial: {VANILLA_ZEROCHECK.name}")
    print(f"Problem size: 2^20 = {problem_size:,} gates")
    
    print(f"\n{'Bandwidth (GB/s)':<18} {'Cycles':>12} {'Runtime (ms)':>14} "
          f"{'Speedup':>10} {'Bottleneck':>12}")
    print("-" * 70)
    
    results = sim.sweep_bandwidth(VANILLA_ZEROCHECK, problem_size, bandwidths)
    baseline_cycles = results[bandwidths[0]].total_cycles
    
    for bw, metrics in results.items():
        speedup = baseline_cycles / metrics.total_cycles
        print(f"{bw:<18} {metrics.total_cycles:>12,} {metrics.runtime_ms:>14.2f} "
              f"{speedup:>10.2f}x {metrics.bottleneck:>12}")
    
    print("\nAnalysis:")
    print("  - Vanilla ZeroCheck is MEMORY-BOUND at typical bandwidths")
    print("  - Doubling bandwidth roughly halves runtime (until compute-bound)")
    print("  - At very high bandwidth, compute becomes the bottleneck")
    print("  - This matches Figure 9 in the zkPHIRE paper")


def demo_pe_scaling():
    """Analyze how performance scales with PE count."""
    print("\n" + "=" * 70)
    print("DEMO 4: PE SCALING ANALYSIS")
    print("=" * 70)
    
    config = create_zkphire_config()
    sim = SumCheckSimulator(config)
    
    problem_size = 2**20
    pe_counts = [1, 2, 4, 8, 16]
    
    print(f"\nPolynomial: {VANILLA_ZEROCHECK.name}")
    print(f"Problem size: 2^20 = {problem_size:,} gates")
    print(f"Bandwidth: {config.hbm_bandwidth_gb_s} GB/s")
    
    print(f"\n{'PEs':<8} {'Cycles':>12} {'Runtime (ms)':>14} "
          f"{'Speedup':>10} {'Efficiency':>12}")
    print("-" * 60)
    
    results = sim.sweep_pes(VANILLA_ZEROCHECK, problem_size, pe_counts)
    baseline_cycles = results[1].total_cycles
    
    for pes, metrics in results.items():
        speedup = baseline_cycles / metrics.total_cycles
        efficiency = speedup / pes
        print(f"{pes:<8} {metrics.total_cycles:>12,} {metrics.runtime_ms:>14.2f} "
              f"{speedup:>10.2f}x {efficiency:>11.1%}")
    
    print("\nAnalysis:")
    print("  - For memory-bound workloads, PE scaling is limited")
    print("  - Efficiency drops because memory bandwidth is shared")
    print("  - Better PE scaling for compute-bound (high-degree) polynomials")


def demo_problem_size_scaling():
    """Analyze how performance scales with problem size."""
    print("\n" + "=" * 70)
    print("DEMO 5: PROBLEM SIZE SCALING")
    print("=" * 70)
    
    config = create_zkphire_config()
    sim = SumCheckSimulator(config)
    
    sizes = [2**i for i in range(16, 25)]  # 2^16 to 2^24
    
    print(f"\nPolynomial: {VANILLA_ZEROCHECK.name}")
    
    print(f"\n{'Size':>12} {'Gates':>14} {'Rounds':>8} "
          f"{'Cycles':>14} {'Runtime (ms)':>14}")
    print("-" * 70)
    
    results = sim.sweep_problem_size(VANILLA_ZEROCHECK, sizes)
    
    for size, metrics in results.items():
        import math
        rounds = int(math.log2(size))
        print(f"2^{int(math.log2(size)):>2} = {size:>10,} {rounds:>8} "
              f"{metrics.total_cycles:>14,} {metrics.runtime_ms:>14.2f}")
    
    print("\nAnalysis:")
    print("  - Runtime scales roughly linearly with problem size")
    print("  - Each doubling of gates adds one more round")
    print("  - Memory bandwidth dominates at all sizes for this polynomial")


def demo_degree_crossover():
    """Find the degree at which compute starts dominating memory."""
    print("\n" + "=" * 70)
    print("DEMO 6: DEGREE CROSSOVER ANALYSIS")
    print("=" * 70)
    
    config = create_zkphire_config()
    sim = SumCheckSimulator(config)
    
    problem_size = 2**20
    
    print(f"\nProblem size: 2^20 = {problem_size:,} gates")
    print(f"Bandwidth: {config.hbm_bandwidth_gb_s} GB/s")
    
    print(f"\n{'Degree':>8} {'Extensions':>12} {'Cycles':>14} "
          f"{'Bottleneck':>12} {'Compute%':>12}")
    print("-" * 65)
    
    for degree in range(2, 20):
        poly = create_polynomial_with_degree(degree, num_terms=5)
        metrics = sim.simulate(poly, problem_size)
        
        compute_pct = metrics.compute_cycles / (metrics.compute_cycles + metrics.memory_cycles) * 100
        print(f"{degree:>8} {poly.num_extensions:>12} {metrics.total_cycles:>14,} "
              f"{metrics.bottleneck:>12} {compute_pct:>11.1f}%")
        
        if degree > 2 and metrics.is_compute_bound:
            break
    
    print("\nAnalysis:")
    print("  - Low degree polynomials are MEMORY-BOUND")
    print("  - High degree polynomials are COMPUTE-BOUND")
    print("  - Crossover typically around degree 15-18")
    print("  - This is why zkPHIRE paper notes d=18 crossover point")


def demo_hardware_comparison():
    """Compare different hardware configurations."""
    print("\n" + "=" * 70)
    print("DEMO 7: HARDWARE CONFIGURATION COMPARISON")
    print("=" * 70)
    
    configs = [
        create_minimal_config(),
        create_zkspeed_config(),
        create_zkphire_config(),
        HardwareConfig(name="high-bandwidth", num_pes=4, hbm_bandwidth_gb_s=4000),
        HardwareConfig(name="high-compute", num_pes=16, hbm_bandwidth_gb_s=2000),
    ]
    
    problem_size = 2**20
    polynomial = VANILLA_ZEROCHECK
    
    print(f"\nPolynomial: {polynomial.name}")
    print(f"Problem size: 2^20 = {problem_size:,} gates")
    
    print(f"\n{'Config':<20} {'PEs':>5} {'BW(GB/s)':>10} "
          f"{'Cycles':>14} {'Runtime':>12} {'Bottleneck':>12}")
    print("-" * 80)
    
    for config in configs:
        sim = SumCheckSimulator(config)
        metrics = sim.simulate(polynomial, problem_size)
        
        print(f"{config.name:<20} {config.num_pes:>5} "
              f"{config.hbm_bandwidth_gb_s:>10.0f} "
              f"{metrics.total_cycles:>14,} "
              f"{metrics.runtime_ms:>11.2f}ms {metrics.bottleneck:>12}")
    
    print("\nAnalysis:")
    print("  - zkPHIRE config balances compute and memory well")
    print("  - High-bandwidth config helps memory-bound workloads most")
    print("  - High-compute config has diminishing returns for memory-bound")


def demo_design_space_exploration():
    """Explore the design space of PE count vs. bandwidth."""
    print("\n" + "=" * 70)
    print("DEMO 8: DESIGN SPACE EXPLORATION")
    print("=" * 70)
    
    problem_size = 2**20
    polynomial = VANILLA_ZEROCHECK
    
    print(f"\nPolynomial: {polynomial.name}")
    print(f"Problem size: 2^20 = {problem_size:,} gates")
    
    results = analyze_design_space(polynomial, problem_size,
                                    pe_range=(1, 8),
                                    bw_range=(512, 2048))
    
    print(f"\n{'PEs':>5} {'BW(GB/s)':>10} {'Cycles':>14} "
          f"{'Runtime':>12} {'Bottleneck':>12}")
    print("-" * 60)
    
    for r in sorted(results, key=lambda x: (x["num_pes"], x["bandwidth"])):
        print(f"{r['num_pes']:>5} {r['bandwidth']:>10.0f} "
              f"{r['cycles']:>14,} {r['runtime_ms']:>11.2f}ms "
              f"{r['bottleneck']:>12}")
    
    # Find Pareto-optimal points
    print("\nPareto-optimal configurations (best runtime for given PEs):")
    by_pes = {}
    for r in results:
        pes = r["num_pes"]
        if pes not in by_pes or r["cycles"] < by_pes[pes]["cycles"]:
            by_pes[pes] = r
    
    for pes in sorted(by_pes.keys()):
        r = by_pes[pes]
        print(f"  {r['num_pes']} PEs, {r['bandwidth']:.0f} GB/s: "
              f"{r['runtime_ms']:.2f} ms")


def main():
    """Run all demos."""
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 12 + "SUMCHECK PERFORMANCE SIMULATOR DEMONSTRATION" + " " * 11 + "║")
    print("╚" + "═" * 68 + "╝")
    
    print("\nThis demo shows performance modeling for SumCheck accelerators.")
    print("The simulator helps understand:")
    print("  - Compute vs. memory bottlenecks")
    print("  - Impact of polynomial complexity")
    print("  - Hardware resource tradeoffs")
    print("\n" + "─" * 70)
    
    demos = [
        ("Basic Simulation", demo_basic_simulation),
        ("Workload Comparison (KEY!)", demo_workload_comparison),
        ("Polynomial Comparison", demo_polynomial_comparison),
        ("Bandwidth Sensitivity", demo_bandwidth_sensitivity),
        ("PE Scaling", demo_pe_scaling),
        ("Problem Size Scaling", demo_problem_size_scaling),
        ("Degree Crossover", demo_degree_crossover),
        ("Hardware Comparison", demo_hardware_comparison),
        ("Design Space Exploration", demo_design_space_exploration),
    ]
    
    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\nError in {name}: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "─" * 70)
        input("Press Enter to continue to next demo...")
    
    print("\n" + "═" * 70)
    print("DEMOS COMPLETE")
    print("═" * 70)
    print("\nKey takeaways for zkSpeed/zkPHIRE context:")
    print("  1. Vanilla polynomials are typically MEMORY-BOUND")
    print("  2. High-degree (Jellyfish) gates shift toward COMPUTE-BOUND")
    print("  3. The crossover happens around degree 15-18")
    print("  4. Memory bandwidth is critical for low-degree workloads")
    print("  5. PE scaling has diminishing returns for memory-bound work")
    print("\nThese insights explain why:")
    print("  - zkSpeed uses unified PE design (memory-bound anyway)")
    print("  - zkPHIRE invests in programmability (future high-degree gates)")
    print("  - Both designs use HBM for high bandwidth")


if __name__ == "__main__":
    main()
