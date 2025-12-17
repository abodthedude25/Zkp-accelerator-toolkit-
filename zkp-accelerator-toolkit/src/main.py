"""
ZKP Accelerator Toolkit - Main Entry Point

This script provides a unified interface to run all three projects:
    1. SumCheck Visualizer - Understand the algorithm
    2. Performance Simulator - Model hardware tradeoffs
    3. Gate Optimizer - Find optimal gate configurations

Run with:
    python -m src.main
"""

import sys
from pathlib import Path


def print_banner():
    """Print the toolkit banner."""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + " " * 15 + "ZKP ACCELERATOR TOOLKIT" + " " * 30 + "║")
    print("║" + " " * 68 + "║")
    print("║" + " " * 10 + "Understanding zkSpeed and zkPHIRE Papers" + " " * 17 + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    print()


def print_menu():
    """Print the main menu."""
    print("This toolkit contains three educational projects:")
    print()
    print("  [1] SumCheck Visualizer")
    print("      Interactive step-by-step visualization of the SumCheck protocol")
    print("      → Understand how MLEs are processed round by round")
    print()
    print("  [2] Performance Simulator")
    print("      Model SumCheck performance on configurable hardware")
    print("      → Understand compute vs memory tradeoffs")
    print()
    print("  [3] Gate Optimizer")
    print("      Find optimal gate configurations for computations")
    print("      → Understand why Jellyfish gates help")
    print()
    print("  [4] Quick Demo (all three)")
    print("      Run a brief demo of each project")
    print()
    print("  [q] Quit")
    print()


def run_visualizer():
    """Run the SumCheck visualizer demo."""
    print("\n" + "=" * 70)
    print("RUNNING SUMCHECK VISUALIZER")
    print("=" * 70)
    
    from src.common.field import PrimeField
    from src.visualizer.mle import MLETable
    from src.visualizer.core import SumCheckVisualizer
    
    # Create a simple example
    field = PrimeField(97)
    
    a = MLETable("a", [3, 7, 2, 5, 1, 8, 4, 6], field)
    b = MLETable("b", [1, 4, 6, 2, 8, 3, 5, 7], field)
    c = MLETable("c", [5, 2, 8, 1, 4, 7, 3, 6], field)
    
    viz = SumCheckVisualizer([a, b, c], field, seed=42)
    result = viz.run_full_protocol(verbose=True)
    
    print("\n✓ Visualizer demo complete!")
    print(f"  Verified: {result.verified}")


def run_simulator():
    """Run the performance simulator demo."""
    print("\n" + "=" * 70)
    print("RUNNING PERFORMANCE SIMULATOR")
    print("=" * 70)
    
    from src.simulator.hardware import create_zkphire_config
    from src.simulator.polynomials import VANILLA_ZEROCHECK, JELLYFISH_ZEROCHECK
    from src.simulator.core import SumCheckSimulator, compare_workload
    
    config = create_zkphire_config()
    sim = SumCheckSimulator(config)
    
    problem_size = 2**20
    
    print(f"\nHardware: {config.name}")
    print(f"Problem size: 2^20 = {problem_size:,} gates")
    
    # OLD comparison: same gate count (misleading!)
    print(f"\n--- Same Gate Count (misleading comparison) ---")
    print(f"{'Polynomial':<25} {'Cycles':>14} {'Runtime':>12} {'Bottleneck':>12}")
    print("-" * 65)
    
    for poly in [VANILLA_ZEROCHECK, JELLYFISH_ZEROCHECK]:
        metrics = sim.simulate(poly, problem_size)
        print(f"{poly.name:<25} {metrics.total_cycles:>14,} "
              f"{metrics.runtime_ms:>11.2f}ms {metrics.bottleneck:>12}")
    
    print("\n⚠️  The above makes Jellyfish look slower because it ignores gate reduction!")
    
    # NEW comparison: accounting for gate reduction
    print(f"\n--- Workload Comparison (accounts for gate reduction) ---")
    print(f"{'Workload':<12} {'V-Gates':>12} {'J-Gates':>12} {'V-Time':>10} {'J-Time':>10} {'Speedup':>12}")
    print("-" * 75)
    
    for workload in ["hash", "ec", "mixed"]:
        result = compare_workload(sim, VANILLA_ZEROCHECK, JELLYFISH_ZEROCHECK,
                                  problem_size, workload)
        
        v_time = f"{result.vanilla_metrics.runtime_ms:.2f}ms"
        j_time = f"{result.jellyfish_metrics.runtime_ms:.2f}ms"
        
        if result.net_speedup >= 1.0:
            speedup_str = f"{result.net_speedup:.2f}x ↑"
        else:
            speedup_str = f"{result.net_speedup:.2f}x ↓"
        
        print(f"{workload:<12} {result.vanilla_gates:>12,} {result.jellyfish_gates:>12,} "
              f"{v_time:>10} {j_time:>10} {speedup_str:>12}")
    
    print("\n✓ Simulator demo complete!")
    print("\nKey insight: When accounting for gate reduction, Jellyfish wins!")


def run_optimizer():
    """Run the gate optimizer demo."""
    print("\n" + "=" * 70)
    print("RUNNING GATE OPTIMIZER")
    print("=" * 70)
    
    from src.optimizer.gates import VANILLA_GATE, JELLYFISH_GATE
    from src.optimizer.core import GateOptimizer
    
    optimizer = GateOptimizer()
    
    print("\nComparing gate counts for x^n computations:")
    print("(Jellyfish has native x^5 support, costs 2.4x per gate)")
    
    print(f"\n{'Exponent':>10} {'Vanilla':>10} {'Jellyfish':>12} "
          f"{'Reduction':>12} {'Net Speedup':>14} {'Winner':>10}")
    print("-" * 75)
    
    for exp in [2, 3, 4, 5, 8, 16, 25]:
        result = optimizer.compare_for_power(exp)
        
        v = result["vanilla_gates"]
        j = result["jellyfish_gates"]
        reduction = result["gate_reduction"]
        speedup = result["net_speedup"]
        winner = result["recommendation"]
        
        # Format speedup
        if speedup >= 1.0:
            speedup_str = f"{speedup:.2f}x ↑"
        else:
            speedup_str = f"{speedup:.2f}x ↓"
        
        print(f"x^{exp:<8} {v:>10} {j:>12} {reduction:>11.1f}x "
              f"{speedup_str:>14} {winner:>10}")
    
    print("\n✓ Optimizer demo complete!")
    print("\nKey insight: Jellyfish wins when gate reduction > 2.4x")


def run_quick_demo():
    """Run a quick demo of all three projects."""
    print("\n" + "=" * 70)
    print("QUICK DEMO - ALL THREE PROJECTS")
    print("=" * 70)
    
    print("\n" + "-" * 70)
    print("1. SUMCHECK VISUALIZER (abbreviated)")
    print("-" * 70)
    
    from src.common.field import PrimeField
    from src.visualizer.mle import MLETable
    from src.visualizer.core import SumCheckVisualizer
    
    field = PrimeField(97)
    a = MLETable("a", [3, 7, 2, 5], field)
    b = MLETable("b", [1, 4, 6, 2], field)
    
    viz = SumCheckVisualizer([a, b], field, seed=42)
    result = viz.run_full_protocol(verbose=False)
    
    print(f"\nSumCheck on f(X) = a(X) × b(X)")
    print(f"  Variables: {viz.num_vars}")
    print(f"  Rounds: {len(result.challenges)}")
    print(f"  Verified: {result.verified}")
    
    print("\n" + "-" * 70)
    print("2. PERFORMANCE SIMULATOR (abbreviated)")
    print("-" * 70)
    
    from src.simulator.hardware import create_zkphire_config
    from src.simulator.polynomials import VANILLA_ZEROCHECK, JELLYFISH_ZEROCHECK
    from src.simulator.core import SumCheckSimulator, compare_workload
    
    config = create_zkphire_config()
    sim = SumCheckSimulator(config)
    
    # Show workload comparison (the right way!)
    result = compare_workload(sim, VANILLA_ZEROCHECK, JELLYFISH_ZEROCHECK,
                              base_gates=2**20, workload_type="hash")
    
    print(f"\nWorkload comparison (hash-heavy, 2^20 base gates):")
    print(f"  Vanilla:   {result.vanilla_gates:,} gates → {result.vanilla_metrics.runtime_ms:.2f} ms")
    print(f"  Jellyfish: {result.jellyfish_gates:,} gates → {result.jellyfish_metrics.runtime_ms:.2f} ms")
    print(f"  Net speedup: {result.net_speedup:.2f}x ({result.winner} wins!)")
    
    print("\n" + "-" * 70)
    print("3. GATE OPTIMIZER (abbreviated)")
    print("-" * 70)
    
    from src.optimizer.gates import VANILLA_GATE, JELLYFISH_GATE
    from src.optimizer.core import GateOptimizer
    
    optimizer = GateOptimizer()
    result = optimizer.compare_for_power(5)
    
    print(f"\nOptimizing x^5 computation:")
    print(f"  Vanilla gates: {result['vanilla_gates']}")
    print(f"  Jellyfish gates: {result['jellyfish_gates']}")
    print(f"  Gate reduction: {result['gate_reduction']:.1f}x")
    print(f"  Net speedup: {result['net_speedup']:.2f}x")
    print(f"  Winner: {result['recommendation']}")
    
    print("\n" + "=" * 70)
    print("QUICK DEMO COMPLETE")
    print("=" * 70)


def main():
    """Main entry point."""
    print_banner()
    
    while True:
        print_menu()
        
        choice = input("Enter your choice: ").strip().lower()
        
        if choice == '1':
            run_visualizer()
        elif choice == '2':
            run_simulator()
        elif choice == '3':
            run_optimizer()
        elif choice == '4':
            run_quick_demo()
        elif choice == 'q':
            print("\nGoodbye!")
            break
        else:
            print("\nInvalid choice. Please try again.")
        
        print()
        input("Press Enter to continue...")
        print("\n" * 2)


if __name__ == "__main__":
    main()
