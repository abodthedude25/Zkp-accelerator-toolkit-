"""
Gate Optimizer Demo

This script demonstrates the gate optimizer with various
computations and workloads.

Run with:
    python -m src.optimizer.demo
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.optimizer.gates import (
    VANILLA_GATE, JELLYFISH_GATE, TURBO_GATE, ULTRA_GATE,
    compare_gates, estimate_gate_reduction,
)
from src.optimizer.computation import (
    Computation, Operation, OpType,
    build_power_computation, build_polynomial_computation,
    analyze_computation,
)
from src.optimizer.core import (
    GateOptimizer, optimize_for_workload,
)


def demo_gate_types():
    """Demonstrate different gate types."""
    print("\n" + "=" * 70)
    print("DEMO 1: GATE TYPE COMPARISON")
    print("=" * 70)
    
    gates = [VANILLA_GATE, JELLYFISH_GATE, TURBO_GATE, ULTRA_GATE]
    
    print(f"\n{'Gate Type':<15} {'Inputs':>8} {'Degree':>8} {'Terms':>8} "
          f"{'Complexity':>12} {'Cost':>8}")
    print("-" * 70)
    
    for gate in gates:
        print(f"{gate.name:<15} {gate.max_inputs:>8} {gate.polynomial_degree:>8} "
              f"{gate.num_polynomial_terms:>8} {gate.complexity_score:>12.0f} "
              f"{gate.compute_cost_factor:>7.1f}x")
    
    print("\nKey insight:")
    print("  Jellyfish gates have 2.4x compute cost per gate,")
    print("  but can reduce gate count significantly for certain operations.")
    print("  Net speedup = gate_reduction / cost_factor")
    print("  Jellyfish wins when gate_reduction > 2.4x")


def demo_power_computations():
    """Demonstrate optimization for power computations."""
    print("\n" + "=" * 70)
    print("DEMO 2: POWER COMPUTATION OPTIMIZATION (x^n)")
    print("=" * 70)
    
    optimizer = GateOptimizer()
    
    print(f"\n{'Exp':>5} {'Vanilla':>9} {'Jellyfish':>11} {'V-Cost':>9} "
          f"{'J-Cost':>9} {'Reduction':>11} {'J-Speedup':>11} {'Winner':>10}")
    print("-" * 90)
    
    for exp in [2, 3, 4, 5, 6, 7, 8, 10, 16, 25, 32]:
        # Use the direct comparison method for accuracy
        comp = optimizer.compare_for_power(exp)
        
        v_gates = comp["vanilla_gates"]
        j_gates = comp["jellyfish_gates"]
        v_cost = comp["vanilla_cost"]
        j_cost = comp["jellyfish_cost"]
        reduction = comp["gate_reduction"]
        speedup = comp["net_speedup"]
        winner = comp["recommendation"]
        
        # Format speedup to show direction
        if speedup >= 1.0:
            speedup_str = f"{speedup:.2f}x ↑"
        else:
            speedup_str = f"{speedup:.2f}x ↓"
        
        print(f"x^{exp:<3} {v_gates:>9} {j_gates:>11} {v_cost:>9.1f} "
              f"{j_cost:>9.1f} {reduction:>10.2f}x {speedup_str:>11} {winner:>10}")
    
    print("\nHow to read this table:")
    print("  - V-Cost = Vanilla gates × 1.0 (vanilla cost factor)")
    print("  - J-Cost = Jellyfish gates × 2.4 (jellyfish cost factor)")
    print("  - J-Speedup = V-Cost / J-Cost (>1.0 means Jellyfish wins)")
    print("\nKey insights:")
    print("  - x^5 is the sweet spot: 3→1 gates = 3.0x reduction")
    print("  - 3.0x reduction / 2.4x cost = 1.25x net speedup for Jellyfish")
    print("  - x^25 uses (x^5)^5, so also benefits from POW5")


def demo_breakeven_analysis():
    """Show where Jellyfish breaks even with Vanilla."""
    print("\n" + "=" * 70)
    print("DEMO 3: BREAK-EVEN ANALYSIS")
    print("=" * 70)
    
    print("\nFor Jellyfish to be worthwhile:")
    print("  net_speedup = gate_reduction / cost_factor > 1.0")
    print("  gate_reduction / 2.4 > 1.0")
    print("  gate_reduction > 2.4")
    print("\nSo Jellyfish needs to reduce gates by MORE than 2.4x to win.")
    
    optimizer = GateOptimizer()
    
    print(f"\n{'Operation':<15} {'V-Gates':>10} {'J-Gates':>10} "
          f"{'Reduction':>12} {'> 2.4x?':>10} {'Winner':>10}")
    print("-" * 70)
    
    test_cases = [
        ("x^2", 2),
        ("x^3", 3),
        ("x^4", 4),
        ("x^5", 5),
        ("x^6", 6),
        ("x^7", 7),
        ("x^8", 8),
        ("x^10", 10),
        ("x^25", 25),
    ]
    
    for name, exp in test_cases:
        comp = optimizer.compare_for_power(exp)
        reduction = comp["gate_reduction"]
        wins = "YES ✓" if reduction > 2.4 else "NO"
        winner = comp["recommendation"]
        
        print(f"{name:<15} {comp['vanilla_gates']:>10} {comp['jellyfish_gates']:>10} "
              f"{reduction:>11.2f}x {wins:>10} {winner:>10}")
    
    print("\nConclusion:")
    print("  - x^5 achieves 3.0x reduction > 2.4x threshold → Jellyfish wins!")
    print("  - x^25 = (x^5)^5: also uses POW5 efficiently")
    print("  - Lower powers don't reduce gates enough to overcome 2.4x cost")


def demo_pattern_detection():
    """Demonstrate pattern detection in computations."""
    print("\n" + "=" * 70)
    print("DEMO 4: PATTERN DETECTION")
    print("=" * 70)
    
    # Create a computation with multiple patterns
    ops = [
        # x^5 pattern (3 operations)
        Operation(OpType.MUL, ["x", "x"], "x2"),
        Operation(OpType.MUL, ["x2", "x2"], "x4"),
        Operation(OpType.MUL, ["x4", "x"], "x5"),
        # y^5 pattern (3 operations)
        Operation(OpType.MUL, ["y", "y"], "y2"),
        Operation(OpType.MUL, ["y2", "y2"], "y4"),
        Operation(OpType.MUL, ["y4", "y"], "y5"),
        # Combining results (3 operations)
        Operation(OpType.MUL, ["x5", "y5"], "t1"),
        Operation(OpType.MUL, ["a", "b"], "t2"),
        Operation(OpType.MUL, ["t1", "t2"], "result"),
    ]
    
    comp = Computation(
        operations=ops,
        inputs={"x", "y", "a", "b"},
        outputs={"result"},
        name="complex_computation"
    )
    
    print(f"\nComputation: Two x^5 operations combined with other products")
    print(f"Total operations: {comp.num_operations}")
    print("\nOperations:")
    for i, op in enumerate(comp.operations):
        print(f"  {i}: {op}")
    
    # Manual analysis
    print("\nManual Analysis:")
    print("  - Vanilla: 9 gates (one per operation)")
    print("  - Jellyfish potential:")
    print("    - x^5: 1 gate (POW5)")
    print("    - y^5: 1 gate (POW5)")
    print("    - t1, t2, result: ~2 gates (can pack)")
    print("    - Total: ~4 gates")
    print("  - Reduction: 9/4 = 2.25x (less than 2.4x, borderline)")


def demo_workload_optimization():
    """Demonstrate optimization for a workload of computations."""
    print("\n" + "=" * 70)
    print("DEMO 5: POSEIDON HASH WORKLOAD")
    print("=" * 70)
    
    print("\nPoseidon hash uses x^5 S-boxes extensively.")
    print("Let's analyze a workload with 10 S-box computations.")
    
    optimizer = GateOptimizer()
    
    # Each S-box is x^5
    num_sboxes = 10
    
    # Calculate totals using direct gate counts
    comp = optimizer.compare_for_power(5)
    
    total_vanilla = comp["vanilla_gates"] * num_sboxes
    total_jellyfish = comp["jellyfish_gates"] * num_sboxes
    total_v_cost = comp["vanilla_cost"] * num_sboxes
    total_j_cost = comp["jellyfish_cost"] * num_sboxes
    
    print(f"\nWorkload: {num_sboxes} × x^5 S-boxes")
    print(f"\n{'Metric':<25} {'Vanilla':>15} {'Jellyfish':>15}")
    print("-" * 55)
    print(f"{'Gates per S-box':<25} {comp['vanilla_gates']:>15} {comp['jellyfish_gates']:>15}")
    print(f"{'Total gates':<25} {total_vanilla:>15} {total_jellyfish:>15}")
    print(f"{'Cost per S-box':<25} {comp['vanilla_cost']:>15.1f} {comp['jellyfish_cost']:>15.1f}")
    print(f"{'Total cost':<25} {total_v_cost:>15.1f} {total_j_cost:>15.1f}")
    print(f"{'Gate reduction':<25} {'-':>15} {comp['gate_reduction']:>14.2f}x")
    print(f"{'Net speedup':<25} {'-':>15} {comp['net_speedup']:>14.2f}x")
    
    print(f"\nResult: Jellyfish provides {comp['net_speedup']:.2f}x speedup")
    print("This is why zkPHIRE excels on hash-heavy workloads!")


def demo_application_comparison():
    """Compare different application types."""
    print("\n" + "=" * 70)
    print("DEMO 6: APPLICATION COMPARISON")
    print("=" * 70)
    
    applications = [
        ("Poseidon Hash (x^5 heavy)", "hash", 1_000_000),
        ("ECDSA (multi-products)", "ec", 500_000),
        ("General Rollup (mixed)", "mixed", 2_000_000),
    ]
    
    print(f"\n{'Application':<30} {'Vanilla':>12} {'Jellyfish':>12} "
          f"{'Reduction':>12} {'Net Speedup':>12}")
    print("-" * 80)
    
    for name, workload_type, vanilla_gates in applications:
        est = estimate_gate_reduction(vanilla_gates, JELLYFISH_GATE, workload_type)
        
        print(f"{name:<30} {est['original_gates']:>12,} {est['new_gates']:>12,} "
              f"{est['reduction_factor']:>11.1f}x {est['net_speedup']:>11.1f}x")
    
    print("\nKey Insights from zkPHIRE paper:")
    print("  - Hash functions: 8x gate reduction → 3.3x net speedup")
    print("  - EC operations: 4x gate reduction → 1.7x net speedup")
    print("  - Mixed workloads: 5x gate reduction → 2.1x net speedup")


def demo_summary():
    """Summary of key findings."""
    print("\n" + "=" * 70)
    print("SUMMARY: WHEN TO USE JELLYFISH GATES")
    print("=" * 70)
    
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                    JELLYFISH vs VANILLA DECISION                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Jellyfish cost factor: 2.4x per gate                               │
│  Break-even point: gate reduction must exceed 2.4x                  │
│                                                                     │
│  USE JELLYFISH FOR:                                                 │
│    ✓ x^5 operations (Poseidon S-boxes): 3.0x reduction → 1.25x win │
│    ✓ x^25 operations: uses POW5 efficiently                        │
│    ✓ Hash-heavy workloads: up to 3.3x net speedup                  │
│    ✓ 4-way products: uses QUAD_MUL capability                      │
│                                                                     │
│  USE VANILLA FOR:                                                   │
│    ✓ x^2, x^3, x^4: not enough reduction to beat 2.4x cost         │
│    ✓ Simple arithmetic: additions, single multiplications          │
│    ✓ Low-complexity circuits: overhead not justified               │
│                                                                     │
│  WHY THIS MATTERS FOR zkPHIRE:                                      │
│    - zkPHIRE's programmable SumCheck handles high-degree gates      │
│    - Enables 8-32x gate reduction for suitable workloads            │
│    - Net 3-13x speedup despite increased per-gate complexity        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")


def main():
    """Run all demos."""
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 17 + "GATE OPTIMIZER DEMONSTRATION" + " " * 22 + "║")
    print("╚" + "═" * 68 + "╝")
    
    print("\nThis demo shows how to optimize gate configurations for ZKP circuits.")
    print("The key question: When does Jellyfish beat Vanilla?")
    print("\n" + "─" * 70)
    
    demos = [
        ("Gate Type Comparison", demo_gate_types),
        ("Power Computations", demo_power_computations),
        ("Break-Even Analysis", demo_breakeven_analysis),
        ("Pattern Detection", demo_pattern_detection),
        ("Poseidon Workload", demo_workload_optimization),
        ("Application Comparison", demo_application_comparison),
        ("Summary", demo_summary),
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


if __name__ == "__main__":
    main()
