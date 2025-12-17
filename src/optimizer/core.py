"""
Gate Optimization Core Implementation.

This module implements the gate optimizer that finds optimal gate
configurations for ZKP circuits.

The optimizer:
    1. Analyzes computation patterns
    2. Maps operations to different gate types
    3. Compares resulting circuit sizes
    4. Recommends the best configuration

Key Insight from zkPHIRE:
    Even though Jellyfish gates are 2.4x more expensive per gate,
    they can reduce gate count by 8-32x, yielding net 3-13x speedup.
    
    The optimizer helps quantify this tradeoff for specific computations.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set

from .gates import GateType, VANILLA_GATE, JELLYFISH_GATE, GateCapability
from .computation import (
    Computation, Operation, OpType,
    analyze_computation, Pattern
)


@dataclass
class GateMapping:
    """
    Result of mapping a computation to a specific gate type.
    
    Captures how many gates are needed and the associated costs.
    
    Attributes:
        gate_type: The gate type used
        num_gates: Total gates required
        operations_per_gate: Which operations map to each gate
        unmapped_operations: Operations that couldn't be mapped
    """
    gate_type: GateType
    num_gates: int
    operations_per_gate: List[List[int]] = field(default_factory=list)
    unmapped_operations: List[int] = field(default_factory=list)
    
    @property
    def polynomial_complexity(self) -> float:
        """
        Estimate SumCheck polynomial complexity.
        
        More gates × higher degree = more work.
        """
        return self.num_gates * self.gate_type.complexity_score
    
    @property
    def total_cost(self) -> float:
        """
        Total cost accounting for per-gate compute cost.
        
        This is the key metric for comparing gate types:
        total_cost = num_gates × compute_cost_factor
        """
        return self.num_gates * self.gate_type.compute_cost_factor
    
    def summary(self) -> str:
        return (
            f"GateMapping({self.gate_type.name}):\n"
            f"  Gates: {self.num_gates}\n"
            f"  Polynomial degree: {self.gate_type.polynomial_degree}\n"
            f"  Polynomial terms: {self.gate_type.num_polynomial_terms}\n"
            f"  Complexity score: {self.polynomial_complexity:.0f}\n"
            f"  Cost factor: {self.gate_type.compute_cost_factor}x\n"
            f"  Total cost: {self.total_cost:.1f}"
        )


@dataclass
class OptimizationResult:
    """
    Complete result from gate optimization.
    
    Contains all mappings evaluated and the recommendation.
    """
    computation: Computation
    analysis: Dict
    mappings: Dict[str, GateMapping]
    recommendation: str
    speedup: float  # Best speedup vs vanilla (can be < 1 if vanilla is better)
    
    def summary(self) -> str:
        lines = [
            f"OptimizationResult for '{self.computation.name}':",
            f"  Recommendation: {self.recommendation}",
            f"  Speedup vs Vanilla: {self.speedup:.2f}x",
            "",
            "  Mappings:"
        ]
        
        for name, mapping in self.mappings.items():
            lines.append(f"    {name}: {mapping.num_gates} gates, cost={mapping.total_cost:.1f}")
        
        return "\n".join(lines)


class GateOptimizer:
    """
    Optimizer for finding the best gate configuration.
    
    Takes a computation and evaluates different gate types to find
    the configuration that minimizes total proving cost.
    
    Usage:
        >>> optimizer = GateOptimizer()
        >>> comp = build_power_computation(5)
        >>> result = optimizer.optimize(comp)
        >>> print(result.recommendation)
    """
    
    def __init__(self, gate_types: Optional[List[GateType]] = None):
        """
        Initialize optimizer with available gate types.
        
        Args:
            gate_types: List of gate types to consider.
                       Defaults to [VANILLA_GATE, JELLYFISH_GATE]
        """
        self.gate_types = gate_types or [VANILLA_GATE, JELLYFISH_GATE]
    
    def optimize(self, computation: Computation) -> OptimizationResult:
        """
        Find optimal gate configuration for a computation.
        
        Args:
            computation: The computation to optimize
            
        Returns:
            OptimizationResult with recommendation
        """
        # Analyze the computation
        analysis = analyze_computation(computation)
        
        # Map to each gate type
        mappings = {}
        for gate_type in self.gate_types:
            mapping = self.map_to_gate(computation, gate_type, analysis)
            mappings[gate_type.name] = mapping
        
        # Find best mapping (lowest total cost)
        best_name, best_mapping = min(
            mappings.items(),
            key=lambda x: x[1].total_cost
        )
        
        # Calculate speedup vs vanilla
        vanilla_mapping = mappings.get("Vanilla")
        if vanilla_mapping:
            vanilla_cost = vanilla_mapping.total_cost
            speedup = vanilla_cost / best_mapping.total_cost if best_mapping.total_cost > 0 else 1.0
        else:
            speedup = 1.0
        
        return OptimizationResult(
            computation=computation,
            analysis=analysis,
            mappings=mappings,
            recommendation=best_name,
            speedup=speedup,
        )
    
    def map_to_gate(self, computation: Computation, gate_type: GateType,
                    analysis: Optional[Dict] = None) -> GateMapping:
        """
        Map a computation to a specific gate type.
        
        Args:
            computation: The computation
            gate_type: Target gate type
            analysis: Pre-computed analysis (optional)
            
        Returns:
            GateMapping showing how many gates needed
        """
        if analysis is None:
            analysis = analyze_computation(computation)
        
        # Check if this is a pure power computation
        power_exp = self._detect_power_exponent(computation)
        if power_exp is not None:
            num_gates = gate_type.gates_for_power(power_exp)
            return GateMapping(
                gate_type=gate_type,
                num_gates=num_gates,
            )
        
        # General mapping
        if gate_type.name == "Vanilla":
            return self._map_to_vanilla(computation, analysis)
        elif gate_type.name == "Jellyfish":
            return self._map_to_jellyfish(computation, analysis)
        else:
            return self._map_generic(computation, gate_type, analysis)
    
    def _detect_power_exponent(self, computation: Computation) -> Optional[int]:
        """
        Detect if computation is computing x^n for some n.
        
        Returns the exponent if detected, None otherwise.
        """
        # Check the computation name first
        name = computation.name.lower()
        if name.startswith("x^") or "^" in name:
            try:
                # Try to extract exponent from name like "x^5"
                parts = name.split("^")
                if len(parts) == 2:
                    return int(parts[1])
            except (ValueError, IndexError):
                pass
        
        # Analyze the operation pattern
        ops = computation.operations
        if not ops:
            return None
        
        # Count multiplications involving the same base
        if len(computation.inputs) != 1:
            return None
        
        input_var = list(computation.inputs)[0]
        
        # Trace through to count effective power
        # For now, use a simple heuristic based on number of MUL ops
        mul_count = sum(1 for op in ops if op.op_type == OpType.MUL)
        
        # A chain of n-1 multiplications gives x^n
        # But we use square-and-multiply, so it's more complex
        # Check for squaring pattern
        
        all_squaring = all(
            op.op_type == OpType.MUL and len(op.inputs) == 2 and op.inputs[0] == op.inputs[1]
            for op in ops if op.op_type == OpType.MUL
        )
        
        if all_squaring and mul_count > 0:
            # Pure squaring chain: x², x⁴, x⁸, ...
            return 2 ** mul_count
        
        # Mixed squaring and multiplication
        # Count based on operation structure
        if mul_count == 1:
            return 2
        elif mul_count == 2:
            return 4  # Could be 3 or 4
        elif mul_count == 3:
            return 5  # Could be 5, 6, 7, or 8
        
        return None
    
    def _map_to_vanilla(self, computation: Computation,
                        analysis: Dict) -> GateMapping:
        """
        Map to vanilla gates (1 operation per gate typically).
        """
        num_gates = 0
        ops_per_gate = []
        
        for i, op in enumerate(computation.operations):
            if op.op_type in [OpType.ADD, OpType.SUB, OpType.MUL, OpType.CONST]:
                num_gates += 1
                ops_per_gate.append([i])
            elif op.op_type == OpType.POW:
                gates_needed = VANILLA_GATE.gates_for_power(op.exponent)
                num_gates += gates_needed
                ops_per_gate.append([i])
            elif op.op_type == OpType.DIV:
                num_gates += 2  # Inversion + multiplication
                ops_per_gate.append([i])
            else:
                num_gates += 1
                ops_per_gate.append([i])
        
        return GateMapping(
            gate_type=VANILLA_GATE,
            num_gates=max(1, num_gates),
            operations_per_gate=ops_per_gate,
        )
    
    def _map_to_jellyfish(self, computation: Computation,
                          analysis: Dict) -> GateMapping:
        """
        Map to Jellyfish gates (can pack multiple operations).
        """
        num_gates = 0
        ops_per_gate = []
        used_ops: Set[int] = set()
        
        patterns = analysis.get('patterns', {})
        
        # Handle power chains with POW5
        for pattern in patterns.get('power_chains', []):
            exp = pattern.details.get('exponent', 0)
            if exp > 0:
                gates = JELLYFISH_GATE.gates_for_power(exp)
                num_gates += gates
                ops_per_gate.append(pattern.operations)
                used_ops.update(pattern.operations)
        
        # Handle multi-way products
        for pattern in patterns.get('multiway_products', []):
            width = pattern.details.get('width', 0)
            # Jellyfish can do 4-way products in one gate
            gates_needed = max(1, (width + 3) // 4)
            num_gates += gates_needed
            ops_per_gate.append(pattern.operations)
            used_ops.update(pattern.operations)
        
        # Handle remaining operations with packing
        remaining = []
        for i, op in enumerate(computation.operations):
            if i in used_ops:
                continue
            
            if op.op_type == OpType.POW:
                num_gates += JELLYFISH_GATE.gates_for_power(op.exponent)
            elif op.op_type in [OpType.ADD, OpType.SUB, OpType.MUL, OpType.CONST]:
                remaining.append(i)
            else:
                num_gates += 1
        
        # Pack remaining simple operations (can fit ~2-3 per Jellyfish gate)
        if remaining:
            num_gates += max(1, (len(remaining) + 2) // 3)
        
        return GateMapping(
            gate_type=JELLYFISH_GATE,
            num_gates=max(1, num_gates),
            operations_per_gate=ops_per_gate,
        )
    
    def _map_generic(self, computation: Computation, gate_type: GateType,
                     analysis: Dict) -> GateMapping:
        """Generic mapping for custom gate types."""
        ops_per_gate = 1 + (gate_type.max_inputs - 2) * 0.5
        num_gates = max(1, int(computation.num_operations / ops_per_gate))
        
        return GateMapping(
            gate_type=gate_type,
            num_gates=num_gates,
        )
    
    def compare(self, computation: Computation) -> Dict:
        """
        Compare all gate types for a computation.
        
        Returns detailed comparison metrics.
        """
        result = self.optimize(computation)
        
        comparison = {
            "computation": computation.name,
            "num_operations": computation.num_operations,
            "gate_comparisons": {},
        }
        
        vanilla_mapping = result.mappings.get("Vanilla")
        
        for name, mapping in result.mappings.items():
            comp_data = {
                "gates": mapping.num_gates,
                "cost_factor": mapping.gate_type.compute_cost_factor,
                "total_cost": mapping.total_cost,
            }
            
            if vanilla_mapping and name != "Vanilla":
                gate_reduction = vanilla_mapping.num_gates / mapping.num_gates if mapping.num_gates > 0 else 1
                cost_ratio = vanilla_mapping.total_cost / mapping.total_cost if mapping.total_cost > 0 else 1
                comp_data["gate_reduction"] = gate_reduction
                comp_data["net_speedup"] = cost_ratio
            
            comparison["gate_comparisons"][name] = comp_data
        
        comparison["recommendation"] = result.recommendation
        comparison["best_speedup"] = result.speedup
        
        return comparison
    
    def compare_for_power(self, exponent: int) -> Dict:
        """
        Specifically compare gate types for computing x^n.
        
        This uses the direct gate counting methods for accuracy.
        """
        vanilla_gates = VANILLA_GATE.gates_for_power(exponent)
        jellyfish_gates = JELLYFISH_GATE.gates_for_power(exponent)
        
        vanilla_cost = vanilla_gates * VANILLA_GATE.compute_cost_factor
        jellyfish_cost = jellyfish_gates * JELLYFISH_GATE.compute_cost_factor
        
        gate_reduction = vanilla_gates / jellyfish_gates if jellyfish_gates > 0 else float('inf')
        net_speedup = vanilla_cost / jellyfish_cost if jellyfish_cost > 0 else float('inf')
        
        if jellyfish_cost < vanilla_cost:
            recommendation = "Jellyfish"
        else:
            recommendation = "Vanilla"
        
        return {
            "exponent": exponent,
            "vanilla_gates": vanilla_gates,
            "jellyfish_gates": jellyfish_gates,
            "vanilla_cost": vanilla_cost,
            "jellyfish_cost": jellyfish_cost,
            "gate_reduction": gate_reduction,
            "net_speedup": net_speedup,
            "recommendation": recommendation,
        }


def optimize_for_workload(computations: List[Computation],
                          gate_types: Optional[List[GateType]] = None) -> Dict:
    """
    Optimize gate selection for a workload of multiple computations.
    """
    optimizer = GateOptimizer(gate_types)
    
    results = []
    total_vanilla = 0
    total_best = 0
    total_vanilla_cost = 0.0
    total_best_cost = 0.0
    
    for comp in computations:
        result = optimizer.optimize(comp)
        results.append(result)
        
        vanilla_mapping = result.mappings.get("Vanilla")
        best_mapping = result.mappings[result.recommendation]
        
        if vanilla_mapping:
            total_vanilla += vanilla_mapping.num_gates
            total_vanilla_cost += vanilla_mapping.total_cost
        
        total_best += best_mapping.num_gates
        total_best_cost += best_mapping.total_cost
    
    # Aggregate recommendation
    recommendations = [r.recommendation for r in results]
    from collections import Counter
    most_common = Counter(recommendations).most_common(1)[0][0]
    
    return {
        "num_computations": len(computations),
        "per_computation_results": results,
        "total_vanilla_gates": total_vanilla,
        "total_optimized_gates": total_best,
        "total_vanilla_cost": total_vanilla_cost,
        "total_optimized_cost": total_best_cost,
        "overall_gate_reduction": total_vanilla / total_best if total_best > 0 else 1.0,
        "overall_speedup": total_vanilla_cost / total_best_cost if total_best_cost > 0 else 1.0,
        "recommendation": most_common,
    }
