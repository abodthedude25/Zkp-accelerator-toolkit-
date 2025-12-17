"""
Gate Type Definitions for ZKP Circuit Optimization.

This module defines the characteristics of different gate types used
in ZKP circuits. Understanding gate types is crucial for optimizing
circuit design.

Key Gate Types:
    - Vanilla: Standard Plonk gates (degree 4, 2 inputs)
    - Jellyfish: High-degree custom gates (degree ~11, 4 inputs)
    - Custom: User-defined gates for specific applications

The key insight from zkPHIRE:
    Higher-degree gates can express more computation per gate,
    reducing total gate count by 8-32x. This reduces:
    - MLE table sizes (less memory)
    - SumCheck rounds (fewer iterations)
    - MSM sizes (smaller commitments)
    
    Even though each gate requires more computation (2.4x),
    the reduction in gate count yields net 3-13x speedup.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional
from enum import Enum


class GateCapability(Enum):
    """Capabilities that a gate type may support."""
    ADD = "add"           # Addition
    MUL = "mul"           # Multiplication
    CONST = "const"       # Constant multiplication
    POW5 = "pow5"         # Fifth power (x^5)
    POW7 = "pow7"         # Seventh power (x^7)
    QUAD_MUL = "quad_mul" # Four-way product (a*b*c*d)
    EC_ADD = "ec_add"     # Elliptic curve addition
    EC_MUL = "ec_mul"     # Elliptic curve scalar multiplication


@dataclass
class GateType:
    """
    Defines a type of gate that can be used in ZKP circuits.
    
    A gate type determines:
        - What operations can be performed in a single gate
        - How complex the gate constraint polynomial is
        - The tradeoff between gate count and per-gate cost
    
    Attributes:
        name: Gate type identifier
        max_inputs: Maximum wire inputs to the gate
        max_outputs: Maximum wire outputs from the gate
        polynomial_degree: Degree of the gate constraint polynomial
        num_polynomial_terms: Number of terms in the constraint
        capabilities: Set of supported operations
        description: Human-readable description
    """
    name: str
    max_inputs: int
    max_outputs: int
    polynomial_degree: int
    num_polynomial_terms: int
    capabilities: Set[GateCapability]
    description: str = ""
    
    # Cost factors (relative to vanilla)
    compute_cost_factor: float = 1.0
    
    @property
    def num_extensions(self) -> int:
        """Number of extension points needed in SumCheck."""
        return self.polynomial_degree + 1
    
    @property
    def complexity_score(self) -> float:
        """
        Overall complexity score for comparing gate types.
        
        Higher = more complex per gate, but may enable fewer total gates.
        """
        return self.polynomial_degree * self.num_polynomial_terms
    
    def supports(self, capability: GateCapability) -> bool:
        """Check if this gate supports a capability."""
        return capability in self.capabilities
    
    def can_compute_power(self, exponent: int) -> bool:
        """Check if this gate can compute x^exponent directly."""
        if exponent <= 1:
            return True
        if exponent == 2:
            return self.supports(GateCapability.MUL)
        if exponent == 5 and self.supports(GateCapability.POW5):
            return True
        if exponent == 7 and self.supports(GateCapability.POW7):
            return True
        return False
    
    def gates_for_power(self, exponent: int) -> int:
        """
        Calculate gates needed to compute x^exponent.
        
        Uses optimal addition chains for common cases.
        """
        if exponent <= 1:
            return 0
        
        # Check for native support
        if self.name == "Jellyfish" and self.supports(GateCapability.POW5):
            return self._jellyfish_gates_for_power(exponent)
        
        # Vanilla: use square-and-multiply
        return self._vanilla_gates_for_power(exponent)
    
    def _vanilla_gates_for_power(self, exponent: int) -> int:
        """
        Vanilla gates needed for x^n using square-and-multiply.
        
        Each gate can do one multiplication.
        """
        if exponent <= 1:
            return 0
        elif exponent == 2:
            return 1  # x * x
        elif exponent == 3:
            return 2  # x², x²·x
        elif exponent == 4:
            return 2  # x², x⁴
        elif exponent == 5:
            return 3  # x², x⁴, x⁴·x
        elif exponent == 6:
            return 3  # x², x³, x⁶ or x², x⁴, x⁶
        elif exponent == 7:
            return 4  # x², x³, x⁶, x⁷ or x², x⁴, x⁶, x⁷
        elif exponent == 8:
            return 3  # x², x⁴, x⁸
        elif exponent == 16:
            return 4  # x², x⁴, x⁸, x¹⁶
        elif exponent == 32:
            return 5  # x², x⁴, x⁸, x¹⁶, x³²
        else:
            # General case: use binary method
            # Number of squarings + number of extra multiplications
            bits = exponent.bit_length() - 1  # Number of squarings
            ones = bin(exponent).count('1') - 1  # Extra multiplications
            return bits + ones
    
    def _jellyfish_gates_for_power(self, exponent: int) -> int:
        """
        Jellyfish gates needed for x^n.
        
        Jellyfish can:
        - Compute x^5 in one gate (native POW5)
        - Compute 4-way products in one gate (QUAD_MUL)
        """
        if exponent <= 1:
            return 0
        elif exponent == 2:
            return 1  # x * x
        elif exponent == 3:
            return 1  # Can pack x² and x²·x, or use 3-way
        elif exponent == 4:
            return 1  # x·x·x·x in one gate (QUAD_MUL)
        elif exponent == 5:
            return 1  # Native POW5!
        elif exponent == 6:
            return 2  # x⁵, x⁵·x
        elif exponent == 7:
            return 2  # x⁵, x⁵·x² (with packing)
        elif exponent == 8:
            return 2  # x⁴, x⁴·x⁴
        elif exponent == 10:
            return 2  # x⁵, x⁵·x⁵
        elif exponent == 16:
            return 3  # x⁴, x⁸, x¹⁶ (with quad mul)
        elif exponent == 25:
            return 2  # x⁵, (x⁵)⁵ = x²⁵
        elif exponent == 32:
            return 4  # x⁴, x⁸, x¹⁶, x³²
        else:
            # General strategy: decompose using base-5
            import math
            # Rough approximation: log base 5
            return max(1, int(math.ceil(math.log(exponent) / math.log(5))))
    
    def summary(self) -> str:
        """Return summary string."""
        caps = ", ".join(c.value for c in self.capabilities)
        return (
            f"GateType '{self.name}':\n"
            f"  {self.description}\n"
            f"  Inputs: {self.max_inputs}, Outputs: {self.max_outputs}\n"
            f"  Polynomial degree: {self.polynomial_degree}\n"
            f"  Polynomial terms: {self.num_polynomial_terms}\n"
            f"  Extensions needed: {self.num_extensions}\n"
            f"  Complexity score: {self.complexity_score}\n"
            f"  Compute cost factor: {self.compute_cost_factor}x\n"
            f"  Capabilities: {caps}"
        )
    
    def __repr__(self) -> str:
        return f"GateType('{self.name}', degree={self.polynomial_degree})"


# =============================================================================
# PREDEFINED GATE TYPES
# =============================================================================

VANILLA_GATE = GateType(
    name="Vanilla",
    max_inputs=2,
    max_outputs=1,
    polynomial_degree=4,
    num_polynomial_terms=5,
    capabilities={
        GateCapability.ADD,
        GateCapability.MUL,
        GateCapability.CONST,
    },
    description="Standard Plonk gate: f = qL*w1 + qR*w2 + qM*w1*w2 - qO*w3 + qC",
    compute_cost_factor=1.0,
)

JELLYFISH_GATE = GateType(
    name="Jellyfish",
    max_inputs=4,
    max_outputs=1,
    polynomial_degree=7,
    num_polynomial_terms=13,
    capabilities={
        GateCapability.ADD,
        GateCapability.MUL,
        GateCapability.CONST,
        GateCapability.POW5,
        GateCapability.QUAD_MUL,
    },
    description="Jellyfish high-degree gate: supports x^5, 4-way products",
    compute_cost_factor=2.4,
)

TURBO_GATE = GateType(
    name="TurboPlonk",
    max_inputs=3,
    max_outputs=1,
    polynomial_degree=5,
    num_polynomial_terms=8,
    capabilities={
        GateCapability.ADD,
        GateCapability.MUL,
        GateCapability.CONST,
    },
    description="TurboPlonk gate: 3 inputs for more flexibility",
    compute_cost_factor=1.5,
)

ULTRA_GATE = GateType(
    name="UltraPlonk",
    max_inputs=4,
    max_outputs=1,
    polynomial_degree=6,
    num_polynomial_terms=10,
    capabilities={
        GateCapability.ADD,
        GateCapability.MUL,
        GateCapability.CONST,
        GateCapability.QUAD_MUL,
    },
    description="UltraPlonk gate: 4 inputs, lookup support",
    compute_cost_factor=2.0,
)


def create_custom_gate(
    name: str,
    max_inputs: int,
    polynomial_degree: int,
    capabilities: Set[GateCapability],
    description: str = "",
    compute_cost_factor: float = None
) -> GateType:
    """
    Create a custom gate type.
    
    Args:
        name: Gate name
        max_inputs: Maximum inputs
        polynomial_degree: Constraint polynomial degree
        capabilities: Set of supported operations
        description: Human-readable description
        compute_cost_factor: Cost relative to vanilla (default: degree/4)
        
    Returns:
        New GateType
    """
    # Estimate terms based on degree and inputs
    num_terms = max_inputs * 2 + polynomial_degree
    
    # Estimate cost based on degree if not provided
    if compute_cost_factor is None:
        compute_cost_factor = polynomial_degree / 4.0
    
    return GateType(
        name=name,
        max_inputs=max_inputs,
        max_outputs=1,
        polynomial_degree=polynomial_degree,
        num_polynomial_terms=num_terms,
        capabilities=capabilities,
        description=description,
        compute_cost_factor=compute_cost_factor,
    )


# =============================================================================
# GATE COMPARISON UTILITIES
# =============================================================================

def compare_gates(gate1: GateType, gate2: GateType) -> Dict:
    """Compare two gate types."""
    return {
        "gate1": gate1.name,
        "gate2": gate2.name,
        "degree_ratio": gate2.polynomial_degree / gate1.polynomial_degree,
        "terms_ratio": gate2.num_polynomial_terms / gate1.num_polynomial_terms,
        "complexity_ratio": gate2.complexity_score / gate1.complexity_score,
        "cost_ratio": gate2.compute_cost_factor / gate1.compute_cost_factor,
        "input_ratio": gate2.max_inputs / gate1.max_inputs,
    }


def estimate_gate_reduction(
    vanilla_gates: int,
    target_gate: GateType,
    workload_type: str = "mixed"
) -> Dict:
    """
    Estimate gate count reduction when using custom gates.
    
    Args:
        vanilla_gates: Number of vanilla gates in original circuit
        target_gate: Target gate type to use
        workload_type: "hash" (pow5-heavy), "ec" (quad_mul-heavy), or "mixed"
        
    Returns:
        Dict with estimated gate counts and speedup
    """
    reduction_factors = {
        "Jellyfish": {"hash": 8.0, "ec": 4.0, "mixed": 5.0},
        "TurboPlonk": {"hash": 2.0, "ec": 2.0, "mixed": 2.0},
        "UltraPlonk": {"hash": 4.0, "ec": 3.0, "mixed": 3.5},
    }
    
    if target_gate.name in reduction_factors:
        reduction = reduction_factors[target_gate.name].get(workload_type, 2.0)
    else:
        reduction = target_gate.polynomial_degree / VANILLA_GATE.polynomial_degree
    
    new_gates = int(vanilla_gates / reduction)
    net_speedup = reduction / target_gate.compute_cost_factor
    
    return {
        "original_gates": vanilla_gates,
        "new_gates": new_gates,
        "reduction_factor": reduction,
        "cost_factor": target_gate.compute_cost_factor,
        "net_speedup": net_speedup,
        "mle_reduction": reduction,
        "rounds_saved": int(vanilla_gates).bit_length() - int(new_gates).bit_length(),
    }
