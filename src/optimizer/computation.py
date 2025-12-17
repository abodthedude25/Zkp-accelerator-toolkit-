"""
Computation Representation for Gate Optimization.

This module provides classes for representing computations that need
to be proven in ZKP circuits. Understanding the structure of a computation
helps find optimal gate configurations.

A computation is a DAG (directed acyclic graph) of operations:
    - Inputs: Public/private values entering the computation
    - Operations: Add, multiply, power, etc.
    - Outputs: Final results to be proven

Pattern Recognition:
    Certain patterns benefit greatly from custom gates:
    - Power chains (x^n): Jellyfish can do x^5 in one gate
    - Multi-way products: a*b*c*d in one gate
    - Repeated subexpressions: Reuse intermediate results
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Tuple
from enum import Enum


class OpType(Enum):
    """Types of operations in a computation."""
    ADD = "add"       # a + b
    SUB = "sub"       # a - b
    MUL = "mul"       # a * b
    DIV = "div"       # a / b (requires inverse)
    NEG = "neg"       # -a
    CONST = "const"   # Constant value
    POW = "pow"       # a^n (power)
    COPY = "copy"     # Wire copy (for fan-out)


@dataclass
class Operation:
    """
    A single operation in the computation.
    
    Operations form the nodes of the computation DAG.
    
    Attributes:
        op_type: Type of operation (ADD, MUL, etc.)
        inputs: List of input variable names
        output: Output variable name
        constant: For CONST operations, the constant value
        exponent: For POW operations, the exponent
        
    Example:
        >>> # Multiplication: x2 = x * x
        >>> op = Operation(OpType.MUL, ["x", "x"], "x2")
        >>> 
        >>> # Constant: c = 5
        >>> op = Operation(OpType.CONST, [], "c", constant=5)
        >>> 
        >>> # Power: y = x^5
        >>> op = Operation(OpType.POW, ["x"], "y", exponent=5)
    """
    op_type: OpType
    inputs: List[str]
    output: str
    constant: Optional[int] = None
    exponent: Optional[int] = None
    
    def __post_init__(self):
        """Validate operation."""
        if self.op_type == OpType.CONST and self.constant is None:
            raise ValueError("CONST operation requires constant value")
        if self.op_type == OpType.POW and self.exponent is None:
            raise ValueError("POW operation requires exponent")
    
    @property
    def num_inputs(self) -> int:
        """Number of inputs to this operation."""
        return len(self.inputs)
    
    @property
    def is_binary(self) -> bool:
        """Check if this is a binary operation."""
        return self.num_inputs == 2
    
    @property
    def is_unary(self) -> bool:
        """Check if this is a unary operation."""
        return self.num_inputs == 1
    
    @property
    def is_power(self) -> bool:
        """Check if this is a power operation."""
        return self.op_type == OpType.POW
    
    @property
    def is_squaring(self) -> bool:
        """Check if this multiplies a value by itself."""
        if self.op_type == OpType.MUL and len(self.inputs) == 2:
            return self.inputs[0] == self.inputs[1]
        return False
    
    def __repr__(self) -> str:
        if self.op_type == OpType.CONST:
            return f"{self.output} = {self.constant}"
        elif self.op_type == OpType.POW:
            return f"{self.output} = {self.inputs[0]}^{self.exponent}"
        elif self.op_type == OpType.ADD:
            return f"{self.output} = {self.inputs[0]} + {self.inputs[1]}"
        elif self.op_type == OpType.SUB:
            return f"{self.output} = {self.inputs[0]} - {self.inputs[1]}"
        elif self.op_type == OpType.MUL:
            return f"{self.output} = {' * '.join(self.inputs)}"
        elif self.op_type == OpType.DIV:
            return f"{self.output} = {self.inputs[0]} / {self.inputs[1]}"
        elif self.op_type == OpType.NEG:
            return f"{self.output} = -{self.inputs[0]}"
        else:
            return f"{self.output} = {self.op_type.value}({', '.join(self.inputs)})"


@dataclass
class Computation:
    """
    A computation to be proven in a ZKP circuit.
    
    Represents a DAG of operations from inputs to outputs.
    The computation can be analyzed for patterns that benefit
    from custom gate types.
    
    Attributes:
        operations: List of operations in topological order
        inputs: Set of input variable names
        outputs: Set of output variable names
        name: Optional name for the computation
        
    Example:
        >>> # Compute x^5
        >>> comp = Computation(
        ...     operations=[
        ...         Operation(OpType.MUL, ["x", "x"], "x2"),
        ...         Operation(OpType.MUL, ["x2", "x2"], "x4"),
        ...         Operation(OpType.MUL, ["x4", "x"], "x5"),
        ...     ],
        ...     inputs={"x"},
        ...     outputs={"x5"}
        ... )
    """
    operations: List[Operation]
    inputs: Set[str]
    outputs: Set[str]
    name: str = "computation"
    
    def __len__(self) -> int:
        """Number of operations."""
        return len(self.operations)
    
    @property
    def num_operations(self) -> int:
        """Total operation count."""
        return len(self.operations)
    
    @property
    def num_inputs(self) -> int:
        """Number of inputs."""
        return len(self.inputs)
    
    @property
    def num_outputs(self) -> int:
        """Number of outputs."""
        return len(self.outputs)
    
    @property
    def all_variables(self) -> Set[str]:
        """All variable names in the computation."""
        vars = set(self.inputs)
        for op in self.operations:
            vars.update(op.inputs)
            vars.add(op.output)
        return vars
    
    def operation_counts(self) -> Dict[OpType, int]:
        """Count operations by type."""
        counts: Dict[OpType, int] = {}
        for op in self.operations:
            counts[op.op_type] = counts.get(op.op_type, 0) + 1
        return counts
    
    def get_producers(self) -> Dict[str, Operation]:
        """Map each variable to the operation that produces it."""
        producers = {}
        for op in self.operations:
            producers[op.output] = op
        return producers
    
    def get_consumers(self) -> Dict[str, List[Operation]]:
        """Map each variable to operations that consume it."""
        consumers: Dict[str, List[Operation]] = {v: [] for v in self.all_variables}
        for op in self.operations:
            for inp in op.inputs:
                if inp in consumers:
                    consumers[inp].append(op)
        return consumers
    
    def get_operation_index(self, output: str) -> Optional[int]:
        """Find index of operation that produces given output."""
        for i, op in enumerate(self.operations):
            if op.output == output:
                return i
        return None
    
    def summary(self) -> str:
        """Return summary string."""
        op_counts = self.operation_counts()
        counts_str = ", ".join(f"{t.value}: {c}" for t, c in op_counts.items())
        
        return (
            f"Computation '{self.name}':\n"
            f"  Inputs: {self.num_inputs}\n"
            f"  Outputs: {self.num_outputs}\n"
            f"  Operations: {self.num_operations}\n"
            f"  Operation types: {{{counts_str}}}"
        )
    
    def __repr__(self) -> str:
        return f"Computation('{self.name}', ops={self.num_operations})"


# =============================================================================
# PATTERN DETECTION
# =============================================================================

@dataclass
class Pattern:
    """A detected pattern in a computation."""
    pattern_type: str
    operations: List[int]  # Indices of operations in pattern
    details: Dict = field(default_factory=dict)


def detect_power_chains(comp: Computation) -> List[Pattern]:
    """
    Detect power chain patterns (x * x * x * ...).
    
    Power chains are sequences of multiplications that compute x^n.
    These benefit greatly from high-degree gates that support POW5.
    
    Returns list of detected power chain patterns.
    """
    patterns = []
    producers = comp.get_producers()
    
    for i, op in enumerate(comp.operations):
        if op.op_type != OpType.MUL:
            continue
        
        # Check if this is squaring (x * x)
        if op.is_squaring:
            # Trace back to find the chain
            chain_length = _trace_power_chain(comp, i)
            if chain_length >= 3:
                patterns.append(Pattern(
                    pattern_type="power_chain",
                    operations=[i],
                    details={
                        "exponent": chain_length,
                        "base_var": op.inputs[0],
                        "vanilla_gates": chain_length - 1,
                        "jellyfish_gates": 1 if chain_length <= 5 else (chain_length + 4) // 5,
                    }
                ))
    
    return patterns


def _trace_power_chain(comp: Computation, start_idx: int) -> int:
    """Trace back through multiplications to find power chain length."""
    producers = comp.get_producers()
    
    op = comp.operations[start_idx]
    if op.op_type != OpType.MUL:
        return 1
    
    # Simple case: x * x = x^2
    if op.is_squaring:
        # Check if input is itself a power
        input_var = op.inputs[0]
        if input_var in producers:
            prev_op = producers[input_var]
            if prev_op.is_squaring:
                # x^2 * x^2 = x^4
                return _trace_power_chain(comp, comp.get_operation_index(input_var)) * 2
        return 2
    
    return 1


def detect_multiway_products(comp: Computation) -> List[Pattern]:
    """
    Detect multi-way product patterns (a * b * c * d).
    
    These can be computed in one Jellyfish gate vs 3 vanilla gates.
    """
    patterns = []
    producers = comp.get_producers()
    consumers = comp.get_consumers()
    
    # Look for chains of multiplications with different operands
    for i, op in enumerate(comp.operations):
        if op.op_type != OpType.MUL:
            continue
        
        # Count how many multiplications feed into this one
        mul_chain = _trace_mul_chain(comp, i, set())
        if len(mul_chain) >= 4:
            patterns.append(Pattern(
                pattern_type="multiway_product",
                operations=list(mul_chain),
                details={
                    "width": len(mul_chain),
                    "vanilla_gates": len(mul_chain) - 1,
                    "jellyfish_gates": (len(mul_chain) + 3) // 4,
                }
            ))
    
    return patterns


def _trace_mul_chain(comp: Computation, idx: int, visited: Set[int]) -> Set[int]:
    """Trace multiplication chain."""
    if idx in visited:
        return set()
    
    op = comp.operations[idx]
    if op.op_type != OpType.MUL:
        return set()
    
    visited.add(idx)
    chain = {idx}
    
    producers = comp.get_producers()
    for inp in op.inputs:
        if inp in producers:
            prev_idx = comp.get_operation_index(inp)
            if prev_idx is not None:
                chain.update(_trace_mul_chain(comp, prev_idx, visited))
    
    return chain


def detect_repeated_subexpressions(comp: Computation) -> List[Pattern]:
    """
    Detect repeated subexpressions that could be shared.
    
    If the same operation is computed multiple times, it can be
    computed once and the result reused.
    """
    patterns = []
    
    # Create signature for each operation
    signatures: Dict[str, List[int]] = {}
    
    for i, op in enumerate(comp.operations):
        sig = f"{op.op_type.value}:{sorted(op.inputs)}"
        if sig not in signatures:
            signatures[sig] = []
        signatures[sig].append(i)
    
    # Find repeated signatures
    for sig, indices in signatures.items():
        if len(indices) > 1:
            patterns.append(Pattern(
                pattern_type="repeated_subexpr",
                operations=indices,
                details={
                    "repetitions": len(indices),
                    "savings": len(indices) - 1,
                }
            ))
    
    return patterns


def analyze_computation(comp: Computation) -> Dict:
    """
    Perform comprehensive pattern analysis on a computation.
    
    Returns dict with detected patterns and optimization suggestions.
    """
    power_chains = detect_power_chains(comp)
    multiway = detect_multiway_products(comp)
    repeated = detect_repeated_subexpressions(comp)
    
    op_counts = comp.operation_counts()
    
    # Calculate potential savings
    vanilla_gates = comp.num_operations
    
    power_savings = sum(
        p.details["vanilla_gates"] - p.details["jellyfish_gates"]
        for p in power_chains
    )
    
    multiway_savings = sum(
        p.details["vanilla_gates"] - p.details["jellyfish_gates"]
        for p in multiway
    )
    
    jellyfish_gates = vanilla_gates - power_savings - multiway_savings
    
    return {
        "operation_counts": op_counts,
        "total_operations": comp.num_operations,
        "patterns": {
            "power_chains": power_chains,
            "multiway_products": multiway,
            "repeated_subexpressions": repeated,
        },
        "optimization_potential": {
            "vanilla_gates": vanilla_gates,
            "jellyfish_gates_estimate": max(1, jellyfish_gates),
            "potential_reduction": vanilla_gates / max(1, jellyfish_gates),
        },
        "recommendations": _generate_recommendations(power_chains, multiway, op_counts),
    }


def _generate_recommendations(power_chains: List[Pattern],
                               multiway: List[Pattern],
                               op_counts: Dict) -> List[str]:
    """Generate optimization recommendations."""
    recs = []
    
    if power_chains:
        total_pow = sum(p.details["exponent"] for p in power_chains)
        recs.append(f"Found {len(power_chains)} power chains (total degree {total_pow}). "
                   f"Jellyfish gates with POW5 support recommended.")
    
    if multiway:
        recs.append(f"Found {len(multiway)} multi-way products. "
                   f"Consider gates with quad-multiply support.")
    
    mul_count = op_counts.get(OpType.MUL, 0)
    add_count = op_counts.get(OpType.ADD, 0)
    
    if mul_count > add_count * 2:
        recs.append("Computation is multiplication-heavy. "
                   "High-degree gates will likely help.")
    
    if not recs:
        recs.append("No significant patterns detected. "
                   "Vanilla gates may be sufficient.")
    
    return recs


# =============================================================================
# COMPUTATION BUILDERS
# =============================================================================

def build_power_computation(exponent: int, var_name: str = "x") -> Computation:
    """
    Build a computation for x^n using square-and-multiply.
    
    Args:
        exponent: The power to compute
        var_name: Name of the input variable
        
    Returns:
        Computation for x^n
    """
    if exponent < 2:
        return Computation([], {var_name}, {var_name})
    
    ops = []
    current_power = 1
    current_var = var_name
    target = exponent
    
    # Use binary method
    powers = {}  # power -> variable name
    powers[1] = var_name
    
    # Generate powers of 2
    p = 1
    while p * 2 <= target:
        new_var = f"{var_name}{p*2}"
        ops.append(Operation(OpType.MUL, [powers[p], powers[p]], new_var))
        powers[p * 2] = new_var
        p *= 2
    
    # Combine powers to get target
    result_var = None
    remaining = target
    
    for power in sorted(powers.keys(), reverse=True):
        if power <= remaining:
            remaining -= power
            if result_var is None:
                result_var = powers[power]
            else:
                new_var = f"{var_name}_{target - remaining}"
                ops.append(Operation(OpType.MUL, [result_var, powers[power]], new_var))
                result_var = new_var
    
    return Computation(
        operations=ops,
        inputs={var_name},
        outputs={result_var},
        name=f"{var_name}^{exponent}"
    )


def build_polynomial_computation(coefficients: List[int], var_name: str = "x") -> Computation:
    """
    Build computation for polynomial evaluation.
    
    Evaluates: c[0] + c[1]*x + c[2]*x^2 + ... + c[n]*x^n
    
    Uses Horner's method for efficiency.
    """
    if not coefficients:
        return Computation([], set(), set())
    
    n = len(coefficients) - 1
    ops = []
    
    # Horner's method: c[n] + x*(c[n-1] + x*(c[n-2] + ...))
    result_var = f"c{n}"
    ops.append(Operation(OpType.CONST, [], result_var, constant=coefficients[n]))
    
    for i in range(n - 1, -1, -1):
        # result = result * x
        mul_var = f"mul{i}"
        ops.append(Operation(OpType.MUL, [result_var, var_name], mul_var))
        
        # result = result + c[i]
        if coefficients[i] != 0:
            const_var = f"c{i}"
            ops.append(Operation(OpType.CONST, [], const_var, constant=coefficients[i]))
            add_var = f"add{i}"
            ops.append(Operation(OpType.ADD, [mul_var, const_var], add_var))
            result_var = add_var
        else:
            result_var = mul_var
    
    return Computation(
        operations=ops,
        inputs={var_name},
        outputs={result_var},
        name=f"poly_deg{n}"
    )


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("COMPUTATION REPRESENTATION DEMO")
    print("=" * 60)
    
    # Build x^5 computation
    comp = build_power_computation(5)
    
    print(f"\n{comp.summary()}")
    print("\nOperations:")
    for op in comp.operations:
        print(f"  {op}")
    
    # Analyze it
    print("\n" + "-" * 60)
    print("PATTERN ANALYSIS")
    print("-" * 60)
    
    analysis = analyze_computation(comp)
    
    print(f"\nOperation counts: {analysis['operation_counts']}")
    print(f"\nDetected patterns:")
    for pattern_type, patterns in analysis['patterns'].items():
        print(f"  {pattern_type}: {len(patterns)} instances")
        for p in patterns:
            print(f"    {p.details}")
    
    print(f"\nOptimization potential:")
    opt = analysis['optimization_potential']
    print(f"  Vanilla gates: {opt['vanilla_gates']}")
    print(f"  Jellyfish gates (est): {opt['jellyfish_gates_estimate']}")
    print(f"  Potential reduction: {opt['potential_reduction']:.1f}x")
    
    print(f"\nRecommendations:")
    for rec in analysis['recommendations']:
        print(f"  - {rec}")
