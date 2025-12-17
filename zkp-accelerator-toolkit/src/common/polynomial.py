"""
Polynomial Representations for SumCheck Protocol.

This module provides classes for representing polynomials as used in
the SumCheck protocol for ZKP systems like HyperPlonk.

Key Concepts:
    - Multilinear Polynomial: Polynomial linear in each variable (degree 1 per var)
    - MLE (Multilinear Extension): Unique multilinear polynomial passing through
      given evaluations at boolean hypercube points
    - SumCheck Polynomial: Sum of products of MLEs (the polynomial we prove sums to 0)

Example of Multilinear Polynomial:
    f(X1, X2) = a00(1-X1)(1-X2) + a01(1-X1)X2 + a10*X1(1-X2) + a11*X1*X2
    
    This is determined by its values at {0,1}²:
        f(0,0) = a00, f(0,1) = a01, f(1,0) = a10, f(1,1) = a11

SumCheck Polynomial Structure (ZeroCheck example):
    f = qL·w1 + qR·w2 + qM·w1·w2 - qO·w3 + qC
    
    Each term is a product of MLEs (qL, qR, qM, qO, qC, w1, w2, w3).
    The polynomial f must equal 0 at all boolean inputs if the circuit is correct.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from .field import PrimeField, FieldElement


@dataclass
class Term:
    """
    A single term in a SumCheck polynomial.
    
    A term is a product of MLEs, possibly with a coefficient.
    For example, in "qM · w1 · w2", the term involves three MLEs.
    
    Attributes:
        mle_names: List of MLE names in this product
        coefficient: Scalar multiplier (default 1, use -1 for subtraction)
        
    Example:
        >>> term = Term(["qM", "w1", "w2"], coefficient=1)
        >>> print(term)  # "qM * w1 * w2"
    """
    mle_names: List[str]
    coefficient: int = 1
    
    @property
    def degree(self) -> int:
        """Number of MLEs multiplied together."""
        return len(self.mle_names)
    
    @property
    def unique_mles(self) -> Set[str]:
        """Set of unique MLE names (for counting data dependencies)."""
        return set(self.mle_names)
    
    def __repr__(self) -> str:
        coef_str = f"{self.coefficient} * " if self.coefficient != 1 else ""
        if self.coefficient == -1:
            coef_str = "-"
        return f"{coef_str}{' * '.join(self.mle_names)}"


@dataclass  
class PolynomialTerm:
    """
    Enhanced term representation with MLE index mapping.
    
    This is used when we have actual MLE tables and need to
    evaluate the term at specific indices.
    
    Attributes:
        mle_indices: Indices into the MLE table array
        coefficient: Scalar multiplier
    """
    mle_indices: List[int]
    coefficient: int = 1
    
    @property
    def degree(self) -> int:
        return len(self.mle_indices)


@dataclass
class Polynomial:
    """
    A SumCheck polynomial represented as a sum of terms.
    
    In HyperPlonk, we need to prove that a polynomial evaluates to 0
    at all points in the boolean hypercube. The polynomial is structured
    as a sum of products of MLEs.
    
    Example - Vanilla Plonk ZeroCheck:
        f = qL·w1 + qR·w2 + qM·w1·w2 - qO·w3 + qC
        
    This would be represented as:
        Polynomial([
            Term(["qL", "w1"]),
            Term(["qR", "w2"]),
            Term(["qM", "w1", "w2"]),
            Term(["qO", "w3"], coefficient=-1),
            Term(["qC"]),
        ])
    
    Attributes:
        terms: List of Term objects
        name: Optional name for the polynomial
    """
    terms: List[Term]
    name: str = "f"
    
    @property
    def num_terms(self) -> int:
        """Number of terms in the polynomial."""
        return len(self.terms)
    
    @property
    def max_degree(self) -> int:
        """Maximum degree (largest product of MLEs in any term)."""
        if not self.terms:
            return 0
        return max(term.degree for term in self.terms)
    
    @property
    def total_degree(self) -> int:
        """
        Total degree across all variables.
        
        For SumCheck, this determines how many extension points we need.
        We need (degree + 1) evaluations to characterize each round polynomial.
        """
        return self.max_degree
    
    @property
    def num_extensions(self) -> int:
        """Number of extension points needed per SumCheck round."""
        return self.max_degree + 1
    
    @property
    def unique_mles(self) -> Set[str]:
        """Set of all unique MLE names across all terms."""
        mles = set()
        for term in self.terms:
            mles.update(term.unique_mles)
        return mles
    
    @property
    def num_mles(self) -> int:
        """Number of unique MLEs referenced."""
        return len(self.unique_mles)
    
    def mle_occurrences(self) -> Dict[str, int]:
        """
        Count occurrences of each MLE across all terms.
        
        This helps analyze data reuse potential. MLEs that appear
        in multiple terms can have their extensions computed once
        and reused.
        """
        counts: Dict[str, int] = {}
        for term in self.terms:
            for mle in term.mle_names:
                counts[mle] = counts.get(mle, 0) + 1
        return counts
    
    def num_multiplications(self) -> int:
        """
        Count multiplications needed to evaluate the polynomial.
        
        Each term with k MLEs needs (k-1) multiplications.
        """
        return sum(max(0, term.degree - 1) for term in self.terms)
    
    def get_term_by_mles(self, mle_names: Set[str]) -> Optional[Term]:
        """Find a term containing exactly the given MLEs."""
        for term in self.terms:
            if term.unique_mles == mle_names:
                return term
        return None
    
    def __repr__(self) -> str:
        if not self.terms:
            return f"{self.name} = 0"
        
        term_strs = []
        for i, term in enumerate(self.terms):
            term_str = str(term)
            if i > 0 and term.coefficient > 0:
                term_str = "+ " + term_str
            term_strs.append(term_str)
        
        return f"{self.name} = {' '.join(term_strs)}"
    
    def summary(self) -> str:
        """Return a summary of polynomial characteristics."""
        return (
            f"Polynomial '{self.name}':\n"
            f"  Terms: {self.num_terms}\n"
            f"  Max degree: {self.max_degree}\n"
            f"  Extensions needed: {self.num_extensions}\n"
            f"  Unique MLEs: {self.num_mles}\n"
            f"  MLEs: {sorted(self.unique_mles)}\n"
            f"  Multiplications/eval: {self.num_multiplications()}"
        )


# =============================================================================
# PREDEFINED POLYNOMIALS
# =============================================================================
# These are the actual polynomial structures used in HyperPlonk.
# Understanding these is key to understanding the zkSpeed/zkPHIRE papers.

def create_vanilla_zerocheck() -> Polynomial:
    """
    Create the vanilla Plonk ZeroCheck polynomial.
    
    f = qL·w1·fr + qR·w2·fr + qM·w1·w2·fr - qO·w3·fr + qC·fr
    
    Where:
        - qL, qR, qM, qO, qC: Selector polynomials (define gate types)
        - w1, w2, w3: Witness polynomials (actual values)
        - fr: Random folding polynomial (for batching)
        
    This is the polynomial that must sum to 0 over all boolean inputs
    if all gates in the circuit are satisfied.
    """
    return Polynomial(
        terms=[
            Term(["qL", "w1", "fr"]),           # Left input term
            Term(["qR", "w2", "fr"]),           # Right input term  
            Term(["qM", "w1", "w2", "fr"]),     # Multiplication term
            Term(["qO", "w3", "fr"], coefficient=-1),  # Output term (subtracted)
            Term(["qC", "fr"]),                 # Constant term
        ],
        name="ZeroCheck"
    )


def create_vanilla_permcheck() -> Polynomial:
    """
    Create the vanilla Plonk PermCheck polynomial.
    
    f = π·fr - p1·p2·fr + α(φ·D1·D2·D3)·fr - α(N1·N2·N3)·fr
    
    This checks that wires are correctly connected (permutation argument).
    The polynomial structure involves fraction polynomials (N/D) which
    require modular inversions - a key bottleneck in zkSpeed.
    """
    return Polynomial(
        terms=[
            Term(["pi", "fr"]),                      # Permutation identity
            Term(["p1", "p2", "fr"], coefficient=-1),  # Product check
            Term(["phi", "D1", "D2", "D3", "fr"]),   # Denominator term
            Term(["N1", "N2", "N3", "fr"], coefficient=-1),  # Numerator term
        ],
        name="PermCheck"
    )


def create_vanilla_opencheck() -> Polynomial:
    """
    Create the vanilla Plonk OpenCheck polynomial.
    
    f = y1·k1 + y2·k2 + y3·k3 + y4·k4 + y5·k5 + y6·k6
    
    This is used for batched polynomial opening (proving claimed evaluations).
    Each term is a product of a random challenge (yi) and a kernel (ki).
    """
    return Polynomial(
        terms=[
            Term(["y1", "k1"]),
            Term(["y2", "k2"]),
            Term(["y3", "k3"]),
            Term(["y4", "k4"]),
            Term(["y5", "k5"]),
            Term(["y6", "k6"]),
        ],
        name="OpenCheck"
    )


def create_jellyfish_zerocheck() -> Polynomial:
    """
    Create the Jellyfish high-degree ZeroCheck polynomial.
    
    Jellyfish gates support higher-degree operations, allowing operations
    like x^5 to be computed in a single gate (vs. 3 gates for vanilla).
    
    f = q1·w1 + q2·w2 + q3·w3 + q4·w4 
      + qM1·w1·w2 + qM2·w3·w4
      + qH1·w1^5 + qH2·w2^5 + qH3·w3^5 + qH4·w4^5
      + qecc·w1·w2·w3·w4
      - qO·w5 + qC
      
    Note: w1^5 is represented as w1·w1·w1·w1·w1 (5 copies of w1)
    """
    return Polynomial(
        terms=[
            # Linear terms
            Term(["q1", "w1", "fr"]),
            Term(["q2", "w2", "fr"]),
            Term(["q3", "w3", "fr"]),
            Term(["q4", "w4", "fr"]),
            # Pairwise multiplication terms
            Term(["qM1", "w1", "w2", "fr"]),
            Term(["qM2", "w3", "w4", "fr"]),
            # High-degree terms (x^5)
            Term(["qH1", "w1", "w1", "w1", "w1", "w1", "fr"]),
            Term(["qH2", "w2", "w2", "w2", "w2", "w2", "fr"]),
            Term(["qH3", "w3", "w3", "w3", "w3", "w3", "fr"]),
            Term(["qH4", "w4", "w4", "w4", "w4", "w4", "fr"]),
            # Four-way product (for EC operations)
            Term(["qecc", "w1", "w2", "w3", "w4", "fr"]),
            # Output term
            Term(["qO", "w5", "fr"], coefficient=-1),
            # Constant
            Term(["qC", "fr"]),
        ],
        name="JellyfishZeroCheck"
    )


def create_simple_product(num_mles: int = 3) -> Polynomial:
    """
    Create a simple product polynomial for testing.
    
    f = MLE_0 · MLE_1 · ... · MLE_{n-1}
    
    Args:
        num_mles: Number of MLEs to multiply together
    """
    mle_names = [f"mle{i}" for i in range(num_mles)]
    return Polynomial(
        terms=[Term(mle_names)],
        name=f"Product{num_mles}"
    )


def create_custom_polynomial(specification: str) -> Polynomial:
    """
    Parse a polynomial specification string into a Polynomial object.
    
    Format: "term1 + term2 - term3 + ..."
    Each term: "mle1*mle2*mle3" or "coef*mle1*mle2"
    
    Example:
        >>> poly = create_custom_polynomial("a*b + c*d - e")
        >>> print(poly)
        f = a * b + c * d - e
    
    Args:
        specification: String specification of the polynomial
        
    Returns:
        Polynomial object
    """
    terms = []
    
    # Split by + and -, keeping the sign
    import re
    parts = re.split(r'\s*([+-])\s*', specification.strip())
    
    # First part has implicit + sign
    if parts[0]:
        parts = ['+'] + parts if parts[0] not in ['+', '-'] else parts
    
    i = 0
    while i < len(parts):
        sign = parts[i] if parts[i] in ['+', '-'] else '+'
        term_str = parts[i] if parts[i] not in ['+', '-'] else parts[i+1]
        
        if parts[i] in ['+', '-']:
            i += 1
        
        if i < len(parts):
            term_str = parts[i]
            i += 1
        else:
            break
            
        # Parse the term (mle1*mle2*... or coef*mle1*...)
        factors = [f.strip() for f in term_str.split('*')]
        
        coefficient = 1 if sign == '+' else -1
        mle_names = []
        
        for factor in factors:
            if factor.isdigit():
                coefficient *= int(factor)
            elif factor.lstrip('-').isdigit():
                coefficient *= int(factor)
            else:
                mle_names.append(factor)
        
        if mle_names:
            terms.append(Term(mle_names, coefficient))
    
    return Polynomial(terms)


# =============================================================================
# Polynomial Analysis Utilities
# =============================================================================

def analyze_polynomial(poly: Polynomial) -> Dict:
    """
    Analyze a polynomial's structure for optimization insights.
    
    Returns a dictionary with:
        - basic_stats: term count, degree, MLE count
        - mle_reuse: how often each MLE appears (for caching)
        - compute_profile: multiplication counts
        - hardware_hints: suggestions for accelerator design
    """
    mle_counts = poly.mle_occurrences()
    
    # Find MLEs that appear in multiple terms (good for caching)
    reused_mles = {m: c for m, c in mle_counts.items() if c > 1}
    
    # Compute profile
    muls_per_term = [term.degree - 1 for term in poly.terms if term.degree > 0]
    total_muls = sum(max(0, m) for m in muls_per_term)
    
    # Hardware hints
    hints = []
    if poly.max_degree > 4:
        hints.append("High-degree polynomial: benefits from programmable SumCheck (zkPHIRE)")
    if len(reused_mles) > len(mle_counts) * 0.3:
        hints.append("Significant MLE reuse: cache extensions across terms")
    if poly.num_terms > 8:
        hints.append("Many terms: consider term-by-term processing")
    
    return {
        "basic_stats": {
            "num_terms": poly.num_terms,
            "max_degree": poly.max_degree,
            "num_mles": poly.num_mles,
            "extensions_needed": poly.num_extensions,
        },
        "mle_reuse": {
            "occurrence_counts": mle_counts,
            "reused_mles": reused_mles,
            "reuse_ratio": len(reused_mles) / len(mle_counts) if mle_counts else 0,
        },
        "compute_profile": {
            "muls_per_term": muls_per_term,
            "total_multiplications": total_muls,
            "avg_term_degree": sum(t.degree for t in poly.terms) / len(poly.terms) if poly.terms else 0,
        },
        "hardware_hints": hints,
    }


def compare_polynomials(poly1: Polynomial, poly2: Polynomial) -> Dict:
    """
    Compare two polynomials for performance characteristics.
    
    Useful for comparing vanilla vs. Jellyfish gate configurations.
    """
    analysis1 = analyze_polynomial(poly1)
    analysis2 = analyze_polynomial(poly2)
    
    return {
        "polynomial1": {
            "name": poly1.name,
            "analysis": analysis1,
        },
        "polynomial2": {
            "name": poly2.name,
            "analysis": analysis2,
        },
        "comparison": {
            "term_ratio": analysis2["basic_stats"]["num_terms"] / analysis1["basic_stats"]["num_terms"] if analysis1["basic_stats"]["num_terms"] else float('inf'),
            "degree_ratio": analysis2["basic_stats"]["max_degree"] / analysis1["basic_stats"]["max_degree"] if analysis1["basic_stats"]["max_degree"] else float('inf'),
            "mle_ratio": analysis2["basic_stats"]["num_mles"] / analysis1["basic_stats"]["num_mles"] if analysis1["basic_stats"]["num_mles"] else float('inf'),
        }
    }


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("POLYNOMIAL REPRESENTATION DEMO")
    print("=" * 60)
    
    # Create and display vanilla ZeroCheck
    vanilla = create_vanilla_zerocheck()
    print(f"\n{vanilla}")
    print(f"\n{vanilla.summary()}")
    
    # Analyze it
    print("\n" + "-" * 60)
    print("ANALYSIS")
    print("-" * 60)
    analysis = analyze_polynomial(vanilla)
    print(f"\nMLE occurrences: {analysis['mle_reuse']['occurrence_counts']}")
    print(f"Reused MLEs: {analysis['mle_reuse']['reused_mles']}")
    print(f"Hardware hints:")
    for hint in analysis['hardware_hints']:
        print(f"  - {hint}")
    
    # Compare vanilla vs Jellyfish
    print("\n" + "-" * 60)
    print("VANILLA vs JELLYFISH COMPARISON")
    print("-" * 60)
    jellyfish = create_jellyfish_zerocheck()
    comparison = compare_polynomials(vanilla, jellyfish)
    
    print(f"\nVanilla: {comparison['polynomial1']['analysis']['basic_stats']}")
    print(f"Jellyfish: {comparison['polynomial2']['analysis']['basic_stats']}")
    print(f"Ratios (Jellyfish/Vanilla):")
    print(f"  Terms: {comparison['comparison']['term_ratio']:.2f}x")
    print(f"  Degree: {comparison['comparison']['degree_ratio']:.2f}x")
    print(f"  MLEs: {comparison['comparison']['mle_ratio']:.2f}x")
