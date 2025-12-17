"""
Predefined Polynomials for SumCheck Simulation.

This module provides polynomial structures used in HyperPlonk and similar
ZKP protocols. Each polynomial captures:
    - Number of terms
    - Degree of each term
    - Number of unique MLEs
    - MLE reuse patterns

These structures determine:
    - Number of extensions needed per round
    - Number of multiplications per term
    - Data movement patterns
    
The key insight from zkPHIRE is that higher-degree custom gates (Jellyfish)
can dramatically reduce gate count, even though each gate is more complex.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Set, Dict


@dataclass
class SimPolynomial:
    """
    Polynomial structure for simulation.
    
    This captures the structure of a SumCheck polynomial without
    actual values - just what's needed for performance modeling.
    
    Attributes:
        name: Polynomial identifier
        terms: List of terms, each term is a list of MLE names
        description: Human-readable description
        
    Example:
        >>> poly = SimPolynomial(
        ...     name="simple_product",
        ...     terms=[["a", "b", "c"]],
        ...     description="Product of three MLEs"
        ... )
        >>> print(poly.max_degree)
        3
    """
    name: str
    terms: List[List[str]]
    description: str = ""
    
    @property
    def num_terms(self) -> int:
        """Number of terms in the polynomial."""
        return len(self.terms)
    
    @property
    def max_degree(self) -> int:
        """Maximum degree (largest term size)."""
        if not self.terms:
            return 0
        return max(len(term) for term in self.terms)
    
    @property
    def num_extensions(self) -> int:
        """Number of extension points needed per round."""
        return self.max_degree + 1
    
    @property
    def unique_mles(self) -> Set[str]:
        """Set of unique MLE names."""
        mles = set()
        for term in self.terms:
            mles.update(term)
        return mles
    
    @property
    def num_mles(self) -> int:
        """Count of unique MLEs."""
        return len(self.unique_mles)
    
    def mle_occurrences(self) -> Dict[str, int]:
        """Count occurrences of each MLE."""
        counts: Dict[str, int] = {}
        for term in self.terms:
            for mle in term:
                counts[mle] = counts.get(mle, 0) + 1
        return counts
    
    def multiplications_per_evaluation(self) -> int:
        """
        Count multiplications needed per evaluation.
        
        Each term with k MLEs needs (k-1) multiplications.
        Plus (num_terms - 1) additions, but we focus on muls.
        """
        return sum(max(0, len(term) - 1) for term in self.terms)
    
    def compute_characteristics(self) -> Dict:
        """
        Compute detailed characteristics for analysis.
        
        Returns dict with various metrics useful for simulation.
        """
        mle_counts = self.mle_occurrences()
        reused = {m: c for m, c in mle_counts.items() if c > 1}
        
        term_degrees = [len(t) for t in self.terms]
        
        return {
            "name": self.name,
            "num_terms": self.num_terms,
            "max_degree": self.max_degree,
            "avg_degree": sum(term_degrees) / len(term_degrees) if term_degrees else 0,
            "num_mles": self.num_mles,
            "extensions_needed": self.num_extensions,
            "muls_per_eval": self.multiplications_per_evaluation(),
            "reused_mles": len(reused),
            "reuse_ratio": len(reused) / self.num_mles if self.num_mles else 0,
        }
    
    def summary(self) -> str:
        """Return summary string."""
        chars = self.compute_characteristics()
        return (
            f"SimPolynomial '{self.name}':\n"
            f"  {self.description}\n"
            f"  Terms: {chars['num_terms']}\n"
            f"  Max degree: {chars['max_degree']}\n"
            f"  Avg degree: {chars['avg_degree']:.1f}\n"
            f"  Unique MLEs: {chars['num_mles']}\n"
            f"  Extensions/round: {chars['extensions_needed']}\n"
            f"  Multiplications/eval: {chars['muls_per_eval']}\n"
            f"  MLE reuse ratio: {chars['reuse_ratio']:.1%}"
        )
    
    def __repr__(self) -> str:
        return f"SimPolynomial('{self.name}', terms={self.num_terms}, degree={self.max_degree})"


# =============================================================================
# VANILLA PLONK POLYNOMIALS (from zkSpeed)
# =============================================================================

VANILLA_ZEROCHECK = SimPolynomial(
    name="VanillaZeroCheck",
    terms=[
        ["qL", "w1", "fr"],           # Left input: qL × w1 × fr
        ["qR", "w2", "fr"],           # Right input: qR × w2 × fr
        ["qM", "w1", "w2", "fr"],     # Multiplication: qM × w1 × w2 × fr
        ["qO", "w3", "fr"],           # Output: qO × w3 × fr
        ["qC", "fr"],                 # Constant: qC × fr
    ],
    description=(
        "Vanilla Plonk ZeroCheck polynomial.\n"
        "f = qL·w1·fr + qR·w2·fr + qM·w1·w2·fr - qO·w3·fr + qC·fr\n"
        "This is the polynomial that must equal 0 for correct gates."
    )
)

VANILLA_PERMCHECK = SimPolynomial(
    name="VanillaPermCheck",
    terms=[
        ["pi", "fr"],                      # π × fr
        ["p1", "p2", "fr"],                # p1 × p2 × fr  
        ["phi", "D1", "D2", "D3", "fr"],   # φ × D1 × D2 × D3 × fr
        ["N1", "N2", "N3", "fr"],          # N1 × N2 × N3 × fr
    ],
    description=(
        "Vanilla Plonk PermCheck polynomial.\n"
        "Checks that wires are correctly connected (permutation argument).\n"
        "Involves fraction polynomials requiring modular inversions."
    )
)

VANILLA_OPENCHECK = SimPolynomial(
    name="VanillaOpenCheck",
    terms=[
        ["y1", "k1"],
        ["y2", "k2"],
        ["y3", "k3"],
        ["y4", "k4"],
        ["y5", "k5"],
        ["y6", "k6"],
    ],
    description=(
        "Vanilla Plonk OpenCheck polynomial.\n"
        "Used for batched polynomial opening proof.\n"
        "Lower degree than ZeroCheck/PermCheck."
    )
)


# =============================================================================
# JELLYFISH HIGH-DEGREE POLYNOMIALS (from zkPHIRE)
# =============================================================================

JELLYFISH_ZEROCHECK = SimPolynomial(
    name="JellyfishZeroCheck",
    terms=[
        # Linear terms
        ["q1", "w1", "fr"],
        ["q2", "w2", "fr"],
        ["q3", "w3", "fr"],
        ["q4", "w4", "fr"],
        # Pairwise products
        ["qM1", "w1", "w2", "fr"],
        ["qM2", "w3", "w4", "fr"],
        # Fifth power terms (x^5 for Poseidon hash)
        ["qH1", "w1", "w1", "w1", "w1", "w1", "fr"],  # w1^5
        ["qH2", "w2", "w2", "w2", "w2", "w2", "fr"],  # w2^5
        ["qH3", "w3", "w3", "w3", "w3", "w3", "fr"],  # w3^5
        ["qH4", "w4", "w4", "w4", "w4", "w4", "fr"],  # w4^5
        # Four-way product (for EC operations)
        ["qecc", "w1", "w2", "w3", "w4", "fr"],
        # Output
        ["qO", "w5", "fr"],
        # Constant
        ["qC", "fr"],
    ],
    description=(
        "Jellyfish high-degree ZeroCheck polynomial.\n"
        "Supports 5th powers (for Poseidon) and 4-way products (for EC).\n"
        "Higher degree per gate, but can express more computation per gate,\n"
        "reducing total gate count by 8-32x."
    )
)

# Simplified Jellyfish (without all the high-degree terms)
JELLYFISH_SIMPLE = SimPolynomial(
    name="JellyfishSimple",
    terms=[
        ["q1", "w1", "fr"],
        ["q2", "w2", "fr"],
        ["qM", "w1", "w2", "fr"],
        ["qH", "w1", "w1", "w1", "w1", "w1", "fr"],  # w1^5
        ["qO", "w3", "fr"],
        ["qC", "fr"],
    ],
    description=(
        "Simplified Jellyfish with one high-degree term.\n"
        "Good for understanding the impact of x^5 gates."
    )
)


# =============================================================================
# ADDITIONAL POLYNOMIAL TYPES
# =============================================================================

def create_product_polynomial(num_mles: int, name: str = None) -> SimPolynomial:
    """
    Create a simple product polynomial.
    
    f = mle0 × mle1 × ... × mle_{n-1}
    
    Args:
        num_mles: Number of MLEs in the product
        name: Optional name
    """
    mle_names = [f"mle{i}" for i in range(num_mles)]
    return SimPolynomial(
        name=name or f"Product{num_mles}",
        terms=[mle_names],
        description=f"Product of {num_mles} MLEs"
    )


def create_sum_of_products(num_terms: int, mles_per_term: int,
                           name: str = None) -> SimPolynomial:
    """
    Create a sum of product polynomial.
    
    f = (a0 × ... × a_{k-1}) + (b0 × ... × b_{k-1}) + ...
    
    Args:
        num_terms: Number of terms in the sum
        mles_per_term: MLEs per term (all same)
        name: Optional name
    """
    terms = []
    for t in range(num_terms):
        term = [f"t{t}_mle{i}" for i in range(mles_per_term)]
        terms.append(term)
    
    return SimPolynomial(
        name=name or f"SumOfProducts_{num_terms}x{mles_per_term}",
        terms=terms,
        description=f"Sum of {num_terms} terms, each with {mles_per_term} MLEs"
    )


def create_polynomial_with_degree(degree: int, num_terms: int = 1,
                                   name: str = None) -> SimPolynomial:
    """
    Create a polynomial with specified degree.
    
    Args:
        degree: Maximum term degree
        num_terms: Number of terms
        name: Optional name
    """
    terms = []
    for t in range(num_terms):
        term = [f"t{t}_x{i}" for i in range(degree)]
        terms.append(term)
    
    return SimPolynomial(
        name=name or f"Degree{degree}",
        terms=terms,
        description=f"Polynomial with degree {degree}, {num_terms} terms"
    )


# =============================================================================
# POLYNOMIAL COMPARISON UTILITIES
# =============================================================================

def compare_polynomials(poly1: SimPolynomial, poly2: SimPolynomial) -> Dict:
    """
    Compare two polynomials for performance characteristics.
    
    Returns comparison metrics useful for understanding the
    tradeoffs between vanilla and custom gates.
    """
    c1 = poly1.compute_characteristics()
    c2 = poly2.compute_characteristics()
    
    return {
        "polynomial1": c1,
        "polynomial2": c2,
        "comparison": {
            "term_ratio": c2["num_terms"] / c1["num_terms"] if c1["num_terms"] else float('inf'),
            "degree_ratio": c2["max_degree"] / c1["max_degree"] if c1["max_degree"] else float('inf'),
            "mle_ratio": c2["num_mles"] / c1["num_mles"] if c1["num_mles"] else float('inf'),
            "mul_ratio": c2["muls_per_eval"] / c1["muls_per_eval"] if c1["muls_per_eval"] else float('inf'),
        }
    }


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("SUMCHECK POLYNOMIAL STRUCTURES")
    print("=" * 70)
    
    polynomials = [
        VANILLA_ZEROCHECK,
        VANILLA_PERMCHECK,
        VANILLA_OPENCHECK,
        JELLYFISH_ZEROCHECK,
        JELLYFISH_SIMPLE,
    ]
    
    for poly in polynomials:
        print(f"\n{poly.summary()}")
        print("-" * 70)
    
    # Compare vanilla vs jellyfish
    print("\n" + "=" * 70)
    print("VANILLA vs JELLYFISH COMPARISON")
    print("=" * 70)
    
    comparison = compare_polynomials(VANILLA_ZEROCHECK, JELLYFISH_ZEROCHECK)
    
    print(f"\nVanilla ZeroCheck:")
    for k, v in comparison["polynomial1"].items():
        print(f"  {k}: {v}")
    
    print(f"\nJellyfish ZeroCheck:")
    for k, v in comparison["polynomial2"].items():
        print(f"  {k}: {v}")
    
    print(f"\nRatios (Jellyfish / Vanilla):")
    for k, v in comparison["comparison"].items():
        print(f"  {k}: {v:.2f}x")
