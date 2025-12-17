"""
SumCheck Protocol Visualizer Core Implementation.

This module implements an interactive visualizer for the SumCheck protocol,
the core algorithm in HyperPlonk that allows proving polynomial sums
efficiently.

The SumCheck Protocol:
    GOAL: Prove that Σ f(x) = C for x ∈ {0,1}^μ
    
    Without SumCheck: Verifier checks 2^μ values (exponential!)
    With SumCheck: μ rounds of interaction (linear in μ)
    
    Each round:
        1. Prover sends a univariate polynomial s_i(X_i)
        2. Verifier checks consistency with previous round
        3. Verifier sends random challenge r_i
        4. Both parties "fix" variable X_i = r_i and continue
    
    Magic: Random challenges make it infeasible to cheat!

This visualizer shows each step in detail, helping understand:
    - How MLE tables are processed
    - How extensions are computed
    - How tables shrink after each round
    - The flow of challenges and responses
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
import random

from .mle import MLETable

if TYPE_CHECKING:
    from ..common.field import PrimeField


@dataclass
class RoundData:
    """
    Captures all data from one SumCheck round.
    
    This is useful for:
        - Visualization: Show what happened at each step
        - Debugging: Verify computations are correct
        - Analysis: Understand data movement patterns
        
    Attributes:
        round_num: Round number (1-indexed)
        challenge: The random challenge for this round
        tables_before: MLE tables at start of round
        extensions: Extensions computed for each MLE
        round_polynomial: The univariate polynomial sent to verifier
        tables_after: MLE tables after update
    """
    round_num: int
    challenge: int
    tables_before: List[Tuple[str, List[int]]]
    extensions: Dict[str, List[List[int]]]
    round_polynomial: List[int]
    tables_after: List[Tuple[str, List[int]]]
    
    # Optional metrics
    num_multiplications: int = 0
    num_additions: int = 0
    
    def __repr__(self) -> str:
        return f"RoundData(round={self.round_num}, challenge={self.challenge})"


@dataclass
class SumCheckResult:
    """
    Complete result of SumCheck protocol execution.
    
    Attributes:
        claimed_sum: The sum being proven
        challenges: All random challenges used
        round_data: Detailed data for each round
        final_value: Final evaluation f(r1, ..., rμ)
        verified: Whether the protocol verified correctly
    """
    claimed_sum: int
    challenges: List[int]
    round_data: List[RoundData]
    final_value: int
    verified: bool = True
    
    @property
    def num_rounds(self) -> int:
        return len(self.challenges)


class SumCheckVisualizer:
    """
    Interactive visualizer for the SumCheck protocol.
    
    This class runs the SumCheck protocol step by step, capturing
    detailed information at each stage for visualization and learning.
    
    The protocol proves: Σ f(X) = C for X ∈ {0,1}^μ
    
    Where f(X) is computed as the product of all input MLEs.
    (In real ZKP, this would be a more complex polynomial like ZeroCheck)
    
    Example:
        >>> field = PrimeField(97)
        >>> a = MLETable("a", [3, 7, 2, 5, 1, 8, 4, 6], field)
        >>> b = MLETable("b", [1, 4, 6, 2, 8, 3, 5, 7], field)
        >>> viz = SumCheckVisualizer([a, b], field)
        >>> result = viz.run_full_protocol(verbose=True)
    """
    
    def __init__(self, mle_tables: List[MLETable], field: 'PrimeField',
                 seed: Optional[int] = None):
        """
        Initialize the visualizer.
        
        Args:
            mle_tables: List of MLE tables (the polynomial is their product)
            field: Prime field for arithmetic
            seed: Random seed for reproducible challenges
        """
        self.original_tables = mle_tables
        self.field = field
        self.num_vars = mle_tables[0].num_vars
        
        # Validate all tables have same size
        for t in mle_tables:
            if t.num_vars != self.num_vars:
                raise ValueError(f"All MLEs must have same num_vars. "
                               f"Expected {self.num_vars}, got {t.num_vars}")
        
        # Working copies (modified during protocol)
        self.current_tables = [t.copy() for t in mle_tables]
        
        # State tracking
        self.round = 0
        self.challenges: List[int] = []
        self.history: List[RoundData] = []
        
        # Random seed
        if seed is not None:
            random.seed(seed)
    
    def reset(self):
        """Reset to initial state."""
        self.current_tables = [t.copy() for t in self.original_tables]
        self.round = 0
        self.challenges = []
        self.history = []
    
    def compute_product_at_index(self, tables: List[MLETable], index: int) -> int:
        """Compute product of all MLEs at a given boolean index."""
        product = 1
        for table in tables:
            product = self.field.mul(product, table.values[index])
        return product
    
    def compute_sum_over_hypercube(self, tables: List[MLETable]) -> int:
        """Compute Σ f(X) = Σ (product of MLEs) over boolean hypercube."""
        total = 0
        for i in range(tables[0].size):
            product = self.compute_product_at_index(tables, i)
            total = self.field.add(total, product)
        return total
    
    def compute_round_polynomial(self, tables: List[MLETable],
                                  num_extensions: int) -> Tuple[List[int], Dict[str, List[List[int]]]]:
        """
        Compute the univariate polynomial for this round.
        
        The round polynomial s(X_i) is defined as:
            s(X_i) = Σ f(r1, ..., r_{i-1}, X_i, x_{i+1}, ..., x_μ)
        summed over all boolean values of the "free" variables.
        
        We compute this by:
            1. For each pair of adjacent entries, compute extensions
            2. Multiply extensions across MLEs
            3. Accumulate into polynomial coefficients
            
        Args:
            tables: Current MLE tables
            num_extensions: Number of extension points (degree + 1)
            
        Returns:
            Tuple of (polynomial evaluations, extension data for visualization)
        """
        num_pairs = tables[0].size // 2
        
        # Accumulators for polynomial evaluations at X = 0, 1, 2, ...
        polynomial = [0] * num_extensions
        
        # Store extensions for visualization
        all_extensions: Dict[str, List[List[int]]] = {t.name: [] for t in tables}
        
        for pair_idx in range(num_pairs):
            idx0 = 2 * pair_idx
            idx1 = 2 * pair_idx + 1
            
            # Compute extensions for each MLE
            pair_extensions: List[List[int]] = []
            for table in tables:
                ext = table.compute_extension(idx0, idx1, num_extensions)
                pair_extensions.append(ext)
                all_extensions[table.name].append(ext)
            
            # Compute products at each extension point
            for k in range(num_extensions):
                product = 1
                for ext in pair_extensions:
                    product = self.field.mul(product, ext[k])
                polynomial[k] = self.field.add(polynomial[k], product)
        
        return polynomial, all_extensions
    
    def verify_round_polynomial(self, polynomial: List[int], 
                                expected_sum: int) -> bool:
        """
        Verify that s(0) + s(1) equals the expected sum.
        
        This is the key consistency check in SumCheck.
        """
        computed_sum = self.field.add(polynomial[0], polynomial[1])
        return computed_sum == expected_sum
    
    def execute_round(self, challenge: Optional[int] = None,
                      expected_sum: Optional[int] = None) -> RoundData:
        """
        Execute one round of the SumCheck protocol.
        
        Args:
            challenge: Random challenge (or None to generate)
            expected_sum: Expected sum for verification (optional)
            
        Returns:
            RoundData capturing all round information
        """
        self.round += 1
        
        if challenge is None:
            # Generate random challenge (in real protocol, from Fiat-Shamir)
            challenge = random.randint(2, self.field.prime - 1)
        
        # Capture state before round
        tables_before = [(t.name, t.values.copy()) for t in self.current_tables]
        
        # Determine number of extensions needed
        # For simple product polynomial, degree = num_mles
        num_extensions = len(self.current_tables) + 1
        
        # Compute round polynomial
        polynomial, extensions = self.compute_round_polynomial(
            self.current_tables, num_extensions
        )
        
        # Verify if expected sum provided
        if expected_sum is not None:
            if not self.verify_round_polynomial(polynomial, expected_sum):
                print(f"WARNING: Round {self.round} verification failed!")
                print(f"  s(0) + s(1) = {self.field.add(polynomial[0], polynomial[1])}")
                print(f"  Expected: {expected_sum}")
        
        # Update tables with challenge
        self.current_tables = [
            t.update_with_challenge(challenge) for t in self.current_tables
        ]
        
        # Capture state after round
        tables_after = [(t.name, t.values.copy()) for t in self.current_tables]
        
        # Create round data
        round_data = RoundData(
            round_num=self.round,
            challenge=challenge,
            tables_before=tables_before,
            extensions=extensions,
            round_polynomial=polynomial,
            tables_after=tables_after,
        )
        
        self.challenges.append(challenge)
        self.history.append(round_data)
        
        return round_data
    
    def run_full_protocol(self, challenges: Optional[List[int]] = None,
                          verbose: bool = False) -> SumCheckResult:
        """
        Run the complete SumCheck protocol.
        
        Args:
            challenges: List of challenges to use (or None for random)
            verbose: If True, print visualization at each step
            
        Returns:
            SumCheckResult with all protocol data
        """
        self.reset()
        
        # Compute initial sum
        claimed_sum = self.compute_sum_over_hypercube(self.current_tables)
        
        if verbose:
            self._print_header()
            self._print_initial_state(claimed_sum)
        
        # Generate challenges if not provided
        if challenges is None:
            challenges = [
                random.randint(2, self.field.prime - 1) 
                for _ in range(self.num_vars)
            ]
        
        # Run all rounds
        current_sum = claimed_sum
        for i, challenge in enumerate(challenges):
            round_data = self.execute_round(challenge, current_sum)
            
            if verbose:
                self._print_round(round_data)
            
            # Next round's expected sum is s(r_i) where s is the round polynomial
            current_sum = self._evaluate_polynomial_at_challenge(
                round_data.round_polynomial, challenge
            )
        
        # Final evaluation
        final_value = self.compute_product_at_index(self.current_tables, 0)
        
        # Verify final value matches
        verified = (final_value == current_sum)
        
        if verbose:
            self._print_final(final_value, current_sum, verified)
        
        return SumCheckResult(
            claimed_sum=claimed_sum,
            challenges=self.challenges,
            round_data=self.history,
            final_value=final_value,
            verified=verified,
        )
    
    def _evaluate_polynomial_at_challenge(self, polynomial: List[int], 
                                          challenge: int) -> int:
        """
        Evaluate the round polynomial at the challenge point.
        
        The polynomial is given by its evaluations at 0, 1, 2, ..., d.
        We use Lagrange interpolation to evaluate at challenge.
        """
        # For simplicity, use direct evaluation since we have evaluations at 0,1,2,...
        # For degree d polynomial with d+1 points, use Lagrange
        result = 0
        n = len(polynomial)
        
        for i, yi in enumerate(polynomial):
            # Compute Lagrange basis polynomial L_i(challenge)
            li = 1
            for j in range(n):
                if i != j:
                    # L_i(x) = Π (x - j) / (i - j)
                    num = self.field.sub(challenge, j)
                    denom = self.field.sub(i, j)
                    # Need modular inverse
                    denom_inv = pow(denom, self.field.prime - 2, self.field.prime)
                    li = self.field.mul(li, self.field.mul(num, denom_inv))
            
            result = self.field.add(result, self.field.mul(yi, li))
        
        return result
    
    # =========================================================================
    # Visualization Methods
    # =========================================================================
    
    def _print_header(self):
        """Print protocol header."""
        print("\n" + "═" * 70)
        print("              SUMCHECK PROTOCOL VISUALIZATION")
        print("═" * 70)
        
        print(f"\nPolynomial: f(X) = product of {len(self.original_tables)} MLEs")
        mle_names = " × ".join(t.name for t in self.original_tables)
        print(f"            f(X) = {mle_names}")
        print(f"Number of variables: μ = {self.num_vars}")
        print(f"Table size: 2^{self.num_vars} = {2**self.num_vars} entries")
        print(f"Field: Z_{self.field.prime}")
    
    def _print_initial_state(self, claimed_sum: int):
        """Print initial MLE tables."""
        print(f"\n{'─' * 70}")
        print("INITIAL STATE")
        print(f"{'─' * 70}")
        
        for table in self.original_tables:
            self._print_table(table)
        
        print(f"\nClaimed sum: Σ f(X) = {claimed_sum}")
        print(f"(Sum over all {2**self.num_vars} boolean inputs)")
    
    def _print_table(self, table: MLETable, max_entries: int = 16):
        """Print an MLE table."""
        if table.size <= max_entries:
            print(f"  {table.name}: {table.values}")
        else:
            print(f"  {table.name}: [{table.values[0]}, {table.values[1]}, ..., "
                  f"{table.values[-2]}, {table.values[-1]}] (size={table.size})")
    
    def _print_round(self, rd: RoundData):
        """Print detailed round information."""
        print(f"\n{'═' * 70}")
        print(f"ROUND {rd.round_num}")
        print(f"{'═' * 70}")
        
        # Tables before
        print(f"\n{'─' * 40}")
        print("Tables at start of round:")
        for name, values in rd.tables_before:
            if len(values) <= 16:
                print(f"  {name}: {values}")
            else:
                print(f"  {name}: [{values[0]}, {values[1]}, ..., "
                      f"{values[-2]}, {values[-1]}] (size={len(values)})")
        
        # Extensions (show first few pairs)
        print(f"\n{'─' * 40}")
        print("Extensions (first 2 pairs):")
        for name, exts in rd.extensions.items():
            print(f"  {name}:")
            for pair_idx, ext in enumerate(exts[:2]):
                idx0, idx1 = 2 * pair_idx, 2 * pair_idx + 1
                print(f"    Pair ({idx0},{idx1}): {ext}")
            if len(exts) > 2:
                print(f"    ... ({len(exts) - 2} more pairs)")
        
        # Round polynomial
        print(f"\n{'─' * 40}")
        print(f"Round polynomial s_{rd.round_num}(X_{rd.round_num}):")
        for k, val in enumerate(rd.round_polynomial):
            print(f"  s({k}) = {val}")
        
        check_sum = self.field.add(rd.round_polynomial[0], rd.round_polynomial[1])
        print(f"\n  Verification: s(0) + s(1) = {rd.round_polynomial[0]} + {rd.round_polynomial[1]} = {check_sum}")
        
        # Challenge
        print(f"\n{'─' * 40}")
        print(f"Random challenge: r_{rd.round_num} = {rd.challenge}")
        
        # Evaluate polynomial at challenge
        eval_at_challenge = self._evaluate_polynomial_at_challenge(
            rd.round_polynomial, rd.challenge
        )
        print(f"s_{rd.round_num}({rd.challenge}) = {eval_at_challenge}")
        print(f"(This becomes the claimed sum for next round)")
        
        # Tables after
        print(f"\n{'─' * 40}")
        print(f"Tables after update (fixing X_{rd.round_num} = {rd.challenge}):")
        for name, values in rd.tables_after:
            if len(values) <= 16:
                print(f"  {name}: {values}")
            else:
                print(f"  {name}: [{values[0]}, {values[1]}, ..., "
                      f"{values[-2]}, {values[-1]}] (size={len(values)})")
        
        print(f"\n  Table size: {len(rd.tables_before[0][1])} → {len(rd.tables_after[0][1])}")
    
    def _print_final(self, final_value: int, expected: int, verified: bool):
        """Print final verification."""
        print(f"\n{'═' * 70}")
        print("PROTOCOL COMPLETE")
        print(f"{'═' * 70}")
        
        print(f"\nAll challenges: {self.challenges}")
        print(f"Final evaluation point: ({', '.join(map(str, self.challenges))})")
        print(f"\nFinal value f(r₁, ..., r_μ) = {final_value}")
        print(f"Expected (from last round): {expected}")
        
        if verified:
            print(f"\n✓ VERIFICATION PASSED")
        else:
            print(f"\n✗ VERIFICATION FAILED")


# Example usage
if __name__ == "__main__":
    from ..common.field import PrimeField
    
    # Create small example
    field = PrimeField(97)
    
    a = MLETable("a", [3, 7, 2, 5, 1, 8, 4, 6], field)
    b = MLETable("b", [1, 4, 6, 2, 8, 3, 5, 7], field)
    c = MLETable("c", [5, 2, 8, 1, 4, 7, 3, 6], field)
    
    viz = SumCheckVisualizer([a, b, c], field, seed=42)
    result = viz.run_full_protocol(verbose=True)
