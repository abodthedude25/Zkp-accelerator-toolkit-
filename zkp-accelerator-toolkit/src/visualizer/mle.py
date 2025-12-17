"""
Multilinear Extension (MLE) Table Implementation.

An MLE table stores a multilinear polynomial by its evaluations at all
points in the boolean hypercube {0,1}^μ. This is the fundamental data
structure processed by SumCheck.

Key Operations:
    - Extension: Evaluate at non-boolean points (X = 0, 1, 2, ...)
    - Update: Fix one variable to a challenge, halving table size
    - Evaluate: Get value at a specific boolean index

Memory Considerations (from zkSpeed/zkPHIRE):
    - For μ = 20 variables: 2^20 ≈ 1M entries
    - Each entry: 32 bytes (255-bit field element)
    - One MLE table: ~32 MB
    - Multiple MLEs (9+ for ZeroCheck): ~300 MB total
    - This is why memory bandwidth is critical!
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, TYPE_CHECKING
import math

if TYPE_CHECKING:
    from ..common.field import PrimeField, FieldElement


@dataclass
class MLETable:
    """
    A multilinear extension stored as a lookup table.
    
    The table stores f(x) for all x ∈ {0,1}^μ, where μ = log2(size).
    The indexing convention is:
        - index 0 = (0, 0, ..., 0)
        - index 1 = (0, 0, ..., 1)  
        - index 2 = (0, 0, ..., 1, 0)
        - etc. (binary representation, LSB = rightmost variable)
    
    Attributes:
        name: Identifier for this MLE (e.g., "qL", "w1")
        values: List of field element values
        field: The prime field for arithmetic
        
    Example:
        >>> field = PrimeField(97)
        >>> # 2 variables, 4 entries: f(0,0)=3, f(0,1)=7, f(1,0)=2, f(1,1)=5
        >>> mle = MLETable("f", [3, 7, 2, 5], field)
        >>> print(mle.num_vars)
        2
    """
    name: str
    values: List[int]
    field: 'PrimeField'
    
    def __post_init__(self):
        """Validate and reduce values to field."""
        # Ensure size is power of 2
        if self.size > 0 and (self.size & (self.size - 1)) != 0:
            raise ValueError(f"Table size must be power of 2, got {self.size}")
        
        # Reduce values to field
        self.values = [v % self.field.prime for v in self.values]
    
    @property
    def size(self) -> int:
        """Number of entries in the table."""
        return len(self.values)
    
    @property
    def num_vars(self) -> int:
        """Number of variables (μ = log2(size))."""
        if self.size == 0:
            return 0
        return int(math.log2(self.size))
    
    def copy(self) -> 'MLETable':
        """Create a deep copy of this table."""
        return MLETable(self.name, self.values.copy(), self.field)
    
    def __getitem__(self, index: int) -> int:
        """Get value at a boolean index."""
        return self.values[index]
    
    def __setitem__(self, index: int, value: int):
        """Set value at a boolean index."""
        self.values[index] = value % self.field.prime
    
    def __repr__(self) -> str:
        if self.size <= 8:
            return f"MLETable({self.name}, {self.values})"
        return f"MLETable({self.name}, size={self.size}, first_few={self.values[:4]}...)"
    
    def evaluate_at_index(self, index: int) -> int:
        """
        Get the value at a boolean index.
        
        Args:
            index: Integer in [0, 2^μ - 1]
            
        Returns:
            Field element value at this index
        """
        return self.values[index]
    
    def compute_extension(self, idx0: int, idx1: int, 
                          num_points: int) -> List[int]:
        """
        Compute linear extension between two adjacent entries.
        
        Given values v0 = table[idx0] and v1 = table[idx1], compute:
            extension(k) = v0 + k * (v1 - v0)
        for k = 0, 1, 2, ..., num_points-1
        
        This is the key operation in each SumCheck round:
        - At k=0: returns v0 (value when variable = 0)
        - At k=1: returns v1 (value when variable = 1)  
        - At k>1: extrapolation for higher-degree polynomials
        
        Args:
            idx0: Index of first value (variable = 0)
            idx1: Index of second value (variable = 1)
            num_points: Number of extension points to compute
            
        Returns:
            List of field elements [ext(0), ext(1), ..., ext(num_points-1)]
        """
        v0 = self.values[idx0]
        v1 = self.values[idx1]
        diff = self.field.sub(v1, v0)
        
        extensions = []
        for k in range(num_points):
            # ext[k] = v0 + k * (v1 - v0)
            ext_val = self.field.add(v0, self.field.mul(k, diff))
            extensions.append(ext_val)
        
        return extensions
    
    def compute_all_extensions(self, num_points: int) -> List[List[int]]:
        """
        Compute extensions for all adjacent pairs.
        
        For a table of size n, there are n/2 pairs:
            (table[0], table[1]), (table[2], table[3]), ...
            
        Args:
            num_points: Extension points per pair
            
        Returns:
            List of extension lists, one per pair
        """
        num_pairs = self.size // 2
        all_extensions = []
        
        for pair_idx in range(num_pairs):
            idx0 = 2 * pair_idx
            idx1 = 2 * pair_idx + 1
            ext = self.compute_extension(idx0, idx1, num_points)
            all_extensions.append(ext)
        
        return all_extensions
    
    def update_with_challenge(self, challenge: int) -> 'MLETable':
        """
        Create a new table by fixing the first variable to challenge.
        
        This is the MLE update step after each SumCheck round:
            new_table[i] = old_table[2i] + challenge * (old_table[2i+1] - old_table[2i])
            
        The result has half the entries (one fewer variable).
        
        Args:
            challenge: The random challenge from the verifier
            
        Returns:
            New MLETable with size/2 entries
        """
        new_size = self.size // 2
        new_values = []
        
        for i in range(new_size):
            v0 = self.values[2 * i]
            v1 = self.values[2 * i + 1]
            
            # new = v0 + challenge * (v1 - v0)
            diff = self.field.sub(v1, v0)
            new_val = self.field.add(v0, self.field.mul(challenge, diff))
            new_values.append(new_val)
        
        return MLETable(self.name, new_values, self.field)
    
    def evaluate_at_point(self, point: List[int]) -> int:
        """
        Evaluate the MLE at an arbitrary point (not just boolean).
        
        Uses the multilinear interpolation formula:
            f(r1, ..., rμ) = Σ f(b1, ..., bμ) * Π eq_i(ri, bi)
        where eq_i(r, b) = r*b + (1-r)*(1-b) = r if b=1, (1-r) if b=0
        
        This is equivalent to the iterative update process.
        
        Args:
            point: List of μ field elements [r1, r2, ..., rμ]
            
        Returns:
            Field element f(r1, ..., rμ)
        """
        if len(point) != self.num_vars:
            raise ValueError(f"Point dimension {len(point)} != num_vars {self.num_vars}")
        
        # Iteratively reduce by fixing each variable
        current = self.copy()
        for r in point:
            current = current.update_with_challenge(r)
        
        return current.values[0]
    
    def sum_over_hypercube(self) -> int:
        """
        Compute sum of all values: Σ f(x) for x ∈ {0,1}^μ
        
        This is what SumCheck proves without computing directly on verifier side.
        
        Returns:
            Sum of all table entries
        """
        total = 0
        for v in self.values:
            total = self.field.add(total, v)
        return total
    
    @staticmethod
    def from_function(name: str, num_vars: int, 
                      func, field: 'PrimeField') -> 'MLETable':
        """
        Create an MLE table from a function.
        
        Args:
            name: Table name
            num_vars: Number of variables
            func: Function taking a tuple of bits and returning a field element
            field: The prime field
            
        Example:
            >>> # Create MLE for f(x,y) = x AND y
            >>> mle = MLETable.from_function("and", 2, lambda b: b[0] & b[1], field)
        """
        size = 2 ** num_vars
        values = []
        
        for i in range(size):
            # Convert index to bit tuple
            bits = tuple((i >> j) & 1 for j in range(num_vars))
            values.append(func(bits))
        
        return MLETable(name, values, field)
    
    @staticmethod
    def random(name: str, num_vars: int, field: 'PrimeField',
               sparse: bool = False, sparsity: float = 0.9) -> 'MLETable':
        """
        Create a random MLE table.
        
        Args:
            name: Table name
            num_vars: Number of variables
            field: The prime field
            sparse: If True, most entries are 0 (like selector polynomials)
            sparsity: Fraction of entries that are 0 (if sparse=True)
            
        Returns:
            Random MLETable
        """
        import random
        size = 2 ** num_vars
        
        if sparse:
            values = [
                0 if random.random() < sparsity 
                else random.randint(1, field.prime - 1)
                for _ in range(size)
            ]
        else:
            values = [random.randint(0, field.prime - 1) for _ in range(size)]
        
        return MLETable(name, values, field)


def create_witness_mles(num_vars: int, field: 'PrimeField') -> Tuple['MLETable', 'MLETable', 'MLETable']:
    """
    Create random witness MLEs (w1, w2, w3) for testing.
    
    In real ZKP, these would be the actual computation values.
    
    Args:
        num_vars: Number of variables (circuit size = 2^num_vars)
        field: The prime field
        
    Returns:
        Tuple of (w1, w2, w3) MLETables
    """
    w1 = MLETable.random("w1", num_vars, field)
    w2 = MLETable.random("w2", num_vars, field)
    w3 = MLETable.random("w3", num_vars, field)
    return w1, w2, w3


def create_selector_mles(num_vars: int, field: 'PrimeField') -> dict:
    """
    Create selector MLEs (qL, qR, qM, qO, qC) for testing.
    
    Selectors are typically sparse (mostly 0s with some 1s).
    
    Args:
        num_vars: Number of variables
        field: The prime field
        
    Returns:
        Dictionary of selector MLETables
    """
    return {
        "qL": MLETable.random("qL", num_vars, field, sparse=True, sparsity=0.7),
        "qR": MLETable.random("qR", num_vars, field, sparse=True, sparsity=0.7),
        "qM": MLETable.random("qM", num_vars, field, sparse=True, sparsity=0.8),
        "qO": MLETable.random("qO", num_vars, field, sparse=True, sparsity=0.5),
        "qC": MLETable.random("qC", num_vars, field, sparse=True, sparsity=0.9),
    }


# Example usage
if __name__ == "__main__":
    from ..common.field import PrimeField
    
    print("=" * 60)
    print("MLE TABLE DEMO")
    print("=" * 60)
    
    field = PrimeField(97)
    
    # Create a small MLE table
    # f(X1, X2, X3) with values at all 8 boolean inputs
    mle = MLETable("f", [3, 7, 2, 5, 1, 8, 4, 6], field)
    
    print(f"\nMLE: {mle.name}")
    print(f"Values: {mle.values}")
    print(f"Size: {mle.size}")
    print(f"Num vars: {mle.num_vars}")
    
    # Compute sum
    total = mle.sum_over_hypercube()
    print(f"\nSum over hypercube: {total}")
    print(f"(3+7+2+5+1+8+4+6 = 36 mod 97 = 36)")
    
    # Compute extensions for first pair
    print("\n" + "-" * 60)
    print("EXTENSIONS")
    print("-" * 60)
    
    ext = mle.compute_extension(0, 1, 5)
    print(f"\nExtension for pair (3, 7) at X=0,1,2,3,4:")
    print(f"  {ext}")
    print(f"  Formula: 3 + k*(7-3) = 3 + 4k")
    print(f"  k=0: 3, k=1: 7, k=2: 11, k=3: 15, k=4: 19")
    
    # Update with challenge
    print("\n" + "-" * 60)
    print("UPDATE WITH CHALLENGE")
    print("-" * 60)
    
    challenge = 23
    updated = mle.update_with_challenge(challenge)
    print(f"\nOriginal: {mle.values}")
    print(f"Challenge: {challenge}")
    print(f"Updated: {updated.values}")
    print(f"Size: {mle.size} → {updated.size}")
