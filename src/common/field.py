"""
Finite Field Arithmetic for ZKP Computations.

This module implements modular arithmetic over prime fields, which is the
foundation of all ZKP computations. In real ZKP systems like HyperPlonk,
the field is typically BLS12-381 with a 255-bit prime. Here we support
both small primes (for visualization/testing) and conceptually large primes.

Key Concepts:
    - All arithmetic is done modulo a prime p
    - Addition: (a + b) mod p
    - Multiplication: (a * b) mod p  
    - Subtraction: (a - b + p) mod p (to keep positive)
    - Division: a * b^(-1) mod p (multiply by modular inverse)
    - Inversion: Find b such that a * b = 1 mod p (expensive!)

Example:
    >>> field = PrimeField(97)  # Small prime for demo
    >>> a = field.element(45)
    >>> b = field.element(67)
    >>> c = a + b  # (45 + 67) mod 97 = 15
    >>> print(c)
    FieldElement(15, mod 97)

For Hardware Context:
    - zkSpeed/zkPHIRE use BLS12-381: p ≈ 2^255 (255-bit numbers)
    - Each modular multiplication takes ~22 cycles on custom hardware
    - Modular inversion takes ~509 cycles (23x slower than multiplication!)
    - This is why Montgomery batching is crucial for performance
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Union, List, Optional
import random


@dataclass
class FieldElement:
    """
    An element of a prime field Z_p.
    
    This class represents a single value in modular arithmetic.
    All operations automatically reduce the result modulo p.
    
    Attributes:
        value: The integer value (always in range [0, p-1])
        field: Reference to the parent PrimeField
        
    Example:
        >>> field = PrimeField(97)
        >>> a = FieldElement(45, field)
        >>> b = FieldElement(67, field)
        >>> print(a + b)  # 45 + 67 = 112 → 112 mod 97 = 15
        FieldElement(15, mod 97)
    """
    value: int
    field: 'PrimeField'
    
    def __post_init__(self):
        """Ensure value is reduced modulo p."""
        self.value = self.value % self.field.prime
    
    def __repr__(self) -> str:
        return f"FieldElement({self.value}, mod {self.field.prime})"
    
    def __str__(self) -> str:
        return str(self.value)
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, FieldElement):
            return self.value == other.value and self.field.prime == other.field.prime
        if isinstance(other, int):
            return self.value == (other % self.field.prime)
        return False
    
    def __hash__(self) -> int:
        return hash((self.value, self.field.prime))
    
    # Arithmetic Operations
    
    def __add__(self, other: Union[FieldElement, int]) -> FieldElement:
        """Addition in the field: (a + b) mod p"""
        other_val = other.value if isinstance(other, FieldElement) else other
        return FieldElement((self.value + other_val) % self.field.prime, self.field)
    
    def __radd__(self, other: int) -> FieldElement:
        return self.__add__(other)
    
    def __sub__(self, other: Union[FieldElement, int]) -> FieldElement:
        """Subtraction in the field: (a - b + p) mod p"""
        other_val = other.value if isinstance(other, FieldElement) else other
        return FieldElement((self.value - other_val + self.field.prime) % self.field.prime, self.field)
    
    def __rsub__(self, other: int) -> FieldElement:
        return FieldElement((other - self.value + self.field.prime) % self.field.prime, self.field)
    
    def __mul__(self, other: Union[FieldElement, int]) -> FieldElement:
        """Multiplication in the field: (a * b) mod p"""
        other_val = other.value if isinstance(other, FieldElement) else other
        return FieldElement((self.value * other_val) % self.field.prime, self.field)
    
    def __rmul__(self, other: int) -> FieldElement:
        return self.__mul__(other)
    
    def __truediv__(self, other: Union[FieldElement, int]) -> FieldElement:
        """Division in the field: a * b^(-1) mod p"""
        if isinstance(other, FieldElement):
            return self * other.inverse()
        return self * self.field.element(other).inverse()
    
    def __neg__(self) -> FieldElement:
        """Negation: -a = p - a"""
        return FieldElement((self.field.prime - self.value) % self.field.prime, self.field)
    
    def __pow__(self, exp: int) -> FieldElement:
        """
        Exponentiation using square-and-multiply.
        
        This is the standard algorithm for modular exponentiation.
        Time complexity: O(log exp) multiplications.
        
        Note: For exp = p-2, this computes the modular inverse (Fermat's little theorem)
        """
        if exp < 0:
            # a^(-n) = (a^(-1))^n
            return self.inverse() ** (-exp)
        
        result = self.field.one()
        base = FieldElement(self.value, self.field)
        
        while exp > 0:
            if exp & 1:  # If least significant bit is 1
                result = result * base
            base = base * base
            exp >>= 1
            
        return result
    
    def inverse(self) -> FieldElement:
        """
        Compute modular inverse using Extended Euclidean Algorithm.
        
        Finds b such that a * b ≡ 1 (mod p)
        
        This is one of the MOST EXPENSIVE operations in ZKP:
            - ~509 iterations for 255-bit primes
            - Compare to ~22 cycles for multiplication
            - This is why zkSpeed uses Montgomery batching!
        
        Raises:
            ValueError: If self.value is 0 (no inverse exists)
            
        Returns:
            FieldElement b such that self * b = 1
        """
        if self.value == 0:
            raise ValueError("Cannot invert zero")
        
        # Extended Euclidean Algorithm
        # We want to find x such that a*x ≡ 1 (mod p)
        # This is equivalent to finding x, y such that a*x + p*y = gcd(a, p) = 1
        
        old_r, r = self.value, self.field.prime
        old_s, s = 1, 0
        
        while r != 0:
            quotient = old_r // r
            old_r, r = r, old_r - quotient * r
            old_s, s = s, old_s - quotient * s
        
        # old_r should be 1 (gcd), old_s is the inverse
        if old_r != 1:
            raise ValueError(f"No inverse exists (gcd = {old_r})")
        
        return FieldElement(old_s % self.field.prime, self.field)
    
    def is_zero(self) -> bool:
        """Check if this element is zero."""
        return self.value == 0
    
    def is_one(self) -> bool:
        """Check if this element is one."""
        return self.value == 1


class PrimeField:
    """
    A prime field Z_p for modular arithmetic.
    
    This class represents the mathematical structure where all ZKP
    computations take place. It provides factory methods for creating
    field elements and utility functions.
    
    Attributes:
        prime: The prime modulus p
        
    Common Primes in ZKP:
        - 97: Good for testing/visualization (small, easy to verify by hand)
        - 2^64 - 2^32 + 1: Goldilocks prime (fast on 64-bit CPUs)
        - BLS12-381 scalar field: ~2^255 (used in Ethereum, Zcash)
        
    Example:
        >>> field = PrimeField(97)
        >>> a = field.element(45)
        >>> b = field.random()
        >>> print(field.add(a.value, b.value))
    """
    
    # Common primes used in ZKP systems
    SMALL_TEST_PRIME = 97
    GOLDILOCKS_PRIME = (1 << 64) - (1 << 32) + 1  # 2^64 - 2^32 + 1
    
    def __init__(self, prime: int):
        """
        Initialize a prime field.
        
        Args:
            prime: The prime modulus. Should be prime for correct behavior.
                   (We don't verify primality for performance reasons)
        """
        if prime < 2:
            raise ValueError("Prime must be at least 2")
        self.prime = prime
    
    def __repr__(self) -> str:
        return f"PrimeField({self.prime})"
    
    def element(self, value: int) -> FieldElement:
        """Create a field element from an integer."""
        return FieldElement(value % self.prime, self)
    
    def zero(self) -> FieldElement:
        """Return the additive identity (0)."""
        return FieldElement(0, self)
    
    def one(self) -> FieldElement:
        """Return the multiplicative identity (1)."""
        return FieldElement(1, self)
    
    def random(self, exclude_zero: bool = False) -> FieldElement:
        """
        Generate a random field element.
        
        Args:
            exclude_zero: If True, never returns zero (useful for testing inverses)
            
        Returns:
            A random FieldElement in [0, p-1] or [1, p-1]
        """
        if exclude_zero:
            return FieldElement(random.randint(1, self.prime - 1), self)
        return FieldElement(random.randint(0, self.prime - 1), self)
    
    # Direct arithmetic (without creating FieldElement objects)
    # Useful for performance-critical inner loops
    
    def add(self, a: int, b: int) -> int:
        """Add two integers in the field."""
        return (a + b) % self.prime
    
    def sub(self, a: int, b: int) -> int:
        """Subtract two integers in the field."""
        return (a - b + self.prime) % self.prime
    
    def mul(self, a: int, b: int) -> int:
        """Multiply two integers in the field."""
        return (a * b) % self.prime
    
    def neg(self, a: int) -> int:
        """Negate an integer in the field."""
        return (self.prime - a) % self.prime
    
    def inv(self, a: int) -> int:
        """Compute modular inverse of an integer."""
        return self.element(a).inverse().value
    
    def pow(self, base: int, exp: int) -> int:
        """Compute base^exp in the field."""
        return pow(base, exp, self.prime)


class BatchInverter:
    """
    Batch modular inversion using Montgomery's trick.
    
    Computing modular inverses is expensive (~509 cycles for 255-bit primes).
    Montgomery's trick lets us compute n inverses with only 1 inversion
    plus O(n) multiplications, dramatically improving performance.
    
    Algorithm:
        1. Compute partial products: P[i] = a[0] * a[1] * ... * a[i]
        2. Invert final product: I = P[n-1]^(-1)
        3. Recover individual inverses by "peeling off" elements
        
    Cost Analysis:
        - Without batching: n inversions = n * 509 cycles
        - With batching: 1 inversion + 3(n-1) multiplications
                       = 509 + 3(n-1)*22 cycles
        - For n=64: 509 + 63*66 = 4667 cycles vs 64*509 = 32576 cycles
        - Speedup: ~7x for batch size 64!
        
    This is why zkSpeed uses batch size 64 for modular inversions.
    
    Example:
        >>> field = PrimeField(97)
        >>> inverter = BatchInverter(field)
        >>> elements = [field.element(i) for i in range(1, 11)]
        >>> inverses = inverter.invert_batch(elements)
        >>> # Verify: a * a^(-1) = 1
        >>> all((e * inv).is_one() for e, inv in zip(elements, inverses))
        True
    """
    
    def __init__(self, field: PrimeField):
        """
        Initialize batch inverter.
        
        Args:
            field: The prime field to operate in
        """
        self.field = field
    
    def invert_batch(self, elements: List[FieldElement]) -> List[FieldElement]:
        """
        Compute inverses of all elements in a batch.
        
        Args:
            elements: List of field elements to invert
            
        Returns:
            List of inverses in the same order
            
        Raises:
            ValueError: If any element is zero
        """
        if not elements:
            return []
        
        n = len(elements)
        
        # Check for zeros
        for i, e in enumerate(elements):
            if e.is_zero():
                raise ValueError(f"Cannot invert zero (element {i})")
        
        # Step 1: Compute partial products
        # products[i] = elements[0] * elements[1] * ... * elements[i]
        products = [elements[0]]
        for i in range(1, n):
            products.append(products[i-1] * elements[i])
        
        # Step 2: Invert the final product (the ONE expensive inversion)
        inv = products[n-1].inverse()
        
        # Step 3: Recover individual inverses (backward pass)
        inverses = [self.field.zero()] * n
        
        for i in range(n-1, 0, -1):
            # inverses[i] = inv * products[i-1]
            # This gives us elements[i]^(-1) because:
            # inv = (a[0]*...*a[i])^(-1)
            # inv * products[i-1] = (a[0]*...*a[i])^(-1) * (a[0]*...*a[i-1])
            #                     = a[i]^(-1)
            inverses[i] = inv * products[i-1]
            
            # Update inv for next iteration: multiply by elements[i]
            # inv becomes (a[0]*...*a[i-1])^(-1)
            inv = inv * elements[i]
        
        # First element: inv is now a[0]^(-1)
        inverses[0] = inv
        
        return inverses
    
    def invert_batch_raw(self, values: List[int]) -> List[int]:
        """
        Batch inversion on raw integers (for performance-critical code).
        
        Args:
            values: List of integers to invert (must be non-zero)
            
        Returns:
            List of inverse integers
        """
        elements = [self.field.element(v) for v in values]
        inverses = self.invert_batch(elements)
        return [inv.value for inv in inverses]


# Utility functions

def verify_field_element(field: PrimeField, value: int) -> bool:
    """Check if a value is a valid field element."""
    return 0 <= value < field.prime


def random_field_elements(field: PrimeField, count: int, 
                          exclude_zero: bool = False) -> List[FieldElement]:
    """Generate multiple random field elements."""
    return [field.random(exclude_zero) for _ in range(count)]


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("FINITE FIELD ARITHMETIC DEMO")
    print("=" * 60)
    
    # Create a small field for demonstration
    field = PrimeField(97)
    print(f"\nField: Z_{field.prime}")
    
    # Basic arithmetic
    a = field.element(45)
    b = field.element(67)
    
    print(f"\na = {a.value}, b = {b.value}")
    print(f"a + b = {(a + b).value}  (45 + 67 = 112 → 112 mod 97 = 15)")
    print(f"a - b = {(a - b).value}  (45 - 67 = -22 → -22 + 97 = 75)")
    print(f"a * b = {(a * b).value}  (45 * 67 = 3015 → 3015 mod 97 = 7)")
    
    # Inverse
    a_inv = a.inverse()
    print(f"\na^(-1) = {a_inv.value}")
    print(f"a * a^(-1) = {(a * a_inv).value} (should be 1)")
    
    # Batch inversion
    print("\n" + "-" * 60)
    print("BATCH INVERSION DEMO")
    print("-" * 60)
    
    inverter = BatchInverter(field)
    elements = [field.element(i) for i in range(1, 11)]
    print(f"\nElements: {[e.value for e in elements]}")
    
    inverses = inverter.invert_batch(elements)
    print(f"Inverses: {[inv.value for inv in inverses]}")
    
    # Verify
    products = [(e * inv).value for e, inv in zip(elements, inverses)]
    print(f"Products (all should be 1): {products}")
