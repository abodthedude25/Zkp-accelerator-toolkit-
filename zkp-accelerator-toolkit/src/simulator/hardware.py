"""
Hardware Configuration for SumCheck Accelerator Simulation.

This module defines the hardware parameters that affect SumCheck performance.
The parameters are inspired by zkSpeed and zkPHIRE accelerator designs.

Key Parameters:
    - Processing Elements (PEs): Parallel SumCheck compute units
    - Extension Engines (EEs): Units that compute polynomial extensions
    - Product Lanes (PLs): Units that multiply MLE values
    - Memory bandwidth: HBM throughput (critical for memory-bound phases)

Hardware Context (from papers):
    - zkSpeed: ~366 mm², 801× speedup, fixed-function
    - zkPHIRE: ~294 mm², 1486× speedup, programmable
    - Both use HBM at ~2 TB/s bandwidth
    - BLS12-381 field: 255-bit scalars, 381-bit curve points
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HardwareConfig:
    """
    Hardware configuration for SumCheck accelerator.
    
    This class captures the key hardware parameters that determine
    SumCheck performance. Default values are inspired by zkPHIRE.
    
    Attributes:
        name: Configuration name for identification
        num_pes: Number of Processing Elements
        extension_engines_per_pe: Extension Engines per PE
        product_lanes_per_pe: Product Lanes per PE
        scratchpad_size_kb: Per-PE scratchpad memory
        hbm_bandwidth_gb_s: HBM bandwidth in GB/s
        frequency_ghz: Operating frequency
        modmul_latency: Cycles for modular multiplication
        modadd_latency: Cycles for modular addition
        memory_latency: Initial HBM access latency
        
    Example:
        >>> # zkSpeed-like configuration
        >>> config = HardwareConfig(
        ...     name="zkSpeed-like",
        ...     num_pes=4,
        ...     hbm_bandwidth_gb_s=2000
        ... )
    """
    
    # Identification
    name: str = "default"
    
    # Compute resources
    num_pes: int = 4
    extension_engines_per_pe: int = 7  # zkPHIRE uses 7 EEs
    product_lanes_per_pe: int = 5      # zkPHIRE uses 5 PLs
    
    # Memory resources
    scratchpad_size_kb: int = 64
    hbm_bandwidth_gb_s: float = 2000  # 2 TB/s typical for HBM2E
    
    # Timing parameters (cycles)
    frequency_ghz: float = 1.0
    modmul_latency: int = 22   # BLS12-381 modular multiply
    modadd_latency: int = 1    # Modular add/subtract
    memory_latency: int = 100  # HBM initial access latency
    
    # Field parameters
    field_bits: int = 255      # BLS12-381 scalar field
    bytes_per_element: int = 32  # 255 bits → 32 bytes
    
    def __post_init__(self):
        """Validate configuration."""
        if self.num_pes < 1:
            raise ValueError("num_pes must be at least 1")
        if self.hbm_bandwidth_gb_s <= 0:
            raise ValueError("bandwidth must be positive")
        if self.frequency_ghz <= 0:
            raise ValueError("frequency must be positive")
    
    @property
    def total_extension_engines(self) -> int:
        """Total EEs across all PEs."""
        return self.num_pes * self.extension_engines_per_pe
    
    @property
    def total_product_lanes(self) -> int:
        """Total PLs across all PEs."""
        return self.num_pes * self.product_lanes_per_pe
    
    @property
    def bandwidth_bytes_per_cycle(self) -> float:
        """
        Memory bandwidth in bytes per cycle.
        
        This is the key metric for memory-bound operations.
        """
        bytes_per_second = self.hbm_bandwidth_gb_s * 1e9
        cycles_per_second = self.frequency_ghz * 1e9
        return bytes_per_second / cycles_per_second
    
    @property
    def scratchpad_bytes(self) -> int:
        """Scratchpad size in bytes."""
        return self.scratchpad_size_kb * 1024
    
    @property
    def scratchpad_elements(self) -> int:
        """Number of field elements that fit in scratchpad."""
        return self.scratchpad_bytes // self.bytes_per_element
    
    def cycles_to_ms(self, cycles: int) -> float:
        """Convert cycles to milliseconds."""
        return cycles / (self.frequency_ghz * 1e6)
    
    def ms_to_cycles(self, ms: float) -> int:
        """Convert milliseconds to cycles."""
        return int(ms * self.frequency_ghz * 1e6)
    
    def summary(self) -> str:
        """Return configuration summary string."""
        return (
            f"HardwareConfig '{self.name}':\n"
            f"  Compute:\n"
            f"    PEs: {self.num_pes}\n"
            f"    EEs/PE: {self.extension_engines_per_pe} (total: {self.total_extension_engines})\n"
            f"    PLs/PE: {self.product_lanes_per_pe} (total: {self.total_product_lanes})\n"
            f"  Memory:\n"
            f"    HBM bandwidth: {self.hbm_bandwidth_gb_s:.0f} GB/s\n"
            f"    Bandwidth/cycle: {self.bandwidth_bytes_per_cycle:.1f} bytes\n"
            f"    Scratchpad/PE: {self.scratchpad_size_kb} KB ({self.scratchpad_elements} elements)\n"
            f"  Timing:\n"
            f"    Frequency: {self.frequency_ghz:.1f} GHz\n"
            f"    ModMul latency: {self.modmul_latency} cycles\n"
            f"    Memory latency: {self.memory_latency} cycles"
        )
    
    def __repr__(self) -> str:
        return (f"HardwareConfig(name='{self.name}', pes={self.num_pes}, "
                f"bw={self.hbm_bandwidth_gb_s}GB/s)")


# =============================================================================
# PREDEFINED CONFIGURATIONS
# =============================================================================

def create_zkspeed_config() -> HardwareConfig:
    """
    Configuration inspired by zkSpeed.
    
    zkSpeed characteristics:
        - First HyperPlonk accelerator
        - Fixed-function SumCheck
        - 366 mm² area
        - 801× speedup over CPU
    """
    return HardwareConfig(
        name="zkSpeed-like",
        num_pes=4,
        extension_engines_per_pe=4,  # Unified PE design
        product_lanes_per_pe=4,
        hbm_bandwidth_gb_s=2000,
        scratchpad_size_kb=128,
    )


def create_zkphire_config() -> HardwareConfig:
    """
    Configuration inspired by zkPHIRE.
    
    zkPHIRE characteristics:
        - Programmable SumCheck
        - Supports custom gates
        - 294 mm² area
        - 1486× speedup over CPU
        - 11.87× faster than zkSpeed at iso-area
    """
    return HardwareConfig(
        name="zkPHIRE-like",
        num_pes=4,
        extension_engines_per_pe=7,
        product_lanes_per_pe=5,
        hbm_bandwidth_gb_s=2000,
        scratchpad_size_kb=64,  # Uses tile-based streaming instead
    )


def create_minimal_config() -> HardwareConfig:
    """
    Minimal configuration for testing.
    
    Useful for understanding basic behavior without
    parallelism complexity.
    """
    return HardwareConfig(
        name="minimal",
        num_pes=1,
        extension_engines_per_pe=1,
        product_lanes_per_pe=1,
        hbm_bandwidth_gb_s=512,
        scratchpad_size_kb=32,
    )


def create_high_bandwidth_config() -> HardwareConfig:
    """
    Configuration with very high bandwidth.
    
    Useful for exploring compute-bound scenarios.
    """
    return HardwareConfig(
        name="high-bandwidth",
        num_pes=4,
        extension_engines_per_pe=7,
        product_lanes_per_pe=5,
        hbm_bandwidth_gb_s=8000,  # 8 TB/s (futuristic)
        scratchpad_size_kb=256,
    )


def create_high_compute_config() -> HardwareConfig:
    """
    Configuration with many compute units.
    
    Useful for exploring memory-bound scenarios.
    """
    return HardwareConfig(
        name="high-compute",
        num_pes=16,
        extension_engines_per_pe=8,
        product_lanes_per_pe=8,
        hbm_bandwidth_gb_s=2000,
        scratchpad_size_kb=64,
    )


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("HARDWARE CONFIGURATION EXAMPLES")
    print("=" * 60)
    
    configs = [
        create_zkspeed_config(),
        create_zkphire_config(),
        create_minimal_config(),
        create_high_bandwidth_config(),
        create_high_compute_config(),
    ]
    
    for config in configs:
        print(f"\n{config.summary()}")
        print("-" * 60)
