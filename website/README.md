# ZKP Accelerator Interactive Visualizer

An interactive web-based visualization of ZKP concepts from the zkSpeed and zkPHIRE papers.

## How to Use

Simply open `index.html` or `advanced.html` in any modern web browser. No server required!

```bash
# Option 1: Open directly
open index.html

# Option 2: Use Python's built-in server (for CORS if needed)
python -m http.server 8000
# Then visit http://localhost:8000
```

## Visualizations

### Basic (index.html)
1. **SumCheck Protocol** - Watch the table halving and round polynomial generation
2. **Gate Comparison** - Interactive Vanilla vs Jellyfish gate analysis
3. **Memory Patterns** - See streaming vs random access patterns
4. **Prover Pipeline** - Animated view of the full ZK proving flow
5. **Hardware Explorer** - Adjust PEs, bandwidth, degree to see bottlenecks
6. **Workload Analysis** - Compare workloads accounting for gate reduction

### Advanced (advanced.html)
1. **MSM Algorithm** - Pippenger's bucket method visualization
2. **NTT Butterfly** - Cooley-Tukey FFT-style diagram
3. **Polynomial Commitments** - KZG, IPA, and FRI schemes compared
4. **Crossover Point** - Memory vs compute bottleneck visualization

## Key Concepts Demonstrated

### From zkSpeed
- SumCheck protocol mechanics
- Streaming memory access patterns
- Hardware PE architecture
- MSM acceleration

### From zkPHIRE
- High-degree gates (Jellyfish)
- Gate reduction tradeoffs
- Programmable polynomial evaluation
- Memory-compute crossover analysis

## Technologies Used
- React 18 (via CDN)
- Tailwind CSS (via CDN)
- Pure HTML/JavaScript (no build step required)
