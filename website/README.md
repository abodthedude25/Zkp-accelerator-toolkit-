# ZKP Accelerator Interactive Visualizer

An interactive web-based visualization of ZKP concepts from the zkSpeed and zkPHIRE papers.

## 🚀 How to Use

Simply open any HTML file in a modern web browser. No server required!

```bash
open index.html        # Basic visualizations
open advanced.html     # MSM, NTT, Commitments
open papers.html       # Paper figures & Circuit compiler
```

## 📄 Pages

### Basic (index.html)
| Tab | Description |
|-----|-------------|
| ∑ SumCheck | Watch table halving and round progression |
| ⊕ Gate Comparison | Interactive Vanilla vs Jellyfish analysis |
| 📊 Memory Patterns | Streaming vs random access animation |
| ⚙️ Prover Pipeline | Full proving flow with time breakdown |
| 🔧 Hardware Explorer | Adjust PEs, bandwidth, degree |
| 📈 Workload Analysis | Compare with gate reduction |

### Advanced (advanced.html)
| Tab | Description |
|-----|-------------|
| 🔢 MSM Algorithm | Pippenger's bucket method |
| 🦋 NTT Butterfly | Cooley-Tukey transform diagram |
| 🔐 Poly Commitments | KZG, IPA, FRI comparison |
| 📈 Crossover Point | Memory vs compute bottleneck |

### Paper Analysis (papers.html) ⭐
| Tab | Description |
|-----|-------------|
| 📊 zkSpeed Fig 9 | Reproduced cycle breakdown chart |
| 📈 zkPHIRE Fig 11 | Reproduced speedup analysis |
| ⚙️ Circuit Compiler | Poseidon/ECDSA/MiMC → gates |
| 📄 Paper Comparison | Side-by-side comparison table |
| 🧮 Calculator | Input requirements → recommendations |

## 🎯 Key Concepts

### From zkSpeed
- SumCheck protocol mechanics
- Streaming memory access patterns
- Hardware PE architecture
- Cycle breakdown analysis

### From zkPHIRE
- High-degree gates (Jellyfish)
- Gate reduction tradeoffs
- Per-gate cost vs effective speedup
- Memory-compute crossover

## 🛠️ Technologies
- React 18 (via CDN)
- Tailwind CSS (via CDN)
- Pure HTML/JS (no build required)
