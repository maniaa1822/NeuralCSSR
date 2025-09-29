# Neural CSSR Research Presentation

This folder contains the LaTeX source and compiled PDF for the Neural CSSR research paper/presentation.

## Files

- `neuralcssr_presentation.tex` - Main LaTeX source file
- `neuralcssr_presentation.pdf` - Compiled PDF document (10 pages)
- `neuralcssr_presentation.aux` - LaTeX auxiliary file
- `neuralcssr_presentation.log` - LaTeX compilation log
- `neuralcssr_presentation.out` - LaTeX outline file for hyperref

## Content Overview

The document covers:

1. **Abstract & Introduction** - Neural CSSR framework overview
2. **Background** - Epsilon machines and classical CSSR limitations
3. **Neural CSSR Framework** - EBM-BLM and AR-BLM architectures
4. **Fast Unsupervised Algorithm** - Three-stage clustering approach
5. **Experimental Results** - Performance on Golden Mean, Even Process, Seven-State Human
6. **Novel Contributions** - Backward stability, representative sampling, state remerging
7. **Implementation Details** - Caching optimization and code examples
8. **Future Directions** - Theoretical extensions and applications

## Key Technical Contributions

- **O(k) complexity** in Stage B through representative sampling
- **Backward stability refinement** for minimal suffix identification
- **Three-stage clustering** architecture (emission → conditional JS → remerging)
- **85% cache hit rate** for k-step distribution computations
- **High purity recovery** across canonical computational mechanics processes

## Experimental Results Summary

| Process | States | Recovery | Performance |
|---------|--------|----------|-------------|
| Golden Mean | 2 | 100% accurate | 0.656 bits/symbol |
| Even Process | 2 | 96-98% purity | 0.720 bits/symbol |
| Seven-State Human | 7 | Complete recovery | Variable entropy |

## Compilation

To recompile the PDF:

```bash
cd docs/presentations/neuralcssr-paper/
pdflatex neuralcssr_presentation.tex
pdflatex neuralcssr_presentation.tex  # Run twice for cross-references
```

## Dependencies

LaTeX packages required:
- `article` (document class)
- `amsmath`, `amsfonts`, `amssymb` (mathematics)
- `algorithm`, `algorithmic` (pseudocode)
- `listings` (code formatting)
- `hyperref` (hyperlinks)
- `booktabs` (professional tables)
- `tikz` (diagrams, if needed)

## Usage

This document can be used for:
- Conference paper submission
- Technical presentation slides (adapt to Beamer)
- Blog post or technical report
- Internal research documentation

## Related Code

The implementation discussed in this paper is primarily in:
- `nanoGPT/js_analysis/unsupervised_fast_original.py` - Main algorithm
- `experiments/ebm/models.py` - Neural probability providers
- `run_neural_cssr.py` - Full pipeline integration