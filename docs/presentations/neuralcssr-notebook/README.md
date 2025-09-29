# Neural CSSR Research Notebook

A friendly, notebook-style presentation of our Neural CSSR research work.

## About This Document

This is a more informal, accessible version of our Neural CSSR paper - designed like a research notebook with:
- Conversational explanations
- Code snippets with syntax highlighting
- Colored boxes for notes, results, and insights
- Step-by-step algorithm walkthroughs
- Interactive feel with practical examples

## Contents

- **Algorithm Overview**: Three-stage clustering approach
- **Neural Models**: EBM-BLM and AR-BLM architectures
- **Experimental Results**: Golden Mean, Even Process, Seven-State Human
- **Implementation Details**: Key code snippets and optimizations
- **Performance Analysis**: Caching, complexity, and scalability

## Building the Document

### Quick Start
```bash
make pdf
```

### All Available Commands
```bash
make          # Build PDF
make clean    # Remove auxiliary files
make rebuild  # Clean and rebuild
make view     # Open PDF in viewer
make help     # Show all options
```

### Requirements
- `pdflatex` (from texlive distribution)
- LaTeX packages: `amsmath`, `algorithm`, `listings`, `tcolorbox`, `xcolor`, `tikz`

Install on Ubuntu/Debian:
```bash
sudo apt install texlive-latex-extra texlive-science
```

## Document Features

### Special Environments
- **Code Boxes**: Algorithm pseudocode and Python snippets
- **Note Boxes**: Key insights and explanations
- **Result Boxes**: Experimental findings and metrics
- **Colored Syntax**: Python code with proper highlighting

### Visual Design
- Clean, readable layout with 1-inch margins
- Consistent color scheme throughout
- Easy navigation with clear section headers
- Technical content balanced with intuitive explanations

## Related Files

This notebook complements the formal paper in `../neuralcssr-paper/`. Both documents cover the same research but with different presentation styles:

- **Formal Paper**: Academic style, comprehensive technical details
- **This Notebook**: Accessible style, tutorial-like explanations

## Research Context

Part of the larger Neural CSSR project exploring epsilon machine discovery using neural networks as probability providers. See the main repository for:
- Full implementation: `nanoGPT/js_analysis/unsupervised_fast_original.py`
- Neural models: `experiments/ebm/models.py`
- Training scripts: `experiments/ebm/train_golden_mean.py`

## Output

Successfully builds to a ~12-page notebook-style document perfect for presentations, tutorials, or sharing with collaborators who want a gentler introduction to the research.