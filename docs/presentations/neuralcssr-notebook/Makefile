# Makefile for Neural CSSR Notebook
# Usage: make [target]

MAIN = neuralcssr_notebook
TEX_FILE = $(MAIN).tex
PDF_FILE = $(MAIN).pdf

# LaTeX compiler
LATEX = pdflatex
LATEX_FLAGS = -interaction=batchmode -halt-on-error

# Default target
.PHONY: all
all: pdf

# Build PDF
.PHONY: pdf
pdf: $(PDF_FILE)

# Build HTML (faster rendering, naturally scrollable)
.PHONY: html
html: $(MAIN).html

$(MAIN).html: $(TEX_FILE)
	@echo "Building HTML..."
	@htlatex $(TEX_FILE) "xhtml,mathml" " -cunihtf -utf8" > /dev/null 2>&1
	@echo "✓ HTML generated: $(MAIN).html"

$(PDF_FILE): $(TEX_FILE)
	@echo "Building $(PDF_FILE)..."
	@$(LATEX) $(LATEX_FLAGS) $(TEX_FILE) > /dev/null
	@$(LATEX) $(LATEX_FLAGS) $(TEX_FILE) > /dev/null  # Second pass for cross-references
	@echo "✓ PDF generated: $(PDF_FILE)"

# Clean auxiliary files
.PHONY: clean
clean:
	@echo "Cleaning auxiliary files..."
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.fls *.fdb_latexmk *.synctex.gz
	@echo "✓ Clean complete"

# Clean everything including PDF
.PHONY: distclean
distclean: clean
	@echo "Removing PDF file..."
	rm -f $(PDF_FILE)
	@echo "✓ Full clean complete"

# Rebuild from scratch
.PHONY: rebuild
rebuild: distclean pdf

# View PDF (if available)
.PHONY: view
view: $(PDF_FILE)
	@if command -v xdg-open >/dev/null 2>&1; then \
		xdg-open $(PDF_FILE); \
	elif command -v open >/dev/null 2>&1; then \
		open $(PDF_FILE); \
	else \
		echo "No PDF viewer found. Please open $(PDF_FILE) manually."; \
	fi

# Show help
.PHONY: help
help:
	@echo "Neural CSSR Notebook Build System"
	@echo "================================="
	@echo ""
	@echo "Available targets:"
	@echo "  pdf       - Build the PDF (default)"
	@echo "  clean     - Remove auxiliary files"
	@echo "  distclean - Remove all generated files"
	@echo "  rebuild   - Clean and rebuild PDF"
	@echo "  view      - Open PDF in default viewer"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Example usage:"
	@echo "  make        # Build PDF"
	@echo "  make clean  # Clean up"
	@echo "  make view   # Open PDF"