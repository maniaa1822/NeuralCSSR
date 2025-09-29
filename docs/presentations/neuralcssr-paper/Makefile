# Makefile for Neural CSSR Presentation

# Main document name (without extension)
DOC = neuralcssr_presentation

# LaTeX compiler
LATEX = pdflatex

# Default target
all: pdf

# Compile PDF (run twice for cross-references)
pdf: $(DOC).pdf

$(DOC).pdf: $(DOC).tex
	$(LATEX) $(DOC).tex
	$(LATEX) $(DOC).tex

# Clean auxiliary files
clean:
	rm -f $(DOC).aux $(DOC).log $(DOC).out $(DOC).toc $(DOC).lof $(DOC).lot

# Clean everything including PDF
distclean: clean
	rm -f $(DOC).pdf

# Force rebuild
rebuild: distclean pdf

# Open PDF (Linux)
view: $(DOC).pdf
	xdg-open $(DOC).pdf

# Show help
help:
	@echo "Available targets:"
	@echo "  pdf       - Compile LaTeX to PDF (default)"
	@echo "  clean     - Remove auxiliary files"
	@echo "  distclean - Remove all generated files including PDF"
	@echo "  rebuild   - Clean and rebuild PDF"
	@echo "  view      - Open PDF with default viewer"
	@echo "  help      - Show this help message"

.PHONY: all pdf clean distclean rebuild view help