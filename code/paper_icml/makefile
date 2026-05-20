main.pdf: main.tex
	pdflatex main.tex
	bibtex main.aux
	pdflatex main.tex
	pdflatex main.tex

diff:
	latexdiff main.tex main_v0.tex > diff_01.tex
	pdflatex diff_01.tex
	bibtex diff_01.aux
	pdflatex diff_01.tex
	pdflatex diff_01.tex

#.PHONY: clean

#clean:
#	rm -f main.pdf diff.pdf *.aux *.bbl *.dvi *.blg *.log
