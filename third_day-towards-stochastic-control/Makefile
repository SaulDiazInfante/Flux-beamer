# Makefile for LaTeX
##################################################################
# Use:
# make
# make clean
# options for ps2pdf: http://pages.cs.wisc.edu/~ghost/doc/AFPL/6.50/Ps2pdf.htm

TEX := $(shell ls *.tex)
FILES = *.tex Makefile *.bst *.pdf *.bib
FOLDER = modelling_and_control_scholl_CIMAT_Nov-2019/third_day-modelling-with-noise/
OTHER = *~ *.aux *.dvi *.toc *.bbl *.blg *.out *.thm *.ps *.idx *.ilg *.ind *.tdo *.pdf *.tar.gz *.log *.spl *.synctex.gz
LATEX	= latex
BIBTEX	= bibtex
MAKEINDEX = makeindex
XDVI	= xdvi -gamma 4
DVIPS	= dvips
DVIPDF  = dvipdft
L2H	= latex2html
GH	= gv
FILES = *.tex *.sty *.bbl *.bib *.pdf Makefile  
FOLDERS = \Imagenes \images \assets 
NAMETAR1 := $(shell date '+%Y%b%d_%H_%M')
NAMETAR = "$(NAMETAR1)_Beamer_modelling_and_control_scholl_CIMAT_2019_day03.tar.gz"
NAMEZIP = "$(NAMETAR1)_Beamer_modelling_and_control_scholl_CIMAT_2019_day03.zip"

pdflatex:third_day_main.tex
	/usr/local/texlive/2019/bin/x86_64-linux/pdflatex --synctex=1 third_day_main.tex
	# /usr/local/texlive/2019/bin/x86_64-linux/bibtex third_day_main.aux
	./biblio.sh
	/usr/local/texlive/2019/bin/x86_64-linux/pdflatex --synctex=1 third_day_main.tex
	/usr/local/texlive/2019/bin/x86_64-linux/pdflatex --synctex=1 third_day_main.tex

clean:
	rm -f $(OTHER) 

tar: $(FILES)
	tar -cvf $(NAMETAR) $(FOLDER)

zip: $(FILES)
	zip -r $(NAMEZIP) $(FOLDER)
