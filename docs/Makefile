# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    	=
SPHINXBUILD   	= sphinx-build
APIBUILD				= sphinx-apidoc 
SOURCEDIR     	= source
APIDIR			= $(SOURCEDIR)/generated/api
BUILDDIR      	= build
TEMPLATEDIR		= source/_templates
PKGDIR		  	= ../src/apollon


build: api html

fresh: clean build

clean:
	rm -rf build/* source/generated/api/*

api:
	$(APIBUILD) -TefM --templatedir $(TEMPLATEDIR) -o $(APIDIR) $(PKGDIR)

html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR) $(SPHINXOPTS) 

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
#%: Makefile
	#@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
