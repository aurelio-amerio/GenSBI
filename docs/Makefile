# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


deploy:	html
	python deploy.py

download-examples:
	@echo "Downloading example notebooks to notebooks ..."
	@mkdir -p notebooks
	curl -L -o notebooks/diffusion_2d_unconditional.ipynb \
		https://raw.githubusercontent.com/aurelio-amerio/GenSBI-examples/refs/heads/main/examples/diffusion_2d_unconditional.ipynb
	curl -L -o notebooks/flow_matching_2d_unconditional.ipynb \
		https://raw.githubusercontent.com/aurelio-amerio/GenSBI-examples/refs/heads/main/examples/flow_matching_2d_unconditional.ipynb


