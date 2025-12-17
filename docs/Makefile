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
	make download
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


deploy:	html
	python deploy.py

download:
	@echo "Downloading example notebooks to notebooks ..."
	@mkdir -p notebooks
	curl -L -o notebooks/diffusion_2d_unconditional.ipynb \
		https://raw.githubusercontent.com/aurelio-amerio/GenSBI-examples/refs/heads/main/examples/diffusion_2d_unconditional.ipynb
	curl -L -o notebooks/flow_matching_2d_unconditional.ipynb \
		https://raw.githubusercontent.com/aurelio-amerio/GenSBI-examples/refs/heads/main/examples/flow_matching_2d_unconditional.ipynb

	curl -L -o notebooks/two_moons_flow_flux.ipynb \
		https://raw.githubusercontent.com/aurelio-amerio/GenSBI-examples/refs/heads/main/examples/sbi-benchmarks/two_moons/flow_flux/two_moons_flow_flux.ipynb
	curl -L -o notebooks/two_moons_flow_simformer.ipynb \
		https://raw.githubusercontent.com/aurelio-amerio/GenSBI-examples/refs/heads/main/examples/sbi-benchmarks/two_moons/flow_simformer/two_moons_flow_simformer.ipynb
		
	curl -L -o notebooks/gaussian_linear_flux1joint.ipynb \
		https://raw.githubusercontent.com/aurelio-amerio/GenSBI-examples/refs/heads/main/examples/sbi-benchmarks/gaussian_linear/flow_flux1joint/gaussian_linear_flow_flux1joint.ipynb

	curl -L -o notebooks/slcp_flow_simformer.ipynb \
		https://raw.githubusercontent.com/aurelio-amerio/GenSBI-examples/refs/heads/main/examples/sbi-benchmarks/slcp/flow_simformer/slcp_flow_simformer.ipynb

	curl -L -o notebooks/my_first_model.ipynb \
		https://raw.githubusercontent.com/aurelio-amerio/GenSBI-examples/refs/heads/main/examples/getting_started/my_first_model.ipynb

	@echo "Done."

