# Future Development Plans

The following tasks are planned for future development:

- [x] Implement OT flow matching techniques.
- [x] Implement diffusion models (EDM and score matching).
- [x] Implement Transformer-based models for conditional posterior estimation (Flux1 and Simformer).
- [x] Unify the API for flow matching and diffusion models.
- [x] Implement wrappers to make training of flow matching and diffusion models similar.
- [x] Write tests for core functionalities.
- [ ] Consider implementing classifier free guidance for conditional models.
- [ ] Add more examples and benchmarks.
- [ ] Improve documentation and tutorials.
- [x] Provide SOTA pre-trained models and checkpoints for some SBI benchmark cases
- [x] Implement VAE training pipeline
- [ ] ... implement better loss functions for 1D and 2D VAEs (e.g. windowed frequency decomposition, and perceptual loss for images)
- [x] Implement wrapper to run posterior calibration checks using the `sbi` library (maybe add this as an additional package to avoid torch dependency?)
- [x] Implement get sampler for every pipeline
- [ ] Diffusion models are underconfident, the EDM sde works well while the VE and VP legacy sdes are not working properly yet
- [ ] Include example for batched sampling in the first tutorial 
- [ ] Include SBC checks in the benchmark notebooks and training script
- [x] Fix the GW example
- [ ] Add tests for the examples 
- [ ] Deploy everything to PyPI 
- [ ] Figure out what is the best way to include the GenSBI dependency into the sub packages without causing circular dependencieds
- [x] Currently Flux1 is optimized for 1D data, we need to generalize it for 2D data, 
- [ ] ... and spherical data as well
- [ ] Retrain the benchmark models using the latest GenSBI version, especially the getting started example
- [ ] Implement contour levels like the ones from corner in the sns plot too
- [x] Write in the documentation some info concernign the ID embedding, and what to use/when 
- [x] Implement RoPE embedding with the 0-out trick
- [ ] For the Flux1Joint, add the possibility to concatenate the embedding ids, instead of just summing them
- [ ] The theoretical overview is very basic currently, we will need to expand it
- [ ] Update two moons example with full diagnostics, 
- [ ] Make notebooks for all the examples with link to colab and github
- [ ] Include plots from the img directory (so users have a reference plot even when they do their own experiments with the notebook)
- [ ] Update lensing and GW notebooks to work on colab, and use relative paths (no hardcoded paths), assuming the notebook is being run from its own example directory
- [ ] make examples from the Flux1 Flux1Joint and Simformer pipelines
- [ ] rerun the examples from the example directory in gensbi, not only gensbi-examples