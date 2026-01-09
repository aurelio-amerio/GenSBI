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
- [ ] Provide SOTA pre-trained models and checkpoints for some SBI benchmark cases
- [x] Implement VAE training pipeline
- [x] Implement wrapper to run posterior calibration checks using the `sbi` library (maybe add this as an additional package to avoid torch dependency?)
- [x] implement get sampler for every pipeline
- [ ] Diffusion models are underconfident, the EDM sde works while the VE and VP legacy sdes are not working properly yet
- [ ] Include example for batched sampling in the first tutorial 
- [ ] Include SBC checks in the benchmark notebooks and training script
- [ ] Fix the GW example
- [ ] Add tests for the examples and validation library
- [ ] Deploy everything to PyPI 
- [ ] Figure out what is the best way to include the GenSBI dependency into the sub packages without causing circular dependencieds
- [ ] Currently Flux1 is optimized for 1D data, we need to generalize it for 2D data, and spherical data as well

- [ ] Retrain the benchmark models using the latest GenSBI version, especially the getting started example