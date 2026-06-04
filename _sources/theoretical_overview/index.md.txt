# Theoretical Overview

Overview of the generative modeling frameworks implemented in GenSBI, based on the unified theory of score-based diffusion models ([Song et al., 2021](https://arxiv.org/abs/2011.13456)), the EDM design space ([Karras et al., 2022](https://arxiv.org/abs/2206.00364)), and conditional flow matching ([Lipman et al., 2024](https://arxiv.org/abs/2412.06264)).

A summary of the methodologies used in this work is also present in the companion paper for this library in [Section 3](https://arxiv.org/abs/2605.27499). 

These pages provide a self-contained introduction to each framework, progressing from the foundational SDE-based score matching to the streamlined EDM formulation, then to flow matching, and finally to the conditional extensions that enable simulation-based inference.

```{toctree}
:maxdepth: 1

score_matching.md
diffusion.md
flow.md
conditional.md

```
