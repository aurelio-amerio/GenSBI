# Porting TarFlow → gensbi (NLE)

**Purpose.** Starting point for a design spec. Goal: a conditional normalizing flow `q(x | θ)` with *exact, fast density evaluation* for Neural Likelihood Estimation (NLE), trained end-to-end, scaling better than MAF+MADE. Reference: TarFlow (`apple/ml-tarflow`, `transformer_flow.py`, ~290 LOC) and paper *Normalizing Flows are Capable Generative Models* (arXiv:2412.06329).

---

## Why TarFlow at all (the one-line case)

- It's a true NF → closed-form `log q(x)`; no ODE solve / trace estimation (unlike our flow-matching & diffusion models).
- Density eval is the **parallel, single-pass** direction; only *sampling* is the slow sequential loop. In NLE we never sample the flow — MCMC samples θ — so TarFlow's main weakness doesn't apply to us.
- Same MLE objective and density semantics as our existing MAF → low-risk drop-in for the NLE estimator role.

---

## Port plan (Keep / Adapt / Drop / Add)

| Component (in `transformer_flow.py`) | Action | Why |
|---|---|---|
| `MetaBlock` autoregressive affine transform + causal mask (`torch.tril`) | **Keep** | This *is* the flow: tractable triangular Jacobian, exact log-det. Core value. |
| Stacked blocks with alternating order (`PermutationIdentity` / `PermutationFlip`) | **Keep, generalize** | Alternating AR direction restores expressivity. For vector data, use **random permutations** per block, not just flip. |
| `nvp` (affine / non-volume-preserving) mode → `-xa` log-det term | **Keep, default ON** | Gives the proper tractable Jacobian we need for exact likelihood. |
| `Model.forward` returning `(z, logdets)` + Gaussian-base loss | **Keep, re-expose** | Already computes NLL. Expose a clean per-example `log_prob(x, θ)` = `log N(z;0,var) + logdet` → this is the MCMC target. |
| `patchify` / `unpatchify` (image `unfold`/`fold`) | **Adapt** | Replace with a tokenizer over the data vector `x`. 1 scalar/token at low D; chunk into blocks at high D to cap sequence length (attention is O(T²)). |
| `class_embed` (categorical conditioning) | **Adapt → continuous** | NLE needs `q(x \| θ)` with continuous θ. Swap the lookup for an MLP embedding of θ added to tokens (or prepended as prefix token[s]). *Main new piece.* |
| KV-cache `reverse` / sampling path | **Drop (or defer)** | We evaluate density, not sample the flow. Keep only if we later want prior/posterior predictive sampling. |
| Noise augmentation (`noise_std`) + uniform dequantization | **Drop / handle with care** | These improve *sample* quality but bias the *density*. See gotcha below. |
| Guidance / CFG (`guidance`, `attn_temp`, annealed) | **Drop** | Sampling-quality machinery, irrelevant to likelihood eval. |
| Image plumbing (`img_size`, channels, FID/BPD scripts) | **Drop** | Not our data type. |
| Deep-shallow capacity split (from STARFlow) | **Add later** | Cheap scalability win; borrow once the base works. Not in scope v1. |

---

## Behavioral changes to nail down

- **Conditioning path.** How θ enters (per-token add vs. prefix tokens vs. FiLM). Affects how high-dim θ scales — θ only touches the embedding, not the flow width.
- **Density direction.** Confirm forward = data→z = parallel eval; lock the `log_prob` API and its shape contract for MCMC.
- **Tokenization scheme.** Scalar-per-token vs. block-per-token; how block size trades expressivity vs. sequence length / attention cost.

---

## Gotchas / risks

- **Noise augmentation is the big one.** Training on noised data ⇒ we'd learn the density of noised x ⇒ biased, over-smoothed likelihood ⇒ biased posteriors. Our simulator outputs are already continuous, so default to **no** augmentation; if any is used, document its effect on the learned density.
- **Variable ordering.** AR flows are order-sensitive; rely on per-block permutations to mitigate. Decide fixed-random vs. learned.
- **Attention cost in data dimension.** O(T²) — chunk dimensions into tokens for high-D x.
- **License.** Apple repo header is "All Rights Reserved" + custom LICENSE. Cleanest path: **reimplement** (~290 LOC) under gensbi's license and cite the paper, rather than vendor the code.

---

## Deliberately out of scope

- **STARFlow's latent-space modeling** — requires a pretrained autoencoder; violates our end-to-end constraint. (Its deep-shallow trick is separable and welcome.)
- **FARMER** — adds a separate AR-GMM head, channel splitting, and distillation tuned for image-pixel redundancy; heavier, less clean likelihood, no clean public code. Wrong fit.

---

## Open questions for the brainstorm

1. Token granularity vs. expected x-dimensionality across our target benchmarks?
2. θ-conditioning mechanism — and does it need to handle variable-dim θ?
3. Do we ever need flow sampling (posterior predictive), i.e. keep the reverse path or not?
4. Default block/layer counts and a deep-shallow option from day one?
5. Validation: how do we sanity-check calibration of `log q(x|θ)` (e.g., SBC) given no noise augmentation?
