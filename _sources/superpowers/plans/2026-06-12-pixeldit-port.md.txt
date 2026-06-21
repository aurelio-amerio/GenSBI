# PixelDiT Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Faithful channel-last JAX port of PixelDiT (arXiv:2511.20645) into `src/gensbi/experimental/models/pixeldit/`, trained with existing CondOT flow matching and sampled with the existing ODE pipeline, verified by FieldDiT-style learning gates.

**Architecture:** Dual-level DiT — patch-level MMDiT pathway (obs patch tokens + cond tokens, joint attention, per-stream RoPE) producing semantic tokens `s_cond`, then a pixel-level pathway (M PiT blocks: pixel-wise AdaLN from `s_cond` + compress→global-attend→expand token compaction) ending in a zero-init per-pixel FinalLayer. Spec: `docs/superpowers/specs/2026-06-12-pixeldit-port-design.md`. Reference code: `reference/PixelDiT/pixdit_core/` (PyTorch — port, never import).

**Tech Stack:** JAX, flax.nnx, einops; reuse `gensbi.models.flux1.math` (`attention`, `apply_rope`), `gensbi.models.embedding.FeatureEmbedder`, `FieldConditionalPipeline`.

**Per-task model assignment (user requirement):** each task names the implementer model. Default **sonnet**; numerically subtle / silent-failure tasks get **opus-4.8 (high effort)**; the two hardest (PiT block, torch parity) get **fable-5 (medium effort)**. When dispatching with the Agent tool, pass `model: "sonnet" | "opus" | "fable"` accordingly.

**Conventions for all tasks:**
- Run tests with `JAX_PLATFORMS=cpu python -m pytest <path> -v` (pyproject already forces xdist `-n 2`; use `-n 0` if you need `-s` prints).
- All modules: channel-last `(B, H, W, C)`; `param_dtype` arg defaulting to `jnp.bfloat16`; constructors take `rngs: nnx.Rngs` (follow `src/gensbi/experimental/models/fielddit/` style).
- Reference files are at `reference/PixelDiT/pixdit_core/{modules.py,pixeldit_c2i.py,pixeldit_t2i.py}` — read the relevant lines before porting; line refs below are to those files.
- Commit after each green task: `git add <files> && git commit -m "<type>(pixeldit): <what>"`.

---

### Task 1: `rope.py` — fractional 2D RoPE + 1D cond RoPE in flux rotation format

**Model: opus-4.8 (high)** — small file, but a silent numerical error here corrupts everything downstream and the parity script depends on it.

**Files:**
- Create: `src/gensbi/experimental/models/pixeldit/__init__.py` (empty for now)
- Create: `src/gensbi/experimental/models/pixeldit/rope.py`
- Test: `tests/experimental/models/pixeldit/test_rope.py` (+ empty `tests/experimental/models/pixeldit/__init__.py` if the suite needs it — check `tests/experimental/models/fielddit/`, which has none)

**Contract.** Port `precompute_freqs_cis_2d` (`modules.py:132-145`) and `fetch_pos_text` (`pixeldit_t2i.py:232-241`), but emit the real-rotation tensor consumed by `gensbi.models.flux1.math.apply_rope` instead of complex `cis`:

```python
def precompute_freqs_cis_2d(head_dim, height, width, theta=10000.0, scale=16.0):
    """2D axial rope, fractional positions on [0, scale]. Returns (1, 1, H*W, head_dim//2, 2, 2) float32."""
    x_pos = jnp.linspace(0, scale, width)
    y_pos = jnp.linspace(0, scale, height)
    yy, xx = jnp.meshgrid(y_pos, x_pos, indexing="ij")          # row-major grid, y first
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 4)[: head_dim // 4] / head_dim))  # (head_dim/4,)
    x_ang = jnp.outer(xx.reshape(-1), freqs)                    # (N, head_dim/4)
    y_ang = jnp.outer(yy.reshape(-1), freqs)
    # reference interleaves x/y cis pairwise: cat([x_cis[...,None], y_cis[...,None]], -1).reshape(N, -1)
    angles = jnp.stack([x_ang, y_ang], axis=-1).reshape(x_ang.shape[0], -1)  # (N, head_dim/2)
    rot = jnp.stack(
        [jnp.cos(angles), -jnp.sin(angles), jnp.sin(angles), jnp.cos(angles)], axis=-1
    ).reshape(*angles.shape, 2, 2)
    return rot[None, None, ...].astype(jnp.float32)             # (1, 1, N, head_dim/2, 2, 2)

def precompute_freqs_cis_1d(head_dim, length, theta=10000.0):
    """1D integer-position rope for cond tokens (reference fetch_pos_text). Same output layout, N=length."""
```

For `_1d`: `freqs = 1/(theta ** (arange(0, head_dim, 2) / head_dim))` (note step 2, `head_dim/2` freqs — the full head dim is rotated by the single axis), `angles = outer(arange(length), freqs)`.

Why this layout is exactly equivalent to the reference: their `apply_rotary_emb` (`modules.py:165-175`) multiplies *consecutive value pairs* `(x[2j], x[2j+1])` by complex `freqs_cis[n, j]`; our `apply_rope` (`flux1/math.py:77-100`) applies a 2×2 rotation to the same consecutive pairs. Equality test below is the proof.

- [x] **Step 1: write failing tests** — in `test_rope.py`:
  - `test_shape_and_dtype`: `precompute_freqs_cis_2d(64, 4, 6)` → shape `(1, 1, 24, 32, 2, 2)`, float32; `_1d(64, 5)` → `(1, 1, 5, 32, 2, 2)`.
  - `test_rotation_orthogonal`: each `(2,2)` block `R` satisfies `R @ R.T ≈ I` (det 1).
  - `test_matches_complex_reference`: reimplement the reference math *with numpy complex* inline in the test (10 lines: `cis = exp(1j*angles)` with the same interleave) and check that rotating a random `q` via `flux1.math.apply_rope` with our table equals complex multiplication of consecutive pairs, `atol=1e-5`. This pins the pair convention.
  - `test_native_grid_spacing`: at `width=height=16, scale=16`, positions spacing is `16/15` (document the near-integer native grid).
- [x] **Step 2: run, verify they fail** (`ModuleNotFoundError`/`AttributeError`).
- [x] **Step 3: implement `rope.py`** as specified above.
- [x] **Step 4: run tests, verify green.**
- [x] **Step 5: commit** — `feat(pixeldit): fractional 2D/1D rope tables in flux rotation format`.

---

### Task 2: `modules.py` — SwiGLU, pixel MLP, TimestepConditioner, FinalLayer, pixel sincos table

**Model: sonnet**

**Files:**
- Create: `src/gensbi/experimental/models/pixeldit/modules.py`
- Test: `tests/experimental/models/pixeldit/test_modules.py`

Port from `reference/PixelDiT/pixdit_core/modules.py`:

- `SwiGLU(dim, mlp_ratio=4.0)` (ref `FeedForward`, lines 119-129): `hidden = int(2 * (dim * mlp_ratio) / 3)`; `w1, w3: Linear(dim→hidden, use_bias=False)`, `w2: Linear(hidden→dim, use_bias=False)`; `w2(silu(w1(x)) * w3(x))`.
- `PixelMLP(dim, mlp_ratio=4.0)` (ref `MLP`, lines 223-238): `Linear(dim→4·dim)` → GELU → `Linear(→dim)`, biases on, no dropout (we never train with dropout).
- `TimestepConditioner(hidden_size, freq_dim=256)` (ref lines 63-91): sinusoid with **`max_period=10` and NO ×1000 time factor** (deliberate deviation from `flux1.timestep_embedding` — do not reuse it); order `cat([cos, sin])`; compute the sinusoid in float32 and cast to `param_dtype` only before the MLP (FieldDiT lesson: bf16 `t` quantizes). MLP = `Linear(256→D)` → SiLU → `Linear(D→D)`, kernels init `normal(0.02)` (ref `initialize_weights`).
- `FinalLayer(hidden_size, out_channels)` (ref lines 241-250): `nnx.RMSNorm(hidden_size, epsilon=1e-6)` (learnable scale, default) + `Linear(hidden→out, use_bias=True)` with **kernel and bias zero-init**.
- `get_2d_sincos_pos_embed(embed_dim, h, w)` (ref lines 10-56): pure-numpy port returning `(h*w, embed_dim)` float32; supports `h != w` (ref `PixelTokenEmbedder._fetch_pixel_pos_image` non-square branch, `pixeldit_c2i.py:84-89`). Keep the reference's axis convention exactly: `emb_h` from `grid[0]` where `grid = meshgrid(grid_w, grid_h)` — copy, don't "fix".
- `class Buffer(nnx.Variable): ...` — the float analog of fielddit's `RopeIds` (`fielddit/model.py:28-34`): non-trainable array buffers (rope tables, sincos tables) filterable out of `nnx.Param` state. Tasks 3 and 6 import it from here.

- [x] **Step 1: failing tests** — shapes; SwiGLU hidden width = `int(2*4*dim/3)`; `FinalLayer` output exactly zero for random input at init; `TimestepConditioner` output finite for `t = jnp.linspace(0, 1, 8)` in bf16 params and **differs between t=0.0 and t=0.001** (the max_period=10 sinusoid must resolve small t in f32); sincos table matches the reference numpy code (inline 15-line numpy copy in the test) to `atol=1e-6`.
- [x] **Step 2: run, fail.**
- [x] **Step 3: implement.**
- [x] **Step 4: run, pass.**
- [x] **Step 5: commit** — `feat(pixeldit): faithful low-level modules (SwiGLU, t-conditioner, FinalLayer, sincos)`.

---

### Task 3: `embedders.py` — patch / pixel / cond token embedders

**Model: sonnet** (the grouping reshape is fully specified below — follow it exactly)

**Files:**
- Create: `src/gensbi/experimental/models/pixeldit/embedders.py`
- Test: `tests/experimental/models/pixeldit/test_embedders.py`

- `PatchTokenEmbedder(in_features, hidden_size)` (ref `pixeldit_c2i.py:21-38`): `Linear(in→hidden, bias=True)`, kernel xavier_uniform, bias zeros (ref init). No norm (norm_layer=None case).
- `PixelTokenEmbedder(in_channels, pixel_hidden_size, field_shape, patch_size, use_abs_pos=True)` (ref `pixeldit_c2i.py:60-111`): per-pixel `Linear(C→D_pix)`; add the Task-2 sincos table (precomputed at init for the fixed `field_shape`, stored in the `Buffer` Variable from Task 2's `modules.py`); then group channel-last:

```python
# x: (B, H, W, D_pix); Hs, Ws = H // p, W // p
x = x.reshape(B, Hs, p, Ws, p, D_pix)
x = x.transpose(0, 1, 3, 2, 4, 5)          # (B, Hs, Ws, p, p, D_pix)
x = x.reshape(B * Hs * Ws, p * p, D_pix)   # (B·L, p², D_pix)
```

  Also export module-level helpers `patchify(x, p) -> (B, L, p²·C)` and `unpatchify(tokens, grid, p, C) -> (B, H, W, C)` using the *same* `(B, Hs, p, Ws, p, C) ↔ transpose` ordering, so patch tokens, pixel groups, and the output fold all index patches row-major `(Hs, Ws)` and pixels row-major `(p, p)`. `unpatchify` is the exact inverse: `(B·L, p², C) → (B, Hs, Ws, p, p, C) → transpose(0,1,3,2,4,5) → (B, H, W, C)`.
- `CondTokenEmbedder(cond_in_channels, hidden_size, n_tokens, id_embedding="absolute")` (ref `y_embedder` + `y_pos_embedding`, `pixeldit_t2i.py:179-180,267-268`): `Linear(D_c→D, bias=True)` → `nnx.RMSNorm(D, epsilon=1e-6)` → add id embedding. `id_embedding ∈ {"absolute", "pos1d", "none"}` via `gensbi.models.embedding.FeatureEmbedder(num_embeddings=n_tokens, hidden_size=D, kind=...)` called on `jnp.arange(n_tokens)` (skip when `"none"`). Accept `(B, K)` input by expanding to `(B, K, 1)` iff `cond_in_channels == 1`, else raise (copy the guard wording from `fielddit/cond.py:36-42`). **No `sqrt(hidden)` token scaling** (that was FieldDiT-specific; the reference adds the pos embedding directly).

- [x] **Step 1: failing tests** — `patchify`/`unpatchify` round-trip exactly on random `(2, 8, 12, 3)` with `p=2` and `p=4`; pixel grouping: pixel `(i, j)` of patch `(a, b)` in the grouped tensor equals input pixel `(a*p+i, b*p+j)` (index-level assertion, not just shape); cond embedder shape `(B, K, D)`, `(B,K)`→`(B,K,1)` path, `"none"` vs `"absolute"` differ, `cond_in_channels=2` with `(B,K)` raises.
- [x] **Step 2: run, fail.** **Step 3: implement.** **Step 4: run, pass.**
- [x] **Step 5: commit** — `feat(pixeldit): patch/pixel/cond token embedders (channel-last)`.

---

### Task 4: `blocks.py::MMDiTBlock` — joint-attention patch block with per-stream rope

**Model: opus-4.8 (high)** — deviates from the library's `attention(pe=...)` pattern: rope must be applied per stream *before* the joint concat.

**Files:**
- Create: `src/gensbi/experimental/models/pixeldit/blocks.py`
- Test: `tests/experimental/models/pixeldit/test_blocks.py`

Faithful port of `MMDiTBlockT2I` + `MMDiTJointAttention` (`pixeldit_t2i.py:19-132`), one class:

```python
class MMDiTBlock(nnx.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, *, zero_init=True, rngs, param_dtype): ...
    def __call__(self, x, y, c, pe_x, pe_y=None):  # x=(B,Lx,D) obs, y=(B,Ly,D) cond, c=(B,1,D)
```

Per stream `s ∈ {x, y}`: `norm_{s}1 = nnx.RMSNorm(D, eps=1e-6)`, `qkv_{s} = Linear(D→3D, no bias)`, per-head q/k RMSNorm over `head_dim` (reuse `gensbi.models.flux1.layers.QKNorm`), `proj_{s} = Linear(D→D, bias)`, `norm_{s}2`, `mlp_{s} = SwiGLU(D, mlp_ratio)`, `adaLN_{s} = Linear(D→6D, bias)` — **plain linear, NO internal silu** (`c` arrives pre-activated; do NOT reuse flux `Modulation`, which applies silu inside), zero-init kernel+bias when `zero_init=True` else default init (`zero_init_blocks` flag plumbing, c2i vs t2i recipe).

Forward (ref lines 120-132 exactly): chunk `adaLN_x(c)` and `adaLN_y(c)` into 6 `(B,1,D)` pieces; modulate `norm1` outputs with `x*(1+scale)+shift`; compute per-stream q/k/v via `rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=num_heads)`; QKNorm; **apply rope per stream**: `q_x, k_x = apply_rope(q_x, k_x, pe_x)`; if `pe_y is not None`, same for the y stream (else leave unrotated); concat `[y, x]` along the token axis (cond first — ref order); `attention(q, k, v, pe=None)` from `flux1.math`; split outputs at `Ly`; per-stream `proj` + gated residual; per-stream SwiGLU branch with the second modulation triple. Returns `(x, y)`. No mask support (fixed-K cond — YAGNI, note in docstring).

- [x] **Step 1: failing tests** — output shapes preserved for `(2, 12, 64)` obs + `(2, 3, 64)` cond, `heads=4`; with `zero_init=True` the block is an **exact identity** on both streams (`jnp.array_equal`); with `zero_init=False` it is not; with `pe_y=None` vs a 1D pe the outputs differ (rope reaches cond); gradient flows to `qkv_x` kernel when `zero_init=False` (one `jax.grad` of a scalar loss).
- [x] **Step 2: run, fail.** **Step 3: implement.** **Step 4: run, pass.**
- [x] **Step 5: commit** — `feat(pixeldit): MMDiT patch block with per-stream rope`.

---

### Task 5: `blocks.py::PiTBlock` — pixel-wise AdaLN + token compaction

**Model: fable-5 (medium)** — the paper's core novelty; per-pixel modulation layout + the `(B·L, p², D_pix) ↔ (B, L, attn)` compress/attend/expand round-trip are the most error-prone reshapes in the port, and both modulation variants must be right.

**Files:**
- Modify: `src/gensbi/experimental/models/pixeldit/blocks.py`
- Test: `tests/experimental/models/pixeldit/test_blocks.py` (extend)

Faithful port of `PiTBlock` (`pixeldit_c2i.py:114-187`):

```python
class PiTBlock(nnx.Module):
    def __init__(self, pixel_dim, context_dim, patch_size, attn_dim, num_heads,
                 mlp_ratio=4.0, post_modulation=False, *, zero_init=True, rngs, param_dtype): ...
    def __call__(self, x, s_cond, pe, batch):  # x=(B·L, p², D_pix), s_cond=(B·L, D), pe=2D rope over the patch grid
```

Components: `norm1, norm2 = nnx.RMSNorm(pixel_dim, eps=1e-6)`; `compress = Linear(p²·pixel_dim → attn_dim, bias)`; `expand = Linear(attn_dim → p²·pixel_dim, bias)`; attention = reuse `gensbi.models.flux1.layers.SelfAttention(dim=attn_dim, num_heads, qkv_features=attn_dim, qkv_bias=False)` (matches ref `RotaryAttention`: no-bias qkv, per-head RMSNorm q/k, biased proj) called as `self.attn(x_comp, pe=pe)`; `mlp = PixelMLP(pixel_dim, mlp_ratio)`; `adaLN = Linear(context_dim → n_mod·pixel_dim·p², bias)`, zero-init per flag, `n_mod = 4 if post_modulation else 6`.

Forward — follow ref lines 157-187 statement by statement:
1. `cond_params = adaLN(s_cond).reshape(BL, p², n_mod·pixel_dim)`, then `jnp.split(..., n_mod, axis=-1)` — **the reshape groups modulation per pixel; order of chunks: pre-variant `(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)`; post-variant `(scale1, shift1, scale2, shift2)` — note scale-before-shift in the post variant** (ref line 168; easy to flip silently).
2. pre: `x_norm = (1 + scale_msa) * norm1(x) + shift_msa`; post: `x_norm = norm1(x)` (un-modulated).
3. compaction: `x_comp = compress(x_norm.reshape(BL, p²·pixel_dim)).reshape(batch, L, attn_dim)` where `L = BL // batch`; `attn_out = self.attn(x_comp, pe=pe)`; `attn_exp = expand(attn_out.reshape(BL, attn_dim)).reshape(BL, p², pixel_dim)`.
4. pre: `x = x + gate_msa * attn_exp`, then `x = x + gate_mlp * mlp((1 + scale_mlp) * norm2(x) + shift_mlp)`.
   post: `x = x + attn_exp * (1 + scale1) + shift1`, then `x = x + mlp(norm2(x)) * (1 + scale2) + shift2`.

- [x] **Step 1: failing tests** — shape preservation for `B=2, grid 3×4 (L=12), p=2, D_pix=8, D=64, attn=64, heads=4`; **pre-variant with `zero_init=True` is exact identity** (all six chunks zero ⇒ gates closed); **post-variant with `zero_init=True` is NOT identity** (`x + attn_exp·1 + 0` — the gate-free post form; assert it differs, this documents the variant semantics); cross-patch mixing: zero out `zero_init`, make one patch's pixels distinctive, assert another patch's output changes (compaction attention is global); `s_cond` sensitivity: two different `s_cond` give different outputs when `zero_init=False`.
- [x] **Step 2: run, fail.** **Step 3: implement.** **Step 4: run, pass.**
- [x] **Step 5: commit** — `feat(pixeldit): PiT block (pixel-wise AdaLN + token compaction)`.

---

### Task 6: `model.py` — `PixelDiTParams` + `PixelDiT` assembly

**Model: opus-4.8 (high)** — wiring has many silent-failure spots: `s_cond` construction, `c = silu(t_emb)` placement, init policy, buffer handling, patch/pixel index alignment.

**Files:**
- Create: `src/gensbi/experimental/models/pixeldit/model.py`
- Test: `tests/experimental/models/pixeldit/test_model.py`

`PixelDiTParams` dataclass (mirror `FieldDiTParams` style, `fielddit/model.py:37-111`, including the live-`rngs` reproducibility docstring note):

```python
in_channels: int; field_shape: Tuple[int, int]; cond_dim: int; rngs: nnx.Rngs
cond_in_channels: int = 1
patch_size: int = 4
hidden_size: int = 384
pixel_hidden_size: int = 16
patch_depth: int = 6
pixel_depth: int = 2
num_heads: int = 6
pixel_attn_hidden_size: Optional[int] = None   # None -> hidden_size
pixel_num_heads: Optional[int] = None          # None -> num_heads
mlp_ratio: float = 4.0
cond_id_embedding: str = "absolute"            # {"absolute", "pos1d", "none"}
use_cond_rope: bool = True                     # reference faithful default; False for unordered theta
use_pixel_abs_pos: bool = True
pit_post_modulation: bool = False
zero_init_blocks: bool = True                  # c2i recipe; False = t2i recipe (final layer still zero)
rope_scale: float = 16.0
theta: float = 10_000.0
param_dtype: DTypeLike = jnp.bfloat16
```

`__post_init__`: assert `H % p == 0`, `W % p == 0`, `hidden_size % num_heads == 0`, resolved `pixel_attn_hidden_size % pixel_num_heads == 0`, both head dims `% 4 == 0` (2D rope needs dim/4 freqs); derive `token_grid = (H//p, W//p)`, `n_obs_tokens`.

`PixelDiT.__init__`: store static primitives only (NOT the params dataclass — GraphDef lesson, `fielddit/model.py:162-171`). Submodules: `PatchTokenEmbedder(in_channels·p², D)`, `PixelTokenEmbedder`, `CondTokenEmbedder`, `TimestepConditioner(D)`, `patch_blocks = [MMDiTBlock(...) for _ in range(N)]`, `pixel_blocks = [PiTBlock(...) for _ in range(M)]`, `FinalLayer(D_pix, in_channels)`. Rope tables built once and stored in the `Buffer` Variable from `modules.py` (Task 2):
- `pe_patch = precompute_freqs_cis_2d(D // num_heads, Hs, Ws, theta, rope_scale)`
- `pe_pit = precompute_freqs_cis_2d(attn_dim // pixel_num_heads, Hs, Ws, theta, rope_scale)`
- `pe_cond = precompute_freqs_cis_1d(D // num_heads, cond_dim, theta)` if `use_cond_rope` else `None`

`__call__(self, t, obs, cond, obs_ids=None, cond_ids=None, conditioned=True, guidance=None)`:
- guards (copy wording style from `fielddit/model.py:189-210`): `conditioned is not True` → `NotImplementedError`; `guidance is not None` → `ValueError` ("PixelDiT has no guidance embedding"); rank-4 obs, spatial == `field_shape`, channels == `in_channels`; cast obs/cond to `param_dtype`, `t` to f32.
- forward (faithful to `pixeldit_t2i.py:252-316`, minus repa/mask/`s` caching — YAGNI):

```python
t_emb = self.t_conditioner(t)[:, None, :]                  # (B, 1, D)
cond_tokens = self.cond_embedder(cond)                     # (B, K, D)
c = nnx.silu(t_emb)                                        # (B, 1, D) — t2i: cond enters via tokens only
s = self.s_embedder(patchify(obs, p))                      # (B, L, D)
for blk in self.patch_blocks:
    s, cond_tokens = blk(s, cond_tokens, c, self.pe_patch[...],
                         self.pe_cond[...] if use_cond_rope else None)
s = nnx.silu(t_emb + s)                                    # (B, L, D)
s_cond = s.reshape(B * L, D)
x_pix = self.pixel_embedder(obs)                           # (B·L, p², D_pix)
for blk in self.pixel_blocks:
    x_pix = blk(x_pix, s_cond, self.pe_pit[...], batch=B)
x_pix = self.final_layer(x_pix)                            # (B·L, p², C)
return unpatchify(x_pix, self.token_grid, p, C)            # (B, H, W, C)
```

Init policy (c2i recipe, ref `pixeldit_c2i.py:252-266`): `s_embedder` xavier/zero-bias (Task 3 already), `t_conditioner` normal-0.02 (Task 2 already), `final_layer` zeros (Task 2 already), all block adaLN zeros via `zero_init_blocks` flag passed down.

- [x] **Step 1: failing tests** (model the suite on `fielddit/test_model.py`):
  - default-config forward `(t=(B,), obs=(B,16,16,1), cond=(B,2,1))` with a small config (`D=64, heads=4, N=2, M=2, p=4, D_pix=8`) → output `(B,16,16,1)`, finite, **exactly zero at init** (zero adaLN + zero final layer).
  - `zero_init_blocks=False` → output still exactly zero (final layer alone gates), but internal `s` differs from `s_embedder` output (probe via a second forward after perturbing one MMDiT qkv kernel — or simpler: assert some block adaLN kernel is non-zero in state).
  - guards: rank-3 obs raises; wrong spatial raises; `conditioned=False` raises `NotImplementedError`; `guidance=1.0` raises `ValueError`; `(B, K)` cond accepted when `cond_in_channels=1`.
  - `obs_ids`/`cond_ids` passed as garbage are ignored (same output).
  - differentiable: `jax.grad` of `sum(model(t, obs, cond)**2)` w.r.t. obs runs (will be zero at init — just assert no error and correct shape).
  - `use_cond_rope=False` and `cond_id_embedding="none"` configs construct and run; `cond_id_embedding="pos1d"` constructs and runs.
  - buffers excluded from `nnx.Param` state: `nnx.state(model, nnx.Param)` contains no `(..., 2, 2)` rope tables.
- [x] **Step 2: run, fail.** **Step 3: implement.** **Step 4: run, pass.**
- [x] **Step 5: commit** — `feat(pixeldit): PixelDiT model assembly + params`.

---

### Task 7: package exports

**Model: sonnet**

**Files:**
- Modify: `src/gensbi/experimental/models/pixeldit/__init__.py`
- Modify: `src/gensbi/experimental/models/__init__.py` (follow how `FieldDiT` is exported)
- Test: extend `tests/experimental/models/pixeldit/test_model.py`

- [x] **Step 1: failing test** — `from gensbi.experimental.models import PixelDiT, PixelDiTParams`.
- [x] **Step 2: run, fail. Step 3: add exports (`PixelDiT`, `PixelDiTParams`, `MMDiTBlock`, `PiTBlock` from the package `__init__`). Step 4: run, pass.**
- [x] **Step 5: run the FULL suite** (`JAX_PLATFORMS=cpu python -m pytest tests/`) — no regressions. **Commit** — `feat(pixeldit): public exports`.

---

### Task 8: pipeline integration test — construct / train-step / sample

**Model: sonnet** (pattern exists: `tests/experimental/recipes/test_field_pipeline.py`)

**Files:**
- Test: `tests/experimental/models/pixeldit/test_pipeline.py`

No production code expected: `FieldConditionalPipeline` takes the model as-is (`obs_ids`/`cond_ids` are `None` in extras and PixelDiT ignores them). Copy the loader/fixture pattern from `test_field_pipeline.py` (tiny in-memory `(obs (B,16,16,1), cond (B,2,1))` dataset, tiny model config, `FlowMatchingMethod`).

- [x] **Step 1: write tests**
  - pipeline constructs; `get_loss_fn()` returns finite scalar loss on one batch.
  - 3 optimizer steps run (loss finite throughout).
  - `pipeline._wrap_model(); pipeline.sample(key, cond[(1,2,1)], nsamples=2, step_size=0.5)` → `(2, 16, 16, 1)`, finite.
- [x] **Step 2-4: run; if anything in the pipeline/wrapper genuinely needs a fix, STOP and report back to the orchestrator instead of patching the pipeline ad hoc** (spec says no pipeline changes expected — a needed change means a contract mismatch worth human eyes).
- [x] **Step 5: commit** — `test(pixeldit): field pipeline construct/train/sample integration`.

---

### Task 9: Gate 1 — gradient aliveness for every subtree

**Model: sonnet** (the ignition cascade is fully specified below)

**Files:**
- Test: `tests/experimental/models/pixeldit/test_training.py` (model on `fielddit/test_training.py`)

**Ignition cascade (why N steps, documented for the test):** with the c2i zero-init, at step 1 only `final_layer` has nonzero grads (zero final kernel blocks upstream flow); after step 1 updates it, step 2 ignites the pixel pathway (pixel embedder, PiT compress/expand/attn/mlp, PiT adaLN); after PiT adaLN becomes nonzero, step 3 ignites `s_cond` consumers (patch blocks' contribution, `t_conditioner`, `s_embedder`); patch-block adaLN/gates then open, so step 4 ignites the cond pathway (cond tokens only influence `s_N` through gated joint attention). **Train 6 steps** (margin) with a real optimizer (`optax.adam(1e-2)`, the FieldDiT gate pattern) on a fixed tiny batch, then assert: every `nnx.Param` leaf in every subtree (`s_embedder`, `cond_embedder`, `t_conditioner`, `patch_blocks`, `pixel_blocks`, `final_layer`, `pixel_embedder`) **changed from its initial value** (compare snapshots, not grads — robust to the cascade). Also assert loss is finite at every step.

- [x] **Step 1: write the test** (one test, parametrized over the two `pit_post_modulation` variants).
- [x] **Step 2: run — if a subtree never moves in 6 steps, that is a real wiring bug: debug the model (likely a detached buffer or a `lax.stop_gradient`-like cast), do NOT just raise the step count without understanding why.**
- [x] **Step 3: green. Commit** — `test(pixeldit): gate 1 — every subtree trains within the ignition cascade`.

---

### Task 10: Gate 2 — tiny overfit + cond sensitivity

**Model: sonnet**

**Files:**
- Test: extend `tests/experimental/models/pixeldit/test_training.py`

- [x] **Step 1: write the test** (FieldDiT gate-2 pattern, `fielddit/test_training.py`): fixed dataset of 4 `(obs, cond)` pairs where obs is a deterministic function of cond (e.g. obs = checkerboard × cond[0] + gradient × cond[1], 16×16); train ~300 steps full-batch through the pipeline loss; assert (a) final loss < initial loss by ≥ 2 orders of magnitude, (b) **cond sensitivity with structurally different conds** — sample or forward with `cond_A = [[1],[0]]` vs `cond_B = [[0],[1]]` and assert mean-squared output difference is significant (uniform-shift probes are annihilated by the cond embedder's RMSNorm — FieldDiT lesson, do not use `cond + const`).
- [x] **Step 2: run (CPU, keep config tiny: D=64, N=2, M=1, p=4, 16×16 field — aim < 90 s).**
- [x] **Step 3: green. Commit** — `test(pixeldit): gate 2 — tiny overfit proves live conditioning`.

---

### Task 11: Gate 3 — opt-in realistic smoke

**Model: sonnet**

**Files:**
- Test: extend `tests/experimental/models/pixeldit/test_training.py`

- [x] **Step 1: write the opt-in test** (skip-by-default pattern from the fielddit 256² smoke — find it via `grep -rn "smoke" tests/experimental/models/fielddit/`): paper-B-ish config scaled to a field workload — `field_shape=(64,64), p=8, D=768, N=12, M=2, D_pix=16, heads=12, K=2` — one forward + one loss/grad step; print param count, peak shapes, walltime with the `[smoke]` prefix (remember: prints only visible with `-n 0 -s`).
- [x] **Step 2: run it once locally (opt-in env var), confirm it passes; it stays skipped in CI.**
- [x] **Step 3: commit** — `test(pixeldit): gate 3 — opt-in realistic-config smoke`.

---

### Task 12: torch parity script (manual, not CI)

**Model: fable-5 (medium)** — cross-framework weight mapping and numerical-mismatch debugging is the hardest task in the plan.

**Files:**
- Create: `scripts/pixeldit_parity.py`

Compare our forward against the PyTorch reference `PixDiT_T2I` on CPU/float32 with copied weights. Run via `uv run --with torch --index-strategy unsafe-best-match python scripts/pixeldit_parity.py` (or any env with torch-cpu; the reference uses only plain torch + sdpa).

Key mapping decisions (work them out in the script, document inline):
- Config pairing: ours `(D=64, heads=4, N=2, M=1, p=4, D_pix=8, K=3, cond_in_channels=5, cond_id_embedding="absolute", use_cond_rope=True, zero_init_blocks=False)` ↔ theirs `PixDiT_T2I(in_channels=1, num_groups=4, hidden_size=64, pixel_hidden_size=8, patch_depth=2, pixel_depth=1, patch_size=4, txt_embed_dim=5, txt_max_length=3, use_text_rope=True)` on a 16×16 input. Init policy is irrelevant — every parameter is copied.
- Weight transport torch→nnx: `Linear.weight` transposes; RMSNorm `weight` → `scale`; `y_pos_embedding (1, K, D)` → our FeatureEmbedder embed table `(K, D)`; t_embedder mlp[0]/mlp[2] → t_conditioner layers; per-block qkv/proj/adaLN/mlp(w1,w3,w2) per stream; PiT compress/expand/adaLN/attn(qkv, q_norm, k_norm, proj)/mlp(fc1, fc2); final layer norm+linear; pixel embedder proj (their sincos cache vs our table should match bit-wise from Task 2's test — assert anyway).
- Input layout: torch `(B, C, H, W)` = `jnp.transpose(ours, (0, 3, 1, 2))`; their output back to NHWC for comparison.
- Their forward signature: `net(x, t, y)` with `y (B, K, D_txt)`; ours `model(t, obs, cond)`.
- Success criterion: `max |Δ| < 1e-4` in float32 (build our model with `param_dtype=jnp.float32`). Print per-stage diffs (after patch embed, after each block via intermediates if needed) on failure.

- [x] **Step 1: write the script** (self-contained; `sys.path.insert` the `reference/PixelDiT` dir to import `pixdit_core`).
- [x] **Step 2: run it; iterate until parity or until a deviation is positively identified as intentional (document any in the script header and in the spec's deviation list).**
- [x] **Step 3: commit** — `chore(pixeldit): manual torch parity script (passes at 1e-4)`.

---

### Task 13: GRF example — PixelDiT config `1b` + model dispatch (final step)

**Model: sonnet**

**Files (separate repo: GenSBI-examples):**
- Modify: `/lhome/ific/a/aamerio/data/github/GenSBI-examples/examples/sbi-benchmarks/gaussian_random_field/train-grf.py`
- Create: `/lhome/ific/a/aamerio/data/github/GenSBI-examples/examples/sbi-benchmarks/gaussian_random_field/config/config_1b.yaml`

**Do NOT train the model** — this task only creates the files and verifies the model constructs from the yaml. NOTE: `config_3.yaml` in that directory has pre-existing uncommitted user modifications — do not touch or commit that file.

- [x] **Step 1: modify `train-grf.py`** with exactly these three edits (everything else unchanged):

  1. Docstring: title line → `"""Train FieldDiT or PixelDiT on gaussian_random_field (32x32), sample the posterior.` and add before the Usage block:
     ```
     The model is chosen by the config's model section: a `fielddit:` key builds
     FieldDiT (configs 1-3), a `pixeldit:` key builds PixelDiT (config 1b).
     ```
     plus a second usage line `python train-grf.py --config config/config_1b.yaml`.
  2. Drop the top-level `from gensbi.experimental.models import FieldDiT, FieldDiTParams` import (moved into `build_model`) and replace `build_model` (currently lines 60-65) with:
     ```python
     def resolve_model_section(cfg):
         """Return (model_kind, model_cfg) from the yaml: 'pixeldit' or 'fielddit'."""
         for kind in ("pixeldit", "fielddit"):
             if kind in cfg:
                 return kind, cfg[kind]
         raise KeyError("config must have a 'pixeldit:' or 'fielddit:' section")


     def build_model(model_kind, model_cfg, seed):
         kw = dict(model_cfg)
         kw["field_shape"] = tuple(kw["field_shape"])
         kw["param_dtype"] = getattr(jnp, kw.pop("param_dtype", "bfloat16"))
         if model_kind == "pixeldit":
             from gensbi.experimental.models import PixelDiT, PixelDiTParams

             return PixelDiT(PixelDiTParams(rngs=nnx.Rngs(seed), **kw))
         from gensbi.experimental.models import FieldDiT, FieldDiTParams

         kw["encoder_widths"] = tuple(kw["encoder_widths"])
         return FieldDiT(FieldDiTParams(rngs=nnx.Rngs(seed), **kw))
     ```
  3. In `main()` (currently lines 189-195 and 201-211): replace
     ```python
     model = build_model(cfg["fielddit"], seed=tcfg.get("seed", 0))
     ...
     print(f"FieldDiT parameters: {n_params / 1e6:.1f}M")
     ```
     with
     ```python
     model_kind, model_cfg = resolve_model_section(cfg)
     model = build_model(model_kind, model_cfg, seed=tcfg.get("seed", 0))
     ...
     print(f"{model_kind} parameters: {n_params / 1e6:.1f}M")
     ```
     and in the `FieldConditionalPipeline(...)` call replace `cfg["fielddit"]["field_shape"]` → `model_cfg["field_shape"]` and `cfg["fielddit"]["cond_dim"]` → `model_cfg["cond_dim"]`.

- [x] **Step 2: create `config/config_1b.yaml`** with exactly:

  ```yaml
  # GRF-32 PixelDiT — experiment 1b: faithful dual-level pixel-space DiT
  # (arXiv:2511.20645 port; spec docs/superpowers/specs/2026-06-12-pixeldit-port-design.md)
  pixeldit:
    in_channels: 1
    field_shape: [32, 32]
    patch_size: 4              # -> 8x8 = 64 patch tokens, 16 pixels/patch
    hidden_size: 384           # patch-level width D
    pixel_hidden_size: 16      # per-pixel width D_pix (paper value)
    patch_depth: 6             # N MMDiT blocks
    pixel_depth: 2             # M PiT blocks
    num_heads: 6               # head_dim 64 (patch and PiT pathways)
    cond_dim: 2                # theta = (log_std, alpha)
    cond_in_channels: 1
    cond_id_embedding: absolute
    use_cond_rope: false       # theta components are unordered -> no sequential rope
    param_dtype: bfloat16

  training:
    batch_size: 128
    val_batch_size: 128
    max_workers: 4             # grain mp_prefetch workers (null -> no prefetch)
    nsteps: 20000
    max_lr: 1.0e-4
    val_every: 100
    early_stopping: false
    multistep: 1
    experiment_id: 1b
    train_model: true
    restore_model: false
    seed: 0

  sampling:
    num_thetas: 3              # test rows used for plots
    nsamples: 16               # posterior samples per theta (P(k) statistics)
    nsamples_grid: 3           # samples shown per row in the field grid
    step_size: 0.01            # Euler ODE step (100 steps)
  ```

- [x] **Step 3: verify construction only** (no training; run from the example dir in the `gensbi` conda env):
  ```bash
  cd /lhome/ific/a/aamerio/data/github/GenSBI-examples/examples/sbi-benchmarks/gaussian_random_field
  JAX_PLATFORMS=cpu python - <<'EOF'
  import importlib.util, yaml
  spec = importlib.util.spec_from_file_location("traingrf", "train-grf.py")
  m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
  cfg = yaml.safe_load(open("config/config_1b.yaml"))
  kind, mcfg = m.resolve_model_section(cfg)
  model = m.build_model(kind, mcfg, seed=0)
  import jax, flax.nnx as nnx
  n = sum(l.size for l in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))
  print(f"OK: {kind}, {n/1e6:.1f}M params")
  EOF
  ```
  Expected: `OK: pixeldit, <~35>M params`. If `PixelDiTParams` kwargs drifted from the implemented dataclass, fix the yaml/script to match the code (the code is the source of truth at this point).
- [x] **Step 4: commit in the GenSBI-examples repo** (only these two files): `git add examples/sbi-benchmarks/gaussian_random_field/train-grf.py examples/sbi-benchmarks/gaussian_random_field/config/config_1b.yaml && git commit -m "feat(grf): PixelDiT config 1b + model dispatch"`.

---

## Final verification (orchestrator, after all tasks)

- [x] Full suite green: `JAX_PLATFORMS=cpu python -m pytest tests/` — expect prior count (667) + new pixeldit tests, 1 pre-existing skip + the new smoke skip.
- [x] `git log --oneline` shows one commit per task on branch `FieldDiT`.
- [x] Spec deviation list updated if the parity script surfaced intentional deviations.
- [x] Invoke `superpowers:requesting-code-review` per SDD; then report to the user (do not merge — `FieldDiT` stays unmerged by policy).
