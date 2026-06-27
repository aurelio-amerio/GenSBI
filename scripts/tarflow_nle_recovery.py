"""TransformerFlow (TarFlow) NLE recovery script. Intended for cluster/GPU scheduling, not the pytest battery."""

import argparse
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(
        description="TransformerFlow NLE recovery — linear-Gaussian posterior check."
    )
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke mode: minimal steps to verify wiring, no recovery assertion.")
    parser.add_argument("--steps", type=int, default=None,
                        help="Training steps (default: 10 smoke / 4000 full).")
    parser.add_argument("--num-warmup", type=int, default=None,
                        help="NUTS warmup steps (default: 5 smoke / 500 full).")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="NUTS samples (default: 20 smoke / 4000 full).")
    parser.add_argument("--n-data", type=int, default=None,
                        help="Total simulated data points (default: 2000 smoke / 20000 full).")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for data/flow/train (default: 0).")
    parser.add_argument("--platform", type=str, default=None,
                        help="JAX platform override, e.g. 'cpu' or 'gpu'. Default: JAX picks.")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Checkpoint directory (default: fresh tempdir).")
    parser.add_argument("--save-model", action="store_true", default=False,
                        help="Save model checkpoints during training.")
    parser.add_argument("--num-blocks", type=int, default=6,
                        help="Number of transformer blocks (default: 6).")
    parser.add_argument("--channels", type=int, default=64,
                        help="Channel width (default: 64).")
    parser.add_argument("--head-dim", type=int, default=16,
                        help="Attention head dimension (default: 16).")
    parser.add_argument("--atol", type=float, default=0.2,
                        help="Recovery tolerance for allclose checks (default: 0.2).")
    args = parser.parse_args()

    # Set platform before importing JAX so the backend choice takes effect.
    if args.platform is not None:
        import os
        os.environ["JAX_PLATFORMS"] = args.platform

    import jax
    import jax.numpy as jnp
    import numpy as np
    import grain
    from flax import nnx

    from gensbi.core.prior import make_gaussian_prior
    from gensbi.models import TarFlow, TarFlowParams
    from gensbi.recipes.flow_pipeline import ConditionalFlowPipeline
    from gensbi.inference import NLEPosterior

    # Resolve defaults based on mode.
    smoke = args.smoke
    n_data = args.n_data if args.n_data is not None else (2_000 if smoke else 20_000)
    nsteps = args.steps if args.steps is not None else (10 if smoke else 4_000)
    val_every = 1 if smoke else 200
    num_warmup = args.num_warmup if args.num_warmup is not None else (5 if smoke else 500)
    num_samples = args.num_samples if args.num_samples is not None else (20 if smoke else 4_000)
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir is not None else tempfile.mkdtemp()

    D, M, SIGMA = 2, 3, 0.5

    print("=" * 60)
    print("TransformerFlow NLE recovery")
    print(f"  mode        : {'SMOKE' if smoke else 'FULL'}")
    print(f"  n_data      : {n_data}")
    print(f"  nsteps      : {nsteps}")
    print(f"  num_warmup  : {num_warmup}")
    print(f"  num_samples : {num_samples}")
    print(f"  num_blocks  : {args.num_blocks}")
    print(f"  channels    : {args.channels}")
    print(f"  head_dim    : {args.head_dim}")
    print(f"  atol        : {args.atol}")
    print(f"  seed        : {args.seed}")
    print(f"  platform    : {jax.default_backend()}")
    print(f"  checkpoint  : {checkpoint_dir}")
    print("=" * 60)

    G = jnp.array([[1.0, 0.5], [0.0, 1.0], [0.5, -1.0]])  # (M, D)

    def _simulate(key, n):
        kth, ke = jax.random.split(key)
        theta = jax.random.normal(kth, (n, D))
        x = theta @ G.T + SIGMA * jax.random.normal(ke, (n, M))
        return theta, x

    def _analytic_posterior(x_o):
        prec = jnp.eye(D) + (G.T @ G) / SIGMA ** 2
        cov = jnp.linalg.inv(prec)
        mean = cov @ (G.T @ x_o) / SIGMA ** 2
        return mean, cov

    def split(d):
        return d[:, :M], d[:, M:]  # (obs=x, cond=theta)

    def make_ds(arr):
        return (grain.MapDataset.source(np.array(arr)).shuffle(0).repeat()
                .to_iter_dataset().batch(256).map(split))

    t0 = time.time()

    data_key = jax.random.PRNGKey(args.seed)
    theta, x = _simulate(data_key, n_data)
    data = jnp.concatenate([x[..., None], theta[..., None]], axis=1)  # x FIRST: (N, M+D, 1)

    n_train = int(n_data * 0.9)
    train_data = data[:n_train]
    val_data = data[n_train:]

    flow = TarFlow(TarFlowParams(
        rngs=nnx.Rngs(args.seed),
        dim=M,
        cond_dim=D,
        head_dim=args.head_dim,
        num_heads=args.channels // args.head_dim,
        num_blocks=args.num_blocks,
        layers_per_block=2,
        standardize=True,
    ))

    cfg = ConditionalFlowPipeline.get_default_training_config()
    cfg.update(dict(
        nsteps=nsteps,
        val_every=val_every,
        max_lr=3e-4,
        checkpoint_dir=checkpoint_dir,
        early_stopping=False,
    ))

    pipe = ConditionalFlowPipeline(
        flow,
        make_ds(train_data),
        make_ds(val_data),
        M, D,
        ch_obs=1,
        ch_cond=1,
        training_config=cfg,
    )
    pipe.fit_standardization(train_data[:, :M])
    pipe.train(nnx.Rngs(args.seed), nsteps=nsteps, save_model=args.save_model)

    x_o = jnp.array([1.0, -0.5, 0.3])
    mean_a, cov_a = _analytic_posterior(x_o)

    from gensbi.inference import MCLMC
    prior = make_gaussian_prior((D,))
    post = NLEPosterior(pipe.ema_model, prior)
    sample_key = jax.random.PRNGKey(7)
    s = post.sample(sample_key, x_o,
                    sampler=MCLMC(num_samples=num_samples, num_tuning_steps=num_warmup))[..., 0]

    mean_s = jnp.mean(s, axis=0)
    cov_s = jnp.cov(s.T)

    elapsed = time.time() - t0

    print(f"\nAnalytic posterior mean : {mean_a}")
    print(f"Achieved posterior mean : {mean_s}")
    print(f"\nAnalytic posterior cov :\n{cov_a}")
    print(f"Achieved posterior cov :\n{cov_s}")
    print(f"\nElapsed: {elapsed:.1f}s")

    if smoke:
        assert jnp.all(jnp.isfinite(mean_s)), "mean_s contains non-finite values"
        assert jnp.all(jnp.isfinite(cov_s)), "cov_s contains non-finite values"
        print("\nSMOKE OK")
        sys.exit(0)
    else:
        mean_ok = bool(jnp.allclose(mean_s, mean_a, atol=args.atol))
        cov_ok = bool(jnp.allclose(cov_s, cov_a, atol=args.atol))
        if mean_ok and cov_ok:
            print("\nRECOVERY PASS")
            sys.exit(0)
        else:
            print(f"\nRECOVERY FAIL  (mean_ok={mean_ok}, cov_ok={cov_ok})")
            sys.exit(1)


if __name__ == "__main__":
    main()
