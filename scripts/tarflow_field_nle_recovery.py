"""Field-level NLE recovery for TransformerFlow (image modeled x, vector theta).
Standalone; intended for cluster/GPU scheduling, not the pytest battery.

Linear-Gaussian: x_image = (G @ theta) reshaped to (H,W,1) + sigma*noise, with a
known G so the posterior over theta given the full image is analytic. NLE+NUTS
should recover it.
"""
import argparse
import sys
import tempfile
import time


def main():
    p = argparse.ArgumentParser(description="Field-NLE TransformerFlow recovery.")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--num-warmup", type=int, default=None)
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--n-data", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--platform", type=str, default=None)
    p.add_argument("--num-blocks", type=int, default=6)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--head-dim", type=int, default=16)
    p.add_argument("--atol", type=float, default=0.25)
    args = p.parse_args()

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

    smoke = args.smoke
    n_data = args.n_data or (2_000 if smoke else 40_000)
    nsteps = args.steps or (10 if smoke else 6_000)
    num_warmup = args.num_warmup or (5 if smoke else 500)
    num_samples = args.num_samples or (20 if smoke else 4_000)
    val_every = 1 if smoke else 200

    H = Wd = 4
    Ch, D, SIGMA = 1, 2, 0.5
    Mdim = H * Wd
    G = jax.random.normal(jax.random.PRNGKey(123), (Mdim, D))   # (16, 2)

    def simulate(key, n):
        kth, ke = jax.random.split(key)
        theta = jax.random.normal(kth, (n, D))
        flat = theta @ G.T + SIGMA * jax.random.normal(ke, (n, Mdim))
        return theta, flat.reshape(n, H, Wd, Ch)

    def analytic_posterior(x_o_flat):
        prec = jnp.eye(D) + (G.T @ G) / SIGMA ** 2
        cov = jnp.linalg.inv(prec)
        mean = cov @ (G.T @ x_o_flat) / SIGMA ** 2
        return mean, cov

    t0 = time.time()
    theta, x = simulate(jax.random.PRNGKey(args.seed), n_data)
    n_train = int(n_data * 0.9)

    def make_ds(obs, cond):
        idx = grain.MapDataset.source(list(range(len(obs))))
        obs_n, cond_n = np.array(obs), np.array(cond)
        return (idx.shuffle(0).repeat().to_iter_dataset().batch(256)
                .map(lambda i: (obs_n[np.array(i)], cond_n[np.array(i)])))

    flow = TarFlow(TarFlowParams(rngs=nnx.Rngs(args.seed), cond_dim=D,
                                 modeled="image", img_size=H, patch_size=2,
                                 img_channels=Ch, head_dim=args.head_dim,
                                 num_heads=args.channels // args.head_dim,
                                 num_blocks=args.num_blocks, layers_per_block=2,
                                 standardize=True))
    cfg = ConditionalFlowPipeline.get_default_training_config()
    cfg.update(dict(nsteps=nsteps, val_every=val_every, max_lr=3e-4,
                    checkpoint_dir=tempfile.mkdtemp(), early_stopping=False))
    pipe = ConditionalFlowPipeline(flow, make_ds(x[:n_train], theta[:n_train]),
                                   make_ds(x[n_train:], theta[n_train:]),
                                   dim_obs=Mdim, dim_cond=D, structured_obs=True,
                                   training_config=cfg)
    pipe.fit_standardization(x[:n_train])
    pipe.train(nnx.Rngs(args.seed), nsteps=nsteps, save_model=False)

    theta_o = jnp.array([0.7, -0.4])
    x_o = (theta_o @ G.T).reshape(H, Wd, Ch)
    mean_a, cov_a = analytic_posterior(x_o.reshape(-1))
    post = NLEPosterior(pipe.ema_model, make_gaussian_prior((D,)),
                        num_warmup=num_warmup, num_samples=num_samples,
                        structured_obs=True)
    s = post.sample(jax.random.PRNGKey(7), x_o)[..., 0]
    mean_s, cov_s = jnp.mean(s, axis=0), jnp.cov(s.T)
    print(f"mode={'SMOKE' if smoke else 'FULL'} elapsed={time.time()-t0:.1f}s")
    print(f"analytic mean {mean_a}  achieved {mean_s}")
    print(f"analytic cov\n{cov_a}\nachieved\n{cov_s}")

    if smoke:
        assert jnp.all(jnp.isfinite(mean_s)) and jnp.all(jnp.isfinite(cov_s))
        print("SMOKE OK"); sys.exit(0)
    ok = bool(jnp.allclose(mean_s, mean_a, atol=args.atol)
              and jnp.allclose(cov_s, cov_a, atol=args.atol))
    print("RECOVERY PASS" if ok else "RECOVERY FAIL"); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
