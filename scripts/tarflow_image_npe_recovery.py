"""Image-conditioned NPE recovery for TransformerFlow (vector theta, image cond).
Standalone; intended for cluster/GPU scheduling, not the pytest battery.

Linear-Gaussian: image x = (G @ theta) reshaped + sigma*noise. Train q(theta|x)
and recover the analytic posterior by direct sampling (no NUTS).
"""
import argparse
import sys
import tempfile
import time


def main():
    p = argparse.ArgumentParser(description="Image-NPE TransformerFlow recovery.")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--n-data", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--platform", type=str, default=None)
    p.add_argument("--num-blocks", type=int, default=6)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--head-dim", type=int, default=16)
    p.add_argument("--atol", type=float, default=0.25)
    p.add_argument("--results-file", type=str, default=None,
                   help="If set, also write the stats/verdict report to this text file.")
    args = p.parse_args()

    if args.platform is not None:
        import os
        os.environ["JAX_PLATFORMS"] = args.platform

    import jax
    import jax.numpy as jnp
    import numpy as np
    import grain
    from flax import nnx
    from gensbi.models import TarFlow, TarFlowParams
    from gensbi.recipes.flow_pipeline import ConditionalFlowPipeline

    smoke = args.smoke
    n_data = args.n_data or (2_000 if smoke else 40_000)
    nsteps = args.steps or (10 if smoke else 6_000)
    num_samples = args.num_samples or (200 if smoke else 4_000)
    val_every = 1 if smoke else 200

    H = Wd = 4
    Ch, D, SIGMA = 1, 2, 0.5
    Mdim = H * Wd
    G = jax.random.normal(jax.random.PRNGKey(123), (Mdim, D))

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

    report_lines = []

    def emit(line=""):
        print(line)
        report_lines.append(line)

    def write_report():
        if args.results_file:
            import os
            d = os.path.dirname(os.path.abspath(args.results_file))
            os.makedirs(d, exist_ok=True)
            with open(args.results_file, "w") as fh:
                fh.write("\n".join(report_lines) + "\n")

    emit("=" * 60)
    emit("TransformerFlow image-NPE recovery")
    emit(f"  mode        : {'SMOKE' if smoke else 'FULL'}")
    emit(f"  n_data      : {n_data}")
    emit(f"  nsteps      : {nsteps}")
    emit(f"  num_samples : {num_samples}")
    emit(f"  num_blocks  : {args.num_blocks}")
    emit(f"  channels    : {args.channels}")
    emit(f"  head_dim    : {args.head_dim}")
    emit(f"  atol        : {args.atol}")
    emit(f"  seed        : {args.seed}")
    emit(f"  platform    : {jax.default_backend()}")
    emit("=" * 60)

    t0 = time.time()
    theta, x = simulate(jax.random.PRNGKey(args.seed), n_data)
    n_train = int(n_data * 0.9)

    def make_ds(obs, cond):
        idx = grain.MapDataset.source(list(range(len(obs))))
        obs_n, cond_n = np.array(obs), np.array(cond)
        return (idx.shuffle(0).repeat().to_iter_dataset().batch(256)
                .map(lambda i: (obs_n[np.array(i)], cond_n[np.array(i)])))

    flow = TarFlow(TarFlowParams(rngs=nnx.Rngs(args.seed), dim=D,
                                 modeled="vector", cond="image",
                                 cond_img_size=H, cond_patch_size=2,
                                 cond_channels=Ch, head_dim=args.head_dim,
                                 num_heads=args.channels // args.head_dim,
                                 num_blocks=args.num_blocks, layers_per_block=2,
                                 standardize=True))
    cfg = ConditionalFlowPipeline.get_default_training_config()
    cfg.update(dict(nsteps=nsteps, val_every=val_every, max_lr=3e-4,
                    checkpoint_dir=tempfile.mkdtemp(), early_stopping=False))
    pipe = ConditionalFlowPipeline(flow, make_ds(theta[:n_train][..., None], x[:n_train]),
                                   make_ds(theta[n_train:][..., None], x[n_train:]),
                                   dim_obs=D, dim_cond=Mdim, structured_cond=True,
                                   training_config=cfg)
    pipe.fit_standardization(theta[:n_train][..., None])
    pipe.train(nnx.Rngs(args.seed), nsteps=nsteps, save_model=False)

    theta_o = jnp.array([0.7, -0.4])
    x_o = (theta_o @ G.T).reshape(1, H, Wd, Ch)
    mean_a, cov_a = analytic_posterior(x_o.reshape(-1))
    s = pipe.sample(jax.random.PRNGKey(7), x_o, nsamples=num_samples)[..., 0]
    mean_s, cov_s = jnp.mean(s, axis=0), jnp.cov(s.T)
    emit(f"\nelapsed={time.time()-t0:.1f}s")
    emit(f"analytic mean {mean_a}  achieved {mean_s}")
    emit(f"analytic cov\n{cov_a}\nachieved\n{cov_s}")

    if smoke:
        assert jnp.all(jnp.isfinite(mean_s)) and jnp.all(jnp.isfinite(cov_s))
        emit("\nSMOKE OK"); write_report(); sys.exit(0)
    ok = bool(jnp.allclose(mean_s, mean_a, atol=args.atol)
              and jnp.allclose(cov_s, cov_a, atol=args.atol))
    emit("\nRECOVERY PASS" if ok else "\nRECOVERY FAIL"); write_report(); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
