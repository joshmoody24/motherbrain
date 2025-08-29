import collections
import functools
import numpy as np

import jax
import jax.numpy as jnp
import haiku as hk
import optax

# ------------------------
# Config (tweak as needed)
# ------------------------
CONTEXT = 32  # transformer context (in steps)
HORIZON = 1024  # steps per PPO rollout
PPO_EPOCHS = 4
MINIBATCH = 256
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
LR_POLICY_VALUE = 3e-4
LR_RND = 1e-3
RND_EMA_DECAY = 0.99  # intrinsic reward normalization running stats
SEED = 0

# Logging cadence
LOG_EVERY = 200  # per-step logs
PRINT_MINIBATCH_METRICS = (
    False  # set True to print every minibatch update (very verbose)
)


# ------------------------
# Utilities
# ------------------------
class RunningNorm:
    """Running mean/std for reward normalization (simple EMA)."""

    def __init__(self, decay=0.99, eps=1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.decay = decay
        self.eps = eps
        self.inited = False

    def update(self, x):
        x = float(x)
        if not self.inited:
            self.mean = x
            self.var = 1.0
            self.inited = True
        else:
            delta = x - self.mean
            self.mean += (1.0 - self.decay) * delta
            self.var = self.decay * self.var + (1.0 - self.decay) * (delta * delta)
        return (x - self.mean) / (np.sqrt(self.var) + self.eps)


def bernoulli_logprob(logits, action):
    """Sum of Bernoulli log-probs for MultiBinary action."""
    return jnp.sum(
        action * jax.nn.log_sigmoid(logits)
        + (1 - action) * jax.nn.log_sigmoid(-logits),
        axis=-1,
    )


def bernoulli_entropy(logits):
    p = jax.nn.sigmoid(logits)
    return jnp.sum(-(p * jnp.log(p + 1e-8) + (1 - p) * jnp.log(1 - p + 1e-8)), axis=-1)


# ------------------------
# Networks (Haiku)
# ------------------------
def encoder_fn(x):
    """Encode observation to latent z (keep tiny & simple)."""
    x = x.astype(jnp.float32) / 255.0
    x = jnp.ravel(x)
    return hk.nets.MLP([256, 128])(x)  # latent dim = 128


def rnd_target_fn(z):
    """Frozen random target network (no training)."""
    return hk.nets.MLP([256, 256, 128])(z)


def rnd_predictor_fn(z):
    """Trainable predictor that tries to match target(z)."""
    return hk.nets.MLP([256, 256, 128])(z)


class TinyTransformer(hk.Module):
    """Small Transformer encoder over a short context of tokens."""

    def __init__(self, d_model=128, n_heads=4, ffw=256, n_layers=2, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.n_heads = n_heads
        self.ffw = ffw
        self.n_layers = n_layers

    def __call__(self, x):
        """
        x: [T, D_in] tokens; returns [T, d_model]
        causal self-attention (no future access)
        """
        T, _ = x.shape
        # linear embed to d_model
        h = hk.Linear(self.d_model)(x)

        # learned positional embeddings
        pos = hk.get_parameter(
            "pos", [CONTEXT, self.d_model], init=hk.initializers.TruncatedNormal(0.02)
        )
        h = h + pos[:T]

        # causal mask: [1, T, T] (batch axis for MHA)
        attn_mask = jnp.tril(jnp.ones((T, T), dtype=bool))[None, :, :]

        for _ in range(self.n_layers):
            # MHA
            ln1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(h)
            mha = hk.MultiHeadAttention(
                self.n_heads,
                self.d_model // self.n_heads,
                w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
            )
            attn_out = mha(ln1, ln1, ln1, mask=attn_mask)
            h = h + attn_out

            # FFN
            ln2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(h)
            ff = hk.nets.MLP([self.ffw, self.d_model])
            h = h + ff(ln2)

        return h  # [T, d_model]


def policy_value_fn(tokens, num_buttons):
    """
    tokens: [T, D_in] where D_in = dim(z) + num_buttons (prev action)
    returns: logits_t (per button), value_t (scalar) at last timestep
    """
    core = TinyTransformer(d_model=128, n_heads=4, ffw=256, n_layers=2)(tokens)
    last = core[-1]  # last token
    logits = hk.Linear(num_buttons)(last)
    value = jnp.squeeze(hk.Linear(1)(last), axis=-1)
    return logits, value


# Transformations
encoder = hk.without_apply_rng(hk.transform(encoder_fn))
rnd_target = hk.without_apply_rng(hk.transform(rnd_target_fn))
rnd_predictor = hk.without_apply_rng(hk.transform(rnd_predictor_fn))
policy_value = lambda num_buttons: hk.without_apply_rng(
    hk.transform(lambda tokens: policy_value_fn(tokens, num_buttons))
)


# ------------------------
# PPO pieces (advantages, etc.)
# ------------------------
def compute_gae(rewards, values, next_value, dones, gamma=GAMMA, lam=LAMBDA):
    """Generalized Advantage Estimation (numpy)."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - float(dones[t])
        delta = (
            rewards[t]
            + gamma * (next_value if t == T - 1 else values[t + 1]) * nonterminal
            - values[t]
        )
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        adv[t] = lastgaelam
    returns = adv + values
    return adv, returns


# ------------------------
# Main entry
# ------------------------
def run_agent(env):
    rng = jax.random.PRNGKey(SEED)
    num_buttons = env.action_space.shape[0]

    # Init params
    dummy_obs = np.zeros(env.observation_space.shape, dtype=np.uint8)
    z0 = jnp.zeros(128)

    enc_params = encoder.init(rng, dummy_obs)
    rnd_t_params = rnd_target.init(rng, z0)  # frozen
    rnd_p_params = rnd_predictor.init(rng, z0)  # trainable

    pv = policy_value(num_buttons)
    tokens0 = jnp.zeros((CONTEXT, 128 + num_buttons))  # token = [z, prev_action]
    pv_params = pv.init(rng, tokens0)

    # Optax optimizers
    opt_pv = optax.adam(LR_POLICY_VALUE)
    opt_rnd = optax.adam(LR_RND)
    opt_pv_state = opt_pv.init(pv_params)
    opt_rnd_state = opt_rnd.init(rnd_p_params)

    # JIT apply fns
    enc_apply = encoder.apply
    pv_apply = pv.apply
    rnd_t_apply = rnd_target.apply
    rnd_p_apply = rnd_predictor.apply

    # Running normalization for intrinsic reward
    rnorm = RunningNorm(decay=RND_EMA_DECAY)

    # Buffers for PPO rollout
    obs_buf, z_buf, a_buf = [], [], []
    prev_a_buf = []  # store prev action at each step (FIX for shape mismatch)
    logp_buf, v_buf = [], []
    rint_buf, rint_raw_buf = [], []
    done_buf = []

    # Context buffer (deque of last tokens)
    ctx = collections.deque(maxlen=CONTEXT)
    prev_action = np.zeros(num_buttons, dtype=np.float32)

    # Simple action statistics
    button_press_counts = np.zeros(num_buttons, dtype=np.int64)

    obs, info = env.reset()
    done = False
    step = 0
    episode = 0

    def make_tokens(z, prev_a):
        tok = np.concatenate(
            [np.array(z, dtype=np.float32), prev_a.astype(np.float32)], axis=-1
        )
        if len(ctx) == 0:
            for _ in range(CONTEXT - 1):
                ctx.append(np.zeros_like(tok))
        ctx.append(tok)
        return np.stack(list(ctx), axis=0)

    # --- JITed loss & update (PPO) ---
    @functools.partial(jax.jit, static_argnums=(7,))
    def ppo_update(
        pv_params,
        rnd_p_params,
        opt_pv_state,
        opt_rnd_state,
        batch,
        clip_eps,
        entropy_coef,
        value_coef,
        num_buttons,
    ):
        (obs_b, prev_a_b, act_b, adv_b, ret_b, old_logp_b) = batch  # shapes: [N, ...]

        def loss_pv(pv_params):
            z_b = jax.vmap(enc_apply, in_axes=(None, 0))(enc_params, obs_b)  # [N,128]
            tokens_b = jnp.concatenate([z_b, prev_a_b], axis=-1)  # [N, 128+nb]
            tokens_ctx = jnp.tile(tokens_b[:, None, :], (1, CONTEXT, 1))  # [N, T, D]
            logits_b, v_b = jax.vmap(pv_apply, in_axes=(None, 0))(pv_params, tokens_ctx)
            logp_new = bernoulli_logprob(logits_b, act_b)  # [N]
            entropy = bernoulli_entropy(logits_b)  # [N]

            ratio = jnp.exp(logp_new - old_logp_b)
            pg1 = ratio * adv_b
            pg2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_b
            policy_loss = -jnp.mean(jnp.minimum(pg1, pg2))

            v_loss = jnp.mean((ret_b - v_b) ** 2)
            ent_loss = -jnp.mean(entropy)

            total = policy_loss + value_coef * v_loss + entropy_coef * ent_loss
            return total, (policy_loss, v_loss, ent_loss)

        grads_pv, aux = jax.grad(loss_pv, has_aux=True)(pv_params)
        updates_pv, opt_pv_state = opt_pv.update(grads_pv, opt_pv_state)
        pv_params = optax.apply_updates(pv_params, updates_pv)

        # RND predictor update (self-supervised)
        def loss_rnd(rnd_p_params):
            z_b = jax.vmap(enc_apply, in_axes=(None, 0))(enc_params, obs_b)
            t = jax.vmap(rnd_t_apply, in_axes=(None, 0))(rnd_t_params, z_b)
            p = jax.vmap(rnd_p_apply, in_axes=(None, 0))(rnd_p_params, z_b)
            return jnp.mean((jax.lax.stop_gradient(t) - p) ** 2)

        rnd_loss = jax.grad(loss_rnd)(rnd_p_params)
        # For logging the value, we also compute the scalar (cheap)
        rnd_scalar = loss_rnd(rnd_p_params)

        updates_rnd, opt_rnd_state = opt_rnd.update(rnd_loss, opt_rnd_state)
        rnd_p_params = optax.apply_updates(rnd_p_params, updates_rnd)

        # aux: (policy_loss, v_loss, ent_loss)
        return pv_params, rnd_p_params, opt_pv_state, opt_rnd_state, aux, rnd_scalar

    # ------------- main loop -------------
    while True:
        # --- Encode, build tokens, policy forward ---
        z = enc_apply(enc_params, obs)  # [128]
        tokens = make_tokens(z, prev_action)  # [CONTEXT, 128+nb]
        logits, value = pv_apply(pv_params, jnp.array(tokens))

        # Sample MultiBinary action
        probs = jax.nn.sigmoid(logits)
        a = (np.random.rand(num_buttons) < np.array(probs)).astype(np.int8)
        logp = float(bernoulli_logprob(logits, a))
        ent_now = float(bernoulli_entropy(logits))

        # Step env
        prev_a_buf.append(prev_action.copy())  # store prev action for this step (FIX)
        next_obs, _, terminated, truncated, info = env.step(a)
        done = terminated or truncated

        # Intrinsic reward via RND (use next state for novelty)
        z_next = enc_apply(enc_params, next_obs)
        t = rnd_t_apply(rnd_t_params, z_next)
        p = rnd_p_apply(rnd_p_params, z_next)
        r_int_raw = float(jnp.mean((t - p) ** 2))
        r_int = rnorm.update(r_int_raw)

        # Log transition
        obs_buf.append(obs)
        z_buf.append(np.asarray(z))
        a_buf.append(a.astype(np.float32))
        logp_buf.append(logp)
        v_buf.append(float(value))
        rint_buf.append(r_int)
        rint_raw_buf.append(r_int_raw)
        done_buf.append(done)

        # Button stats
        button_press_counts += a.astype(np.int64)

        step += 1
        prev_action = a.astype(np.float32)
        obs = next_obs

        # Per-step logging
        if step % LOG_EVERY == 0:
            pressed_rate = float(np.mean(a))
            mean_prob = float(np.mean(probs))
            print(
                f"[step {step}] val={float(value):.3f} ent={ent_now:.3f} "
                f"press_rate={pressed_rate:.2f} mean_prob={mean_prob:.2f} "
                f"r_int_raw={r_int_raw:.4f} r_int_norm={r_int:.4f} "
                f"r_int_norm_mean={rnorm.mean:.4f} r_int_norm_std={np.sqrt(rnorm.var):.4f}"
            )

        # Rollout end -> PPO update
        if step % HORIZON == 0 or done:
            # Bootstrap value for last state
            if done:
                next_v = 0.0
            else:
                z_last = enc_apply(enc_params, obs)
                tokens_last = make_tokens(z_last, prev_action)
                _, v_last = pv_apply(pv_params, jnp.array(tokens_last))
                next_v = float(v_last)

            # Compute advantages (numpy domain)
            adv, ret = compute_gae(
                np.array(rint_buf, dtype=np.float32),
                np.array(v_buf, dtype=np.float32),
                next_v,
                np.array(done_buf, dtype=np.bool_),
            )

            # Prepare flat batches
            obs_b = np.array(obs_buf, dtype=np.uint8)
            prev_a_b = np.array(
                prev_a_buf, dtype=np.float32
            )  # << only prev action (FIX)
            act_b = np.array(a_buf, dtype=np.float32)
            adv_b = (adv - adv.mean()) / (adv.std() + 1e-8)
            ret_b = ret.astype(np.float32)
            old_logp_b = np.array(logp_buf, dtype=np.float32)

            N = len(obs_b)
            idxs = np.arange(N)

            # PPO epochs
            epoch_pl, epoch_vl, epoch_ent, epoch_rnd = [], [], [], []
            for epoch in range(PPO_EPOCHS):
                np.random.shuffle(idxs)
                for s in range(0, N, MINIBATCH):
                    mb = idxs[s : s + MINIBATCH]
                    batch = (
                        obs_b[mb],
                        prev_a_b[mb],  # tokens reconstructed in jit as [z, prev_a]
                        act_b[mb],
                        adv_b[mb],
                        ret_b[mb],
                        old_logp_b[mb],
                    )
                    (
                        pv_params,
                        rnd_p_params,
                        opt_pv_state,
                        opt_rnd_state,
                        aux,
                        rnd_scalar,
                    ) = ppo_update(
                        pv_params,
                        rnd_p_params,
                        opt_pv_state,
                        opt_rnd_state,
                        batch,
                        CLIP_EPS,
                        ENTROPY_COEF,
                        VALUE_COEF,
                        num_buttons,
                    )
                    pl, vl, el = map(float, aux)
                    epoch_pl.append(pl)
                    epoch_vl.append(vl)
                    epoch_ent.append(el)
                    epoch_rnd.append(float(rnd_scalar))
                    if PRINT_MINIBATCH_METRICS:
                        print(
                            f"  mb[{s}:{s+MINIBATCH}] pol={pl:.4f} val={vl:.4f} ent={el:.4f} rnd={float(rnd_scalar):.4f}"
                        )

            # Rollout summary logging
            mean_r_raw = float(np.mean(rint_raw_buf)) if rint_raw_buf else 0.0
            mean_r_norm = float(np.mean(rint_buf)) if rint_buf else 0.0
            std_r_norm = float(np.std(rint_buf)) if rint_buf else 0.0
            mean_adv = float(np.mean(adv)) if len(adv) else 0.0
            act_press_rate = float(np.mean(act_b)) if len(act_b) else 0.0
            top_buttons = np.argsort(-button_press_counts)[:3]
            top_counts = button_press_counts[top_buttons]

            print(
                f"[steps {step}] rollout N={N} | r_int_raw(mean)={mean_r_raw:.4f} "
                f"r_int_norm(mean±std)={mean_r_norm:.4f}±{std_r_norm:.4f} | "
                f"adv(mean)={mean_adv:.4f} | "
                f"ppo_loss(pol/val/ent)={np.mean(epoch_pl):.4f}/{np.mean(epoch_vl):.4f}/{np.mean(epoch_ent):.4f} | "
                f"rnd_loss(mean)={np.mean(epoch_rnd):.4f} | "
                f"press_rate={act_press_rate:.2f} | top_buttons={top_buttons.tolist()} counts={top_counts.tolist()}"
            )

            # Clear rollout buffers (keep current obs/ctx)
            obs_buf.clear()
            z_buf.clear()
            a_buf.clear()
            prev_a_buf.clear()
            logp_buf.clear()
            v_buf.clear()
            rint_buf.clear()
            rint_raw_buf.clear()
            done_buf.clear()

        if done:
            obs, info = env.reset()
            prev_action = np.zeros(num_buttons, dtype=np.float32)
            ctx.clear()
            episode += 1
            print(
                f"[episode {episode}] reset environment; cumulative button counts={button_press_counts.tolist()}"
            )
