import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow_probability import distributions as tfd

import cnns
from cells import *
import tools


class CWVAE:
    def __init__(
        self,
        levels,
        tmp_abs_factor,
        state_sizes,
        embed_size,
        cell_type,
        lr,
        min_stddev,
        mean_only_cell=False,
        reset_states=False,
        var_scope="CWVAE",
    ):
        self.cell_type = cell_type
        self._levels = levels
        self._state_size = state_sizes["stoch"]
        self._detstate_size = state_sizes["deter"]
        self._embed_size = embed_size
        self._var_scope = var_scope
        self.lr = lr
        self._min_stddev = min_stddev
        self._tmp_abs_factor = tmp_abs_factor
        self._reset_states = reset_states

        self.cells = []
        for i_lvl in range(self._levels):
            if self.cell_type == "RSSMCell":
                assert (
                    self._detstate_size
                ), "deter state size should be non-zero int, found {}".format(
                    self._detstate_size
                )
                cell = RSSMCell(
                    self._state_size,
                    self._detstate_size,
                    self._embed_size,
                    reset_states=self._reset_states,
                    min_stddev=self._min_stddev,
                    mean_only=mean_only_cell,
                    var_scope="cell_" + str(i_lvl),
                )
            else:
                raise ValueError("Cell type {} not supported".format(self.cell_type))
            self.cells.append(cell)

    def hierarchical_unroll(
        self, inputs, actions=None, use_observations=None, initial_state=None
    ):
        """
        Used to unroll a list of recurrent cells.

        Arguments:
            cells
            inputs : list of encoded observations
                Number of nodes at every level in 'inputs' is the exact number of nodes to be unrolled
            actions
            use_observations : bool or list[bool]
            initial_state : list of cell states
        """
        if use_observations is None:
            use_observations = self._levels * [True]
        elif not isinstance(use_observations, list):
            use_observations = self._levels * [use_observations]

        if initial_state is None:
            initial_state = self._levels * [None]

        level_top = self._levels - 1
        inputs_top = inputs[level_top]

        level = level_top

        # Feeding in zeros as context to the top level.
        context = tf.zeros(
            shape=tf.concat(
                [tf.shape(inputs_top)[0:2], [self.cells[-1].state_size["output"]]], -1
            )
        )

        # Init reset_state: alternate zeros and ones, of the size of input[level=-2]
        if level_top >= 1:
            inputs_top_ = inputs[level_top - 1]
            temp_zeros = tf.zeros(
                tf.concat([tf.shape(inputs_top_)[:2], [tf.constant(1)]], -1)
            )
            temp_ones = tf.ones_like(temp_zeros)
            _reset_state = tf.concat([temp_zeros, temp_ones], -1)
            _reset_state = tf.reshape(_reset_state, [tf.shape(_reset_state)[0], -1, 1])
            _reset_state = tf.slice(
                _reset_state,
                [0, 0, 0],
                tf.concat([tf.shape(inputs_top_)[:2], [tf.constant(1)]], -1),
            )
        else:
            _reset_state = tf.no_op()

        prior_list = []  # Stored bot to top.
        posterior_list = []  # Stored bot to top.

        last_state_all_levels = list([])

        for level in range(level_top, -1, -1):
            obs_inputs = inputs[level]
            print("Input shape in CWVAE level {}: {}".format(level, obs_inputs.shape))
            if level == level_top:
                reset_state, reset_state_next = (
                    tf.ones(shape=tf.concat([tf.shape(obs_inputs)[0:2], [1]], -1)),
                    _reset_state,
                )
            else:
                reset_state, reset_state_next = (
                    reset_state,
                    tf.tile(reset_state, [1, self._tmp_abs_factor, 1]),
                )

            # Pruning reset_state, context from previous layer, to match the num of nodes reqd as in inputs[level]
            reset_state = tf.slice(
                reset_state,
                len(context.shape.as_list()) * [0],
                tf.concat([tf.shape(obs_inputs)[:2], [1]], -1),
            )
            context = tf.slice(
                context,
                len(context.shape.as_list()) * [0],
                tf.concat([tf.shape(obs_inputs)[:2], context.shape.as_list()[2:]], -1),
            )

            # Concatenating actions (if provided) to the context at the bottom-most level.
            if level == 0 and actions is not None:
                context = tf.concat([context, actions], axis=-1)

            # Dynamic unroll of RNN cell.
            initial = self.cells[level].zero_out_state(tf.shape(obs_inputs)[0])
            if initial_state[level] is not None:
                initial["state"] = initial_state[level]
            (prior, posterior), posterior_last_step = tools.scan(
                self.cells[level],
                (obs_inputs, context, reset_state),
                use_observations[level],
                initial,
            )

            last_state_all_levels.insert(0, posterior_last_step)
            context = posterior["output"]

            prior_list.insert(0, prior)
            posterior_list.insert(0, posterior)

            # Tiling context by a factor of tmp_abs_factor for use at the level below.
            if level != 0:
                context = tf.expand_dims(context, axis=2)
                context = tf.tile(
                    context,
                    [1, 1, self._tmp_abs_factor]
                    + (len(context.shape.as_list()) - 3) * [1],
                )
                context = tf.reshape(
                    context,
                    [tf.shape(context)[0], tf.reduce_prod(tf.shape(context)[1:3])]
                    + context.shape.as_list()[3:],
                )

            reset_state = reset_state_next
        output_bot_level = context

        return output_bot_level, last_state_all_levels, prior_list, posterior_list

    def open_loop_unroll(self, inputs, ctx_len, actions=None, use_observations=None):
        if use_observations is None:
            use_observations = self._levels * [True]
        ctx_len_backup = ctx_len
        pre_inputs = []
        post_inputs = []
        for lvl in range(self._levels):
            pre_inputs.append(inputs[lvl][:, :ctx_len, ...])
            post_inputs.append(tf.zeros_like(inputs[lvl][:, ctx_len:, ...]))
            ctx_len = int(ctx_len / self._tmp_abs_factor)
        ctx_len = ctx_len_backup
        actions_pre = actions_post = None
        if actions is not None:
            actions_pre = actions[:, :ctx_len, :]
            actions_post = actions[:, ctx_len:, :]

        (
            _,
            pre_last_state_all_levels,
            pre_priors,
            pre_posteriors,
        ) = self.hierarchical_unroll(
            pre_inputs, actions=actions_pre, use_observations=use_observations
        )
        _, _, post_priors, _ = self.hierarchical_unroll(
            post_inputs,
            actions=actions_post,
            use_observations=self._levels * [False],
            initial_state=pre_last_state_all_levels,
        )

        return pre_posteriors, pre_priors, post_priors

    def _gaussian_KLD(self, dist1, dist2):
        """
        Computes KL(dist1 || dist2)

        Arguments:
            dist1 : dict containing 'mean' and 'stddev' for multivariate normal distributions
                shape of mean/stddev: (batch size, timesteps, [dim1, dim2, ...])
            dist2 : (same as dist1)
        """
        if len(dist1["mean"].shape[2:]) > 1:
            new_shape = tf.concat([tf.shape(dist1["mean"])[:2], [-1]], -1)
            dist1["mean"] = tf.reshape(dist1["mean"], new_shape)
            dist1["stddev"] = tf.reshape(dist1["stddev"], new_shape)
            dist2["mean"] = tf.reshape(dist2["mean"], new_shape)
            dist2["stddev"] = tf.reshape(dist2["stddev"], new_shape)
        mvn1 = tfd.MultivariateNormalDiag(loc=dist1["mean"], scale_diag=dist1["stddev"])
        mvn2 = tfd.MultivariateNormalDiag(loc=dist2["mean"], scale_diag=dist2["stddev"])
        return mvn1.kl_divergence(mvn2)

    def _log_prob_obs(self, samples, mean, stddev):
        """
        Returns prob density of samples in the given distribution
        The last dim of the samples is the one taken sum over.
        """
        if len(samples.shape[2:]) > 1:
            new_shape = tf.concat([tf.shape(samples)[:2], [-1]], -1)
            samples = tf.reshape(samples, new_shape)
            mean = tf.reshape(mean, new_shape)
            if isinstance(stddev, tf.Tensor):
                stddev = tf.reshape(stddev, new_shape)
        dist = tfd.Independent(tfd.Normal(mean, stddev), reinterpreted_batch_ndims=1)
        return dist.log_prob(samples)

    def _stop_grad_dist(self, dist):
        dist["mean"] = tf.stop_gradient(dist["mean"])
        dist["stddev"] = tf.stop_gradient(dist["stddev"])
        return dist

    def compute_losses(
        self,
        obs,
        obs_decoded,
        priors,
        posteriors,
        dec_stddev=0.1,
        kl_grad_post_perc=None,
        free_nats=None,
        beta=None,
    ):
        """
        Computes ELBO.

        Arguments:
            obs : Placeholder
                Observed video
            obs_decoded : Tensor
                Decoded video
            priors : list[dict]
                each dict holds the priors at all timesteps for a particular level in the model
            posteriors : list[dict]
                each dict holds the posteriors at all timesteps for a particular level in the model
        """
        nll_term = -self._log_prob_obs(obs, obs_decoded, dec_stddev)  # shape: (B,T)
        nll_term = tf.reduce_mean(tf.reduce_sum(nll_term, axis=1), 0)
        assert len(nll_term.shape) == 0, nll_term.shape

        # Computing KLs between priors and posteriors
        self.kld_all_levels = list([])
        kl_term = tf.constant(0.0)
        for i in range(self._levels):
            kld_level = self._gaussian_KLD(posteriors[i], priors[i])
            if kl_grad_post_perc is None:
                kld_level_total = tf.reduce_mean(tf.reduce_sum(kld_level, axis=1))
            else:
                # Scaling gradient between prior and posterior.
                kld_level_p = (1 - kl_grad_post_perc) * self._gaussian_KLD(
                    self._stop_grad_dist(posteriors[i]), priors[i]
                )  # shape: (B,T)
                kld_level_q = kl_grad_post_perc * self._gaussian_KLD(
                    posteriors[i], self._stop_grad_dist(priors[i])
                )  # shape: (B,T)
                kld_level_total_p = tf.reduce_mean(tf.reduce_sum(kld_level_p, axis=1))
                kld_level_total_q = tf.reduce_mean(tf.reduce_sum(kld_level_q, axis=1))
                # Giving 1 free nat to the posterior.
                kld_level_total_q = tf.maximum(1.0, kld_level_total_q)
                kld_level_total = kld_level_total_p + kld_level_total_q

            if free_nats is None:
                if beta is None:
                    kl_term += kld_level_total
                else:
                    if isinstance(beta, list):
                        kl_term += beta[i] * kld_level_total
                    else:
                        kl_term += beta * kld_level_total
            else:
                if beta is None:
                    kl_term += tf.maximum(0.0, kld_level_total - free_nats)
                else:
                    if isinstance(beta, list):
                        kl_term += beta[i] * tf.maximum(
                            0.0, kld_level_total - free_nats
                        )
                    else:
                        kl_term += beta * tf.maximum(0.0, kld_level_total - free_nats)
            self.kld_all_levels.insert(i, kld_level)

        neg_elbo = nll_term + kl_term

        num_timesteps_obs = tf.cast(tf.shape(obs)[1], tf.float32)
        self.loss = neg_elbo / num_timesteps_obs
        self._kl_term = kl_term / num_timesteps_obs
        self._nll_term = nll_term / num_timesteps_obs

        return self.loss


def build_model(cfg, open_loop=True):
    obs = tf.placeholder(tf.float32, [None, None, 64, 64, cfg.channels], name="obs")
    encoder = cnns.Encoder(
        cfg.levels,
        cfg.tmp_abs_factor,
        dense_layers=cfg.enc_dense_layers,
        embed_size=cfg.enc_dense_embed_size,
        channels_mult=cfg.channels_mult,
    )
    decoder = cnns.Decoder(cfg.channels, channels_mult=cfg.channels_mult)
    obs_encoded = encoder(obs)
    model = CWVAE(
        cfg.levels,
        cfg.tmp_abs_factor,
        dict(stoch=cfg.cell_stoch_size, deter=cfg.cell_deter_size),
        cfg.cell_embed_size,
        cfg.cell_type,
        cfg.lr,
        cfg.cell_min_stddev,
        mean_only_cell=cfg.cell_mean_only,
        reset_states=cfg.cell_reset_state,
    )
    outputs_bot, _, priors, posteriors = model.hierarchical_unroll(obs_encoded)
    obs_decoded = decoder(outputs_bot)
    loss = model.compute_losses(
        obs,
        obs_decoded,
        priors,
        posteriors,
        dec_stddev=cfg.dec_stddev,
        kl_grad_post_perc=cfg.kl_grad_post_perc,
        free_nats=cfg.free_nats,
        beta=cfg.beta,
    )
    out = {
        "training": {
            "obs": obs,
            "encoder": encoder,
            "decoder": decoder,
            "obs_encoded": obs_encoded,
            "obs_decoded": obs_decoded,
            "priors": priors,
            "posteriors": posteriors,
            "loss": loss,
        },
        "meta": {"model": model},
    }
    if open_loop:
        posteriors_recon, priors_onestep, priors_multistep = model.open_loop_unroll(
            obs_encoded, cfg.open_loop_ctx, use_observations=cfg.use_obs
        )
        obs_decoded_posterior_recon = decoder(posteriors_recon[0]["output"])
        obs_decoded_prior_onestep = decoder(priors_onestep[0]["output"])
        obs_decoded_prior_multistep = decoder(priors_multistep[0]["output"])
        gt_multistep = obs[:, cfg.open_loop_ctx :, ...]
        out.update(
            {
                "open_loop_obs_decoded": {
                    "posterior_recon": obs_decoded_posterior_recon,
                    "prior_onestep": obs_decoded_prior_onestep,
                    "prior_multistep": obs_decoded_prior_multistep,
                    "gt_multistep": gt_multistep,
                }
            }
        )
    return out
