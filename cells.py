import tensorflow.compat.v1 as tf
from tensorflow.keras import layers as tfkl
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import numpy as np
from collections import namedtuple

import tools


class RSSMCell(tools.Module):
    def __init__(
        self,
        state_size,
        detstate_size,
        embed_size,
        reset_states=False,
        reuse=None,
        min_stddev=0.0001,
        mean_only=False,
        var_scope="rssm_cell",
    ):
        super().__init__()
        self._state_size = state_size
        self._detstate_size = detstate_size
        self._embed_size = embed_size
        self._min_stddev = min_stddev
        self._mean_only = mean_only
        self._reset_states = reset_states  # whether or not to reset states as per the reset_state tensor passed to __call__() (VTA-like behavior)
        self._var_scope = var_scope

        with tf.name_scope(self._var_scope):
            self._cell = tfkl.GRUCell(units=self._detstate_size)

    def _prior(self, prev_state, context):
        with tf.name_scope(self._var_scope):
            inputs = tf.concat([prev_state["sample"], context], -1)
            hl = self.get(
                "prior_h1_dense", tfkl.Dense, self._embed_size, activation=tf.nn.relu
            )(inputs)
            det_out, det_state = self._cell(hl, (prev_state["det_state"],))
            det_state = det_state[0]
            hl = det_out
            hl = self.get(
                "prior_h2_dense", tfkl.Dense, self._embed_size, activation=tf.nn.relu
            )(hl)
            mean = self.get(
                "prior_mean_dense", tfkl.Dense, self._state_size, activation=None
            )(hl)
            stddev = self.get(
                "prior_stddev_dense",
                tfkl.Dense,
                self._state_size,
                activation=tf.nn.softplus,
            )(hl + 0.54)
            stddev += self._min_stddev
            if self._mean_only:
                sample = mean
            else:
                sample = tfd.MultivariateNormalDiag(mean, stddev).sample()
        return {
            "mean": mean,
            "stddev": stddev,
            "sample": sample,
            "det_out": det_out,
            "det_state": det_state,
            "output": tf.concat([sample, det_out], -1),
        }

    def _posterior(self, obs_inputs, prev_state, context):
        with tf.name_scope(self._var_scope):
            prior = self._prior(prev_state, context)
            inputs = tf.concat([prior["det_out"], obs_inputs], -1)
            hl = self.get(
                "posterior_h1_dense",
                tfkl.Dense,
                self._embed_size,
                activation=tf.nn.relu,
            )(inputs)
            hl = self.get(
                "posterior_h2_dense",
                tfkl.Dense,
                self._embed_size,
                activation=tf.nn.relu,
            )(hl)
            mean = self.get(
                "posterior_mean_dense", tfkl.Dense, self._state_size, activation=None
            )(hl)
            stddev = self.get(
                "posterior_stddev_dense",
                tfkl.Dense,
                self._state_size,
                activation=tf.nn.softplus,
            )(hl + 0.54)
            stddev += self._min_stddev
            if self._mean_only:
                sample = mean
            else:
                sample = tfd.MultivariateNormalDiag(mean, stddev).sample()
        return {
            "mean": mean,
            "stddev": stddev,
            "sample": sample,
            "det_out": prior["det_out"],
            "det_state": prior["det_state"],
            "output": tf.concat([sample, prior["det_out"]], -1),
        }

    @property
    def state_size(self):
        return {
            "mean": self._state_size,
            "stddev": self._state_size,
            "sample": self._state_size,
            "det_out": self._detstate_size,
            "det_state": self._detstate_size,
            "output": self._state_size + self._detstate_size,
        }

    @property
    def out_state_size(self):
        return {"out": (self.state_size, self.state_size), "state": self.state_size}

    def zero_state(self, batch_size, dtype=tf.float32):
        return dict(
            [
                (k, tf.zeros([batch_size, v], dtype=dtype))
                for k, v in self.state_size.items()
            ]
        )

    def zero_out_state(self, batch_size, dtype=tf.float32):
        zero_st = self.zero_state(batch_size, dtype)
        return {"out": (zero_st, zero_st), "state": zero_st}

    def __call__(self, prev_out, inputs, use_obs):
        """
        Arguments:
            prev_out : dict
                output of this __call__ at the previous time-step during unroll.
            inputs : dict
                dict of context and other inputs (including observations).
                obs_input will remain unused during test phase when the posterior is not computed.
            use_obs : bool
        Returns:
            dict
                'out': (prior, posterior) --> cell out
                'state': (posterior) --> cell state
        """
        prev_state = prev_out["state"]
        obs_input, context, reset_state = inputs
        if not self._reset_states:
            reset_state = tf.ones_like(reset_state)
        prev_state["sample"] = tf.multiply(prev_state["sample"], reset_state)

        prior = self._prior(prev_state, context)
        if use_obs:
            posterior = self._posterior(obs_input, prev_state, context)
        else:
            posterior = prior

        return {"out": (prior, posterior), "state": posterior}
