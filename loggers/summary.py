import os

import tensorflow.compat.v1 as tf
from tensorflow_probability import distributions as tfd

import loggers.gif_summary


class Summary:
    def __init__(self, log_dir_root, save_gifs=True, var_scope="cwvae"):
        self._log_dir_root = log_dir_root
        self._log_dir_train = os.path.join(self._log_dir_root, "train")
        self._log_dir_val = os.path.join(self._log_dir_root, "val")
        os.makedirs(self._log_dir_train, exist_ok=True)
        os.makedirs(self._log_dir_val, exist_ok=True)
        self._writer_train = tf.summary.FileWriter(self._log_dir_train)
        self._writer_val = tf.summary.FileWriter(self._log_dir_val)

        self._save_gifs = save_gifs
        self._var_scope = var_scope
        self._summaries = []

        self.scalar_summary = None
        self.gif_summary = None

    def build_summary(self, cfg, model_components, **kwargs):
        assert self.scalar_summary is None, "Can only call self.scalar_summary once."

        with tf.name_scope(self._var_scope):
            model = model_components["meta"]["model"]
            self._summaries.append(tf.summary.scalar("total_loss", model.loss))
            self._summaries.append(tf.summary.scalar("nll_term", model._nll_term))
            self._summaries.append(tf.summary.scalar("kl_term", model._kl_term))
            self._summaries.append(tf.summary.scalar("learning_rate", cfg.lr))
            self._summaries.append(tf.summary.scalar("grad_norm", kwargs["grad_norm"]))

            # Adding per-level summaries.
            for lvl in range(cfg.levels):
                # KL(posterior || prior) at each level (avg across batch, sum across time).
                kl_mean = tf.reduce_mean(
                    tf.reduce_sum(model.kld_all_levels[lvl], axis=1)
                )
                self._summaries.append(
                    tf.summary.scalar(
                        "avg_kl_prior_posterior__level_" + str(lvl), kl_mean
                    )
                )

                # Prior entropy.
                prior = model_components["training"]["priors"][lvl]
                prior_entropy_mean = tf.reduce_mean(
                    tf.reduce_sum(
                        tfd.MultivariateNormalDiag(
                            prior["mean"], prior["stddev"]
                        ).entropy(),
                        axis=1,
                    )
                )
                self._summaries.append(
                    tf.summary.scalar(
                        "avg_entropy_prior__level_" + str(lvl), prior_entropy_mean
                    )
                )

                # Posterior entropy.
                posterior = model_components["training"]["posteriors"][lvl]
                posterior_entropy_mean = tf.reduce_mean(
                    tf.reduce_sum(
                        tfd.MultivariateNormalDiag(
                            posterior["mean"], posterior["stddev"]
                        ).entropy(),
                        axis=1,
                    )
                )
                self._summaries.append(
                    tf.summary.scalar(
                        "avg_entropy_posterior__level_" + str(lvl),
                        posterior_entropy_mean,
                    )
                )

            self.scalar_summary = tf.summary.merge(self._summaries)

            if self._save_gifs:
                self.gif_summary = loggers.gif_summary.image_summaries(
                    tfd.Normal(
                        model_components["open_loop_obs_decoded"]["prior_multistep"],
                        cfg.dec_stddev,
                    ),
                    model_components["open_loop_obs_decoded"]["gt_multistep"],
                    clip_by=[0.0, 1.0],
                    name=self._var_scope,
                    max_batch=8,
                )

    def save(self, summary, step, train):
        """
        Arguments:
            summary
                Obtained after a sess.run on self.scalar_summary or self.gif_summary.
            step
            train : bool
                Determines which summary writer will be used.
        """
        if train:
            self._writer_train.add_summary(summary, step)
        else:
            self._writer_val.add_summary(summary, step)
