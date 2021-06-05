import tensorflow.compat.v1 as tf
from tensorflow.keras import layers as tfkl
import numpy as np

import tools


class Encoder(tools.Module):
    """
    Multi-level Video Encoder.
    1. Extracts hierarchical features from a sequence of observations.
    2. Encodes observations using Conv layers, uses them directly for the bottom-most level.
    3. Uses dense features for each level of the hierarchy above the bottom-most level.
    """

    def __init__(
        self,
        levels,
        tmp_abs_factor,
        dense_layers=3,
        embed_size=100,
        channels_mult=1,
        var_scope="encoder_convdense",
    ):
        """
        Arguments:
            obs : Tensor
                Flattened/Non-flattened observations of shape (batch size, timesteps, [dim(s)])
            levels : int
                Number of levels in the hierarchy
            tmp_abs_factor : int
                Temporal abstraction factor used at each level
            dense_layers : int
                Number of dense hidden layers at each level
            embed_size : int
                Size of dense hidden embeddings
            channels_mult: int
                Multiplier for the number of channels in the conv encoder
        """
        super().__init__()
        self._levels = levels
        self._tmp_abs_factor = tmp_abs_factor
        self._dense_layers = dense_layers
        self._embed_size = embed_size
        self._channels_mult = channels_mult
        self._activation = tf.nn.leaky_relu
        self._kwargs = dict(strides=2, activation=self._activation, use_bias=True)
        self._var_scope = var_scope

        assert levels >= 1, "levels should be >=1, found {}".format(levels)
        assert tmp_abs_factor >= 1, "tmp_abs_factor should be >=1, found {}".format(
            tmp_abs_factor
        )
        assert (
            not dense_layers or embed_size
        ), "embed_size={} invalid for Dense layer".format(embed_size)

    def __call__(self, obs):
        """
        Arguments:
            obs : Tensor
                Un-flattened observations (videos) of shape (batch size, timesteps, [dim1, dim2, dim3])
        """
        with tf.name_scope(self._var_scope):
            # Squeezing batch and time dimensions.
            hidden = tf.reshape(obs, tf.concat([[-1], tf.shape(obs)[2:]], -1))

            filters = 32
            hidden = self.get(
                "h1_conv", tfkl.Conv2D, self._channels_mult * filters, 4, **self._kwargs
            )(hidden)
            hidden = self.get(
                "h2_conv",
                tfkl.Conv2D,
                self._channels_mult * filters * 2,
                4,
                **self._kwargs
            )(hidden)
            hidden = self.get(
                "h3_conv",
                tfkl.Conv2D,
                self._channels_mult * filters * 4,
                4,
                **self._kwargs
            )(hidden)
            hidden = self.get(
                "h4_conv",
                tfkl.Conv2D,
                self._channels_mult * filters * 8,
                4,
                **self._kwargs
            )(hidden)
            hidden = tf.layers.flatten(hidden)  # shape: (BxT, :)
            hidden = tf.reshape(
                hidden, tf.concat([tf.shape(obs)[:2], [hidden.shape.as_list()[-1]]], -1)
            )  # shape: (B, T, :)
            layer = hidden

            layers = list([])
            layers.append(layer)
            print("Input shape at level {}: {}".format(0, layer.shape))

            feat_size = layer.shape[-1]

            for level in range(1, self._levels):
                for i_dl in range(self._dense_layers - 1):
                    hidden = self.get(
                        "h{}_dense".format(5 + (level - 1) * self._dense_layers + i_dl),
                        tfkl.Dense,
                        self._embed_size,
                        activation=tf.nn.relu,
                    )(hidden)
                if self._dense_layers > 0:
                    hidden = self.get(
                        "h{}_dense".format(4 + level * self._dense_layers),
                        tfkl.Dense,
                        feat_size,
                        activation=None,
                    )(hidden)
                layer = hidden

                timesteps_to_merge = np.power(self._tmp_abs_factor, level)
                # Padding the time dimension.
                timesteps_to_pad = tf.mod(
                    timesteps_to_merge - tf.mod(tf.shape(layer)[1], timesteps_to_merge),
                    timesteps_to_merge,
                )
                paddings = tf.convert_to_tensor([[0, 0], [0, timesteps_to_pad], [0, 0]])
                layer = tf.pad(layer, paddings, mode="CONSTANT", constant_values=0)
                # Reshaping and merging in time.
                layer = tf.reshape(
                    layer,
                    [
                        tf.shape(layer)[0],
                        -1,
                        timesteps_to_merge,
                        layer.shape.as_list()[2],
                    ],
                )
                layer = tf.reduce_sum(layer, axis=2)
                layers.append(layer)
                print("Input shape at level {}: {}".format(level, layer.shape))

        return layers


class Decoder(tools.Module):
    """ States to Images Decoder. """

    def __init__(self, out_channels, channels_mult=1, var_scope="decoder_conv"):
        """
        Arguments:
            out_channels : int
                Number of channels in the output video
            channels_mult : int
                Multiplier for the number of channels in the conv encoder
        """
        super().__init__()
        self._out_channels = out_channels
        self._channels_mult = channels_mult
        self._out_activation = tf.nn.tanh
        self._kwargs = dict(strides=2, activation=tf.nn.leaky_relu, use_bias=True)
        self._var_scope = var_scope

    def __call__(self, states):
        """
        Arguments:
            states : Tensor
                State tensor of shape (batch_size, timesteps, feature_dim)

        Returns:
            out : Tensor
                Output video of shape (batch_size, timesteps, 64, 64, out_channels)
        """
        with tf.name_scope(self._var_scope):
            hidden = self.get("h1", tfkl.Dense, self._channels_mult * 1024, None)(
                states
            )  # (B, T, 1024)

            # Squeezing batch and time dimensions, and expanding two extra dims.
            hidden = tf.reshape(
                hidden, [-1, 1, 1, hidden.shape[-1].value]
            )  # (BxT, 1, 1, 1024)

            filters = 32
            hidden = self.get(
                "h2",
                tfkl.Conv2DTranspose,
                self._channels_mult * filters * 4,
                5,
                **self._kwargs
            )(
                hidden
            )  # (BxT, 5, 5, 128)
            hidden = self.get(
                "h3",
                tfkl.Conv2DTranspose,
                self._channels_mult * filters * 2,
                5,
                **self._kwargs
            )(
                hidden
            )  # (BxT, 13, 13, 64)
            hidden = self.get(
                "h4",
                tfkl.Conv2DTranspose,
                self._channels_mult * filters,
                6,
                **self._kwargs
            )(
                hidden
            )  # (BxT, 30, 30, 32)
            out = self.get(
                "out",
                tfkl.Conv2DTranspose,
                self._out_channels,
                6,
                strides=2,
                activation=self._out_activation,
            )(
                hidden
            )  # (BxT, 64, 64, out_channels)
            out = tf.reshape(
                out, tf.concat([tf.shape(states)[:2], tf.shape(out)[1:]], -1)
            )  # (B, T, 64, 64, out_channels)
        return out
