import yaml
from pathlib import Path
import numpy as np
import imageio
import os
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf


class AttrDict(dict):

    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


class Module(tf.Module):
    def __init__(self):
        super().__init__()
        self._modules = {}

    def get(self, name, Tf_layer, *args, **kwargs):
        if name not in self._modules.keys():
            self._modules[name] = Tf_layer(*args, **kwargs)
        return self._modules[name]


class Step:
    def __init__(self, session):
        self._session = session
        self._step = tf.get_variable("step", initializer=lambda: 0, trainable=False)

    def increment(self):
        self._session.run(self._step.assign_add(1))

    def __call__(self):
        return self._session.run(self._step)


def exp_name(cfg, model_dir_prefix=None):
    exp_name = "{}_cwvae_{}".format(cfg.dataset, cfg.cell_type.lower())
    exp_name += "_{}l_f{}".format(cfg.levels, cfg.tmp_abs_factor)
    exp_name += "_decsd{}".format(cfg.dec_stddev)
    exp_name += "_enchl{}_ences{}_edchnlmult{}".format(
        cfg.enc_dense_layers, cfg.enc_dense_embed_size, cfg.channels_mult
    )
    exp_name += "_ss{}_ds{}_es{}".format(
        cfg.cell_stoch_size, cfg.cell_deter_size, cfg.cell_embed_size
    )
    exp_name += "_seq{}_lr{}_bs{}".format(cfg.seq_len, cfg.lr, cfg.batch_size)
    return exp_name


def validate_config(cfg):
    assert (
        cfg.channels is not None and cfg.channels > 0
    ), "Incompatible channels = {} found in config.".format(cfg.config)
    assert (
        cfg.open_loop_ctx % (cfg.tmp_abs_factor ** (cfg.levels - 1)) == 0
    ), "Incompatible open-loop context length {} and temporal abstraction factor {} for levels {}".format(
        cfg.open_loop_ctx, cfg.tmp_abs_factor, cfg.levels
    )
    assert cfg.datadir is not None, "data root directory cannot be None."
    assert cfg.logdir is not None, "log root directory cannot be None."


def read_configs(config_path, base_config_path=None, **kwargs):
    if base_config_path is not None:
        base_config = yaml.load(Path(base_config_path).read_text())
        config = base_config.copy()
        config.update(yaml.load(Path(config_path).read_text()))
        assert (
            len(set(config).difference(base_config)) == 0
        ), "Found new keys in config. Make sure to set them in base_config first."
    else:
        with open(config_path, "r") as f:
            config = yaml.load(f)
    config = AttrDict(config)

    if kwargs.get("datadir", None) is not None:
        config.datadir = kwargs["datadir"]
    if kwargs.get("logdir", None) is not None:
        config.logdir = kwargs["logdir"]

    validate_config(config)
    return config


def scan(cell, inputs, use_obs, initial):
    assert initial is not None, "initial cannot be None. Pass zero_state instead."
    # Transpose inputs to (T, B, ...)
    inputs = tf.nest.map_structure(
        lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape.as_list())))),
        inputs,
    )
    outputs = tf.scan(
        lambda agg, inp: cell(agg, inp, use_obs=use_obs), inputs, initializer=initial
    )
    # Transpose outputs back to (B, T, ...)
    outputs = tf.nest.map_structure(
        lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape.as_list())))),
        outputs,
    )
    return outputs["out"], tf.nest.map_structure(
        lambda x: x[:, -1, ...], outputs["state"]
    )


def _to_padded_strip(images):
    if len(images.shape) <= 3:
        images = np.expand_dims(images, -1)
    c_dim = images.shape[-1]
    x_dim = images.shape[-3]
    y_dim = images.shape[-2]
    padding = 1
    result = np.zeros(
        (x_dim, y_dim * images.shape[0] + padding * (images.shape[0] - 1), c_dim),
        dtype=np.uint8,
    )
    for i in range(images.shape[0]):
        result[:, i * y_dim + i * padding : (i + 1) * y_dim + i * padding, :] = images[
            i
        ]
    if result.shape[-1] == 1:
        result = np.reshape(result, result.shape[:2])
    return result


def save_as_grid(images, save_dir, filename, strip_width=50):
    # Creating a grid of images.
    # images shape: (T, ...)
    results = list([])
    if images.shape[0] < strip_width:
        results.append(_to_padded_strip(images))
    else:
        for i in range(0, images.shape[0], strip_width):
            if i + strip_width < images.shape[0]:
                results.append(_to_padded_strip(images[i : i + strip_width]))
    grid = np.concatenate(results, 0)
    imageio.imwrite(os.path.join(save_dir, filename), grid)
    print("Written grid file {}".format(os.path.join(save_dir, filename)))


def compute_metrics(gt, pred):
    gt = np.transpose(gt, [0, 1, 4, 2, 3])
    pred = np.transpose(pred, [0, 1, 4, 2, 3])
    bs = gt.shape[0]
    T = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            for c in range(gt[i][t].shape[0]):
                ssim[i, t] += ssim_metric(gt[i][t][c], pred[i][t][c])
                psnr[i, t] += psnr_metric(gt[i][t][c], pred[i][t][c])
            ssim[i, t] /= gt[i][t].shape[0]
            psnr[i, t] /= gt[i][t].shape[0]

    return ssim, psnr


def plot_metrics(metrics, logdir, name):
    mean_metric = np.squeeze(np.mean(metrics, 0))
    stddev_metric = np.squeeze(np.std(metrics, 0))
    np.savez(os.path.join(logdir, "{}_mean.npz".format(name)), mean_metric)
    np.savez(os.path.join(logdir, "{}_stddev.npz".format(name)), stddev_metric)

    plt.figure()
    axes = plt.gca()
    axes.yaxis.grid(True)
    plt.plot(mean_metric, color="blue")
    axes.fill_between(
        np.arange(0, mean_metric.shape[0]),
        mean_metric - stddev_metric,
        mean_metric + stddev_metric,
        color="blue",
        alpha=0.4,
    )
    plt.savefig(os.path.join(logdir, "{}_range.png".format(name)))
