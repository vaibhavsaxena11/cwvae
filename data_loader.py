import numpy as np

import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds


class MineRL:
    def __init__(self, batch_size, epochs, train=True, seq_len=None, data_root=None):
        self._train = train
        self._batch_size = batch_size
        self._epochs = epochs
        self._data_seq_len = 500
        self._seq_len = seq_len
        if self._train:
            ds = tfds.load("minerl_navigate", data_dir=data_root, shuffle_files=True)[
                "train"
            ]
        else:
            ds = tfds.load("minerl_navigate", data_dir=data_root, shuffle_files=False)[
                "test"
            ]
        ds = ds.map(lambda vid: vid["video"]).flat_map(
            lambda x: tf.data.Dataset.from_tensor_slices(self._process_seq(x))
        )
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        ds = ds.repeat(self._epochs)
        if self._train:
            ds = ds.shuffle(10 * self._batch_size)
        ds = ds.batch(self._batch_size)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        self.batch = tf.data.make_one_shot_iterator(ds).get_next()

    def get_batch(self):
        return self.batch

    def _process_seq(self, seq):
        if self._seq_len:
            seq_len_tr = self._data_seq_len - (self._data_seq_len % self._seq_len)
            seq = seq[:seq_len_tr]
            seq = tf.reshape(
                seq,
                tf.concat(
                    [[seq_len_tr // self._seq_len, self._seq_len], tf.shape(seq)[1:]],
                    -1,
                ),
            )
        else:
            seq = tf.expand_dims(seq, 0)
        seq = tf.cast(seq, tf.float32) / 255.0
        return seq


class GQNMazes:
    def __init__(self, batch_size, epochs, train=True, seq_len=None, data_root=None):
        self._train = train
        self._batch_size = batch_size
        self._epochs = epochs
        self._data_seq_len = 300
        self._seq_len = seq_len
        if self._train:
            ds = tfds.load("gqn_mazes", data_dir=data_root, shuffle_files=True)["train"]
        else:
            ds = tfds.load("gqn_mazes", data_dir=data_root, shuffle_files=False)["test"]
        ds = ds.map(lambda vid: vid["video"]).flat_map(
            lambda x: tf.data.Dataset.from_tensor_slices(self._process_seq(x))
        )
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        ds = ds.repeat(self._epochs)
        if self._train:
            ds = ds.shuffle(10 * self._batch_size)
        ds = ds.batch(self._batch_size)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        self.batch = tf.data.make_one_shot_iterator(ds).get_next()

    def get_batch(self):
        return self.batch

    def _process_seq(self, seq):
        if self._seq_len:
            seq_len_tr = self._data_seq_len - (self._data_seq_len % self._seq_len)
            seq = seq[:seq_len_tr]
            seq = tf.reshape(
                seq,
                tf.concat(
                    [[seq_len_tr // self._seq_len, self._seq_len], tf.shape(seq)[1:]],
                    -1,
                ),
            )
        else:
            seq = tf.expand_dims(seq, 0)
        seq = tf.cast(seq, tf.float32) / 255.0
        return seq


class MovingMNIST:
    def __init__(self, batch_size, epochs, train=True, seq_len=None, data_root=None):
        self._train = train
        self._batch_size = batch_size
        self._epochs = epochs
        if self._train:
            self._data_seq_len = 100
        else:
            self._data_seq_len = 1000
        self._seq_len = seq_len
        if self._train:
            ds = tfds.load(
                "moving_mnist_2digit", data_dir=data_root, shuffle_files=True
            )["train"]
        else:
            ds = tfds.load(
                "moving_mnist_2digit", data_dir=data_root, shuffle_files=False
            )["test"]
        ds = ds.map(lambda vid: vid["video"]).flat_map(
            lambda x: tf.data.Dataset.from_tensor_slices(self._process_seq(x))
        )
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        ds = ds.repeat(self._epochs)
        if self._train:
            ds = ds.shuffle(10 * self._batch_size)
        ds = ds.batch(self._batch_size)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        self.batch = tf.data.make_one_shot_iterator(ds).get_next()

    def get_batch(self):
        return self.batch

    def _process_seq(self, seq):
        if self._seq_len:
            seq_len_tr = self._data_seq_len - (self._data_seq_len % self._seq_len)
            seq = seq[:seq_len_tr]
            seq = tf.reshape(
                seq,
                tf.concat(
                    [[seq_len_tr // self._seq_len, self._seq_len], tf.shape(seq)[1:]],
                    -1,
                ),
            )
        else:
            seq = tf.expand_dims(seq, 0)
        seq = tf.cast(seq, tf.float32) / 255.0
        return seq


def load_dataset(cfg, **kwargs):
    if cfg.dataset == "minerl":
        import minerl_navigate

        train_data_batch = MineRL(
            cfg.batch_size,
            cfg.num_epochs,
            train=True,
            seq_len=cfg.seq_len,
            data_root=cfg.datadir,
        ).get_batch()
        test_data_batch = MineRL(
            cfg.batch_size,
            1,
            train=False,
            seq_len=cfg.eval_seq_len,
            data_root=cfg.datadir,
        ).get_batch()
    elif cfg.dataset == "mmnist":
        import datasets.moving_mnist

        train_data_batch = MovingMNIST(
            cfg.batch_size,
            cfg.num_epochs,
            train=True,
            seq_len=cfg.seq_len,
            data_root=cfg.datadir,
        ).get_batch()
        test_data_batch = MovingMNIST(
            cfg.batch_size,
            1,
            train=False,
            seq_len=cfg.eval_seq_len,
            data_root=cfg.datadir,
        ).get_batch()
    elif cfg.dataset == "mazes":
        import datasets.gqn_mazes

        train_data_batch = GQNMazes(
            cfg.batch_size,
            cfg.num_epochs,
            train=True,
            seq_len=cfg.seq_len,
            data_root=cfg.datadir,
        ).get_batch()
        test_data_batch = GQNMazes(
            cfg.batch_size,
            1,
            train=False,
            seq_len=cfg.eval_seq_len,
            data_root=cfg.datadir,
        ).get_batch()

    else:
        raise ValueError("Dataset {} not supported.".format(cfg.dataset))
    return train_data_batch, test_data_batch


def get_multiple_batches(batch_op, num_batches, sess):
    batches = []
    for _ in range(num_batches):
        batches.append(sess.run(batch_op))
    batches = np.concatenate(batches, 0)
    return batches


def get_single_batch(batch_op, sess):
    return sess.run(batch_op)
