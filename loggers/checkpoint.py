import os

import tensorflow.compat.v1 as tf


class Checkpoint:
    def __init__(self, log_dir_root):
        self._log_dir_root = log_dir_root
        self.log_dir_model = os.path.join(self._log_dir_root, "model")
        self._ckpt_name = "model.ckpt"

        self._saver = tf.train.Saver()

    def save(self, sess, save_dir=None):
        if save_dir is None:
            os.makedirs(self.log_dir_model, exist_ok=True)
            save_path = os.path.join(self.log_dir_model, self._ckpt_name)
        else:
            os.makedirs(os.path.join(self._log_dir_root, save_dir), exist_ok=True)
            save_path = os.path.join(self._log_dir_root, save_dir, self._ckpt_name)
        self._saver.save(sess, save_path)

    def restore(self, sess, restore_dir=None):
        if restore_dir is None:
            restore_path = os.path.join(self.log_dir_model, self._ckpt_name)
        else:
            restore_path = os.path.join(
                self._log_dir_root, restore_dir, self._ckpt_name
            )
        self._saver.restore(sess, restore_path)
