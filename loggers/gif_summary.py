import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.ops import summary_op_util


def encode_gif(images, fps):
    """Encodes numpy images into gif string.
    Args:
        images: A 5-D `uint8` `np.array` (or a list of 4-D images) of shape
            `[batch_size, time, height, width, channels]` where `channels` is 1 or 3.
        fps: frames per second of the animation
    Returns:
        The encoded gif string.
    Raises:
        IOError: If the ffmpeg command returns an error.
    """
    from subprocess import Popen, PIPE

    h, w, c = images[0].shape
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-r",
        "%.02f" % fps,
        "-s",
        "%dx%d" % (w, h),
        "-pix_fmt",
        {1: "gray", 3: "rgb24"}[c],
        "-i",
        "-",
        "-filter_complex",
        "[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse",
        "-r",
        "%.02f" % fps,
        "-f",
        "gif",
        "-",
    ]
    proc = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    for image in images:
        proc.stdin.write(image.tostring())
    out, err = proc.communicate()
    if proc.returncode:
        err = "\n".join([" ".join(cmd), err.decode("utf8")])
        raise IOError(err)
    del proc
    return out


def py_gif_summary(tag, images, max_outputs, fps):
    """Outputs a `Summary` protocol buffer with gif animations.
    Args:
        tag: Name of the summary.
        images: A 5-D `uint8` `np.array` of shape `[batch_size, time, height,
            width, channels]` where `channels` is 1 or 3.
        max_outputs: Max number of batch elements to generate gifs for.
        fps: frames per second of the animation
    Returns:
        The serialized `Summary` protocol buffer.
    Raises:
        ValueError: If `images` is not a 5-D `uint8` array with 1 or 3 channels.
    """
    is_bytes = isinstance(tag, bytes)
    if is_bytes:
        tag = tag.decode("utf-8")
    images = np.asarray(images)
    if images.dtype != np.uint8:
        raise ValueError("Tensor must have dtype uint8 for gif summary.")
    if images.ndim != 5:
        raise ValueError("Tensor must be 5-D for gif summary.")
    batch_size, _, height, width, channels = images.shape
    if channels not in (1, 3):
        raise ValueError("Tensors must have 1 or 3 channels for gif summary.")
    summ = tf.Summary()
    num_outputs = min(batch_size, max_outputs)
    for i in range(num_outputs):
        image_summ = tf.Summary.Image()
        image_summ.height = height
        image_summ.width = width
        image_summ.colorspace = channels  # 1: grayscale, 3: RGB
        try:
            image_summ.encoded_image_string = encode_gif(images[i], fps)
        except (IOError, OSError) as e:
            tf.logging.warning(
                "Unable to encode images to a gif string because either ffmpeg is "
                "not installed or ffmpeg returned an error: %s. Falling back to an "
                "image summary of the first frame in the sequence.",
                e,
            )
            try:
                from PIL import Image  # pylint: disable=g-import-not-at-top
                import io  # pylint: disable=g-import-not-at-top

                with io.BytesIO() as output:
                    Image.fromarray(images[i][0]).save(output, "PNG")
                    image_summ.encoded_image_string = output.getvalue()
            except Exception:
                tf.logging.warning(
                    "Gif summaries requires ffmpeg or PIL to be installed: %s", e
                )
                image_summ.encoded_image_string = "".encode("utf-8") if is_bytes else ""
        if num_outputs == 1:
            summ_tag = f"{tag}/gif"
        else:
            summ_tag = f"{tag}/gif/{i}"
        summ.value.add(tag=summ_tag, image=image_summ)
    summ_str = summ.SerializeToString()
    return summ_str


def gif_summary(name, tensor, max_outputs, fps, collections=None, family=None):
    """Outputs a `Summary` protocol buffer with gif animations.
    Args:
        name: Name of the summary.
        tensor: A 5-D `uint8` `Tensor` of shape `[batch_size, time, height, width,
            channels]` where `channels` is 1 or 3.
        max_outputs: Max number of batch elements to generate gifs for.
        fps: frames per second of the animation
        collections: Optional list of tf.GraphKeys. The collections to add the
            summary to. Defaults to [tf.GraphKeys.SUMMARIES]
        family: Optional; if provided, used as the prefix of the summary tag name,
            which controls the tab name used for display on Tensorboard.
    Returns:
        A scalar `Tensor` of type `string`. The serialized `Summary` protocol
        buffer.
    """
    tensor = tf.convert_to_tensor(tensor)
    if tensor.dtype in (tf.float32, tf.float64):
        tensor = tf.cast(255.0 * tensor, tf.uint8)
    with summary_op_util.summary_scope(name, family, values=[tensor]) as (tag, scope):
        val = tf.py_func(
            py_gif_summary,
            [tag, tensor, max_outputs, fps],
            tf.string,
            stateful=False,
            name=scope,
        )
        summary_op_util.collect(val, collections, [tf.GraphKeys.SUMMARIES])
    return val


def image_summaries(dist, target, name="pred_gif", max_batch=6, clip_by=[0.0, 1.0]):
    # Inputs are expected to be (batch, time, width, height, channels).
    image = tf.clip_by_value(dist.mode()[:max_batch], clip_by[0], clip_by[1])

    image = (image - clip_by[0]) / (clip_by[1] - clip_by[0])
    target = (target - clip_by[0]) / (clip_by[1] - clip_by[0])

    target = target[:max_batch]
    error = ((image - target) + 1) / 2
    # Concat ground truth, prediction, and error vertically.
    frames = tf.concat([target, image, error], 2)
    # Concat batch entries horizontally.
    frames = tf.transpose(frames, [1, 2, 0, 3, 4])
    s = shape(frames)
    frames = tf.reshape(frames, [s[0], s[1], s[2] * s[3], s[4]])
    return gif_summary(name, frames[None], max_outputs=1, fps=20, family="gifs")


def shape(tensor):
    static = tensor.get_shape().as_list()
    dynamic = tf.unstack(tf.shape(tensor))
    assert len(static) == len(dynamic)
    combined = [d if s is None else s for s, d in zip(static, dynamic)]
    return combined
