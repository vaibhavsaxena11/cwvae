import argparse
import pathlib
import os
from datetime import datetime
import numpy as np

from cwvae import build_model
from data_loader import *
import tools
from loggers.checkpoint import Checkpoint


if __name__ == "__main__":
    tf.disable_v2_behavior()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir",
        default=None,
        type=str,
        required=True,
        help="path to dir containing model checkpoints (with config in the parent dir)",
    )
    parser.add_argument(
        "--num-examples", default=100, type=int, help="number of examples to eval on"
    )
    parser.add_argument(
        "--eval-seq-len",
        default=None,
        type=int,
        help="total length of evaluation sequences",
    )
    parser.add_argument("--datadir", default=None, type=str)
    parser.add_argument(
        "--num-samples", default=1, type=int, help="samples to generate per example"
    )
    parser.add_argument(
        "--open-loop-ctx", default=36, type=int, help="number of context frames"
    )
    parser.add_argument(
        "--use-obs",
        default=None,
        type=str,
        help="string of T/Fs per level, e.g. TTF to skip obs at the top level",
    )
    parser.add_argument(
        "--no-save-grid",
        action="store_true",
        default=False,
        help="to prevent saving grids of images",
    )

    args = parser.parse_args()

    assert os.path.exists(args.logdir)

    # Set directories.
    exp_rootdir = str(pathlib.Path(args.logdir).resolve().parent)
    eval_logdir = os.path.join(
        exp_rootdir, "eval_{}".format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    )
    os.makedirs(eval_logdir, exist_ok=True)

    # Load config.
    cfg = tools.read_configs(os.path.join(exp_rootdir, "config.yml"))
    cfg.batch_size = 1
    cfg.open_loop_ctx = args.open_loop_ctx
    if args.eval_seq_len is not None:
        cfg.eval_seq_len = args.eval_seq_len
    if args.datadir:
        cfg.datadir = args.datadir
    if args.use_obs is not None:
        assert len(args.use_obs) == cfg.levels
        args.use_obs = args.use_obs.upper()
        cfg.use_obs = [dict(T=True, F=False)[c] for c in args.use_obs]
    else:
        cfg.use_obs = True
    tools.validate_config(cfg)

    # Load dataset.
    _, val_data_batch = load_dataset(cfg)

    # Define session
    session_config = tf.ConfigProto(device_count={"GPU": 1}, log_device_placement=False)
    session = tf.Session(config=session_config)

    # Build model.
    model_components = build_model(cfg)

    # Define checkpoint saver for variables currently in session.
    checkpoint = Checkpoint(exp_rootdir)

    print("Restoring model from {}".format(args.logdir))
    checkpoint.restore(session, os.path.basename(os.path.normpath(args.logdir)))

    # Evaluating.
    ssim_all = []
    psnr_all = []
    for i_ex in range(args.num_examples):
        try:
            gts = np.tile(
                get_single_batch(val_data_batch, session),
                [args.num_samples, 1, 1, 1, 1],
            )
            preds = session.run(
                model_components["open_loop_obs_decoded"]["prior_multistep"],
                feed_dict={model_components["training"]["obs"]: gts},
            )

            # Computing metrics.
            ssim, psnr = tools.compute_metrics(gts[:, args.open_loop_ctx :], preds)

            # Getting arrays save-ready
            gts = np.uint8(np.clip(gts, 0, 1) * 255)
            preds = np.uint8(np.clip(preds, 0, 1) * 255)

            # Finding the order within samples wrt avg metric across time.
            order_ssim = np.argsort(np.mean(ssim, -1))
            order_psnr = np.argsort(np.mean(psnr, -1))

            # Setting aside the best metrics among all samples for plotting.
            ssim_all.append(np.expand_dims(ssim[order_ssim[-1]], 0))
            psnr_all.append(np.expand_dims(psnr[order_psnr[-1]], 0))

            # Storing gt for prediction and the context.
            path = os.path.join(eval_logdir, "sample" + str(i_ex) + "_gt/")
            os.makedirs(path, exist_ok=True)
            np.savez(path + "gt_ctx.npz", gts[0, : args.open_loop_ctx])
            np.savez(path + "gt_pred.npz", gts[0, args.open_loop_ctx :])
            if not args.no_save_grid:
                tools.save_as_grid(gts[0, : args.open_loop_ctx], path, "gt_ctx.png")
                tools.save_as_grid(gts[0, args.open_loop_ctx :], path, "gt_pred.png")

            # Storing best and random samples.
            path = os.path.join(eval_logdir, "sample" + str(i_ex) + "/")
            os.makedirs(path, exist_ok=True)
            np.savez(path + "random_sample_1.npz", preds[0])
            if args.num_samples > 1:
                np.savez(path + "best_ssim_sample.npz", preds[order_ssim[-1]])
                np.savez(path + "best_psnr_sample.npz", preds[order_psnr[-1]])
                np.savez(path + "random_sample_2.npz", preds[1])
            if not args.no_save_grid:
                tools.save_as_grid(preds[0], path, "random_sample_1.png")
                if args.num_samples > 1:
                    tools.save_as_grid(
                        preds[order_ssim[-1]], path, "best_ssim_sample.png"
                    )
                    tools.save_as_grid(
                        preds[order_psnr[-1]], path, "best_psnr_sample.png"
                    )
                    tools.save_as_grid(preds[1], path, "random_sample_2.png")

        except tf.errors.OutOfRangeError:
            break

    # Plotting.
    tools.plot_metrics(ssim_all, eval_logdir, "ssim")
    tools.plot_metrics(psnr_all, eval_logdir, "psnr")
