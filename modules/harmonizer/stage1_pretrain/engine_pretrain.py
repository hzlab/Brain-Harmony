import math
import os
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import torch

import modules.harmonizer.util.lr_sched as lr_sched
import modules.harmonizer.util.misc as misc


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))
    os.makedirs(f"{args.output_dir}/train_imgs", exist_ok=True)
    for data_iter_step, samples in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        sample = samples[0].to(device, non_blocking=True)
        attn_mask = samples[1].to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            loss, pred, _ = model(sample, attn_mask)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    model.eval()

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, 20, header)
    ):
        images = batch
        images = [
            images[0].to(device, non_blocking=True),
            images[1].to(device, non_blocking=True),
        ]

        # compute output
        with torch.cuda.amp.autocast():
            loss, pred, _ = model(images)

        metric_logger.update(loss=loss.item())
    metric_logger.synchronize_between_processes()
