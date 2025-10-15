
import math
import sys
from typing import Iterable, Optional

import numpy as np
import torch
from sklearn.metrics import f1_score
from timm.data import Mixup
from timm.utils import accuracy

import modules.harmonizer.util.lr_sched as lr_sched
import modules.harmonizer.util.misc as misc


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    mixup_fn: Optional[Mixup] = None,
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

    for data_iter_step, (samples, targets, attn_mask) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        attn_mask = attn_mask.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples, attn_mask)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", max_lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, dataset_name):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    model.eval()

    batch_idx = 0
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]
        attn_mask = batch[2]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        attn_mask = attn_mask.to(device, non_blocking=True)

        attn_list = None
        with torch.amp.autocast("cuda"):
            outputs = model(images, attn_mask)
            if len(outputs) == 2:
                output, attn_list = outputs[0]
            else:
                output = outputs

            try:
                loss = criterion(output, target)
            except:
                print(f"output shape: {output.shape}, target shape: {target.shape}")
                raise

        if attn_list:
            import os

            os.makedirs(f"vis_attn_map/harmonizer/{dataset_name}", exist_ok=True)
            torch.save(
                attn_list,
                f"vis_attn_map/harmonizer/{dataset_name}/attn_list_batch_{batch_idx}.pt",
            )
            print(f"Saved attn_list to attn_list_batch_{batch_idx}.pt")
            batch_idx += 1

        acc1 = accuracy(output, target, topk=(1,))[0]

        predict = np.argmax(output.detach().cpu().numpy(), axis=1)

        if dataset_name == "PPMI":
            f1score = f1_score(
                target.detach().cpu().numpy(), predict, average="weighted"
            )
        else:
            f1score = f1_score(target.detach().cpu().numpy(), predict)
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["f1score"].update(f1score, n=batch_size)
    metric_logger.synchronize_between_processes()
    print(
        "* Acc@1 {top1.global_avg:.3f} f1score {f1.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1, f1=metric_logger.f1score, losses=metric_logger.loss
        )
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
