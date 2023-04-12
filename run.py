# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import argparse
import logging
import os
import shutil
from typing import *

import gin
import torch
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.plugins import DDPPlugin

from utils.select_option import select_callback, select_dataset, select_model


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise Exception("Boolean value expected.")


@gin.configurable()
def run(
    ginc: str,
    ginb: str,
    resume_training: bool,
    ckpt_path: Optional[str],
    scene_name: Optional[str] = None,
    datadir: Optional[str] = None,
    logbase: Optional[str] = None,
    model_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    postfix: Optional[str] = None,
    entity: Optional[str] = None,
    # Hyper Parameter in Aleth-NeRF
    K_l: float = 0.5,
    K_g: float = 0.5,
    eta: float = 0.2,
    overall_g: float = 1.0,
    # Optimization
    max_steps: int = -1,
    max_epochs: int = -1,
    precision: int = 32,
    # Logging
    log_every_n_steps: int = 1000,
    progressbar_refresh_rate: int = 5,
    # Run Mode
    run_train: bool = True,
    run_eval: bool = True,
    run_render: bool = False,
    num_devices: Optional[int] = None,
    num_sanity_val_steps: int = 0,
    seed: int = 777,
    debug: bool = False,
    save_last: bool = True,
    grad_max_norm=0.0,
    grad_clip_algorithm="norm",
):
    print('the scene name is:', scene_name)
    logging.getLogger("lightning").setLevel(logging.ERROR)
    datadir = datadir.rstrip("/")
    
    exp_name = (model_name + "_" + dataset_name + "_" + scene_name + "_" + str(seed).zfill(3) + \
        'eta' + str(eta))

    if postfix is not None:
        exp_name += "_" + postfix
    if debug:
        exp_name += "_debug"

    if num_devices is None:
        num_devices = torch.cuda.device_count()

    if model_name in ["plenoxel"]:
        num_devices = 1

    if logbase is None:
        logbase = "/data/unagi0/cui_data/Night_NeRF/logs_ablation"

    os.makedirs(logbase, exist_ok=True)
    logdir = os.path.join(logbase, exp_name)
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(os.path.join(logdir, exp_name), exist_ok=True)
    print('the log dir is:', logdir)
    logger = pl_loggers.TensorBoardLogger(
        save_dir=logdir,
        name=exp_name,
    )
    # Logging all parameters
    if run_train:
        txt_path = os.path.join(logdir, "config.gin")
        with open(txt_path, "w") as fp_txt:
            for config_path in ginc:
                fp_txt.write(f"Config from {config_path}\n\n")
                with open(config_path, "r") as fp_config:
                    readlines = fp_config.readlines()
                for line in readlines:
                    fp_txt.write(line)
                fp_txt.write("\n")

            fp_txt.write("\n### Binded options\n")
            for line in ginb:
                fp_txt.write(line + "\n")

    seed_everything(seed, workers=True)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(
        monitor="val/psnr",
        dirpath=logdir,
        filename="best",
        save_top_k=1,
        mode="max",
        save_last=save_last,
    )
    tqdm_progrss = TQDMProgressBar(refresh_rate=progressbar_refresh_rate)

    callbacks = []
    if not model_name in ["plenoxel"]:
        callbacks.append(lr_monitor)
    callbacks += [model_checkpoint, tqdm_progrss]
    callbacks += select_callback(model_name)

    ddp_plugin = DDPPlugin(find_unused_parameters=False) if num_devices > 1 else None

    trainer = Trainer(
        logger=logger if run_train else None,
        log_every_n_steps=log_every_n_steps,
        devices=num_devices,
        max_epochs=max_epochs,  #-1
        max_steps=max_steps,
        accelerator="gpu",
        replace_sampler_ddp=False,
        strategy=ddp_plugin,
        check_val_every_n_epoch=1,
        precision=precision,
        num_sanity_val_steps=num_sanity_val_steps,
        callbacks=callbacks,
        gradient_clip_algorithm=grad_clip_algorithm,
        gradient_clip_val=grad_max_norm,
    )

    if resume_training:
        if ckpt_path is None:
            ckpt_path = f"{logdir}/last.ckpt"

    data_module = select_dataset(
        dataset_name=dataset_name,
        scene_name=scene_name,
        datadir=datadir,
    )

    model = select_model(model_name=model_name, K_g = K_g, 
                        K_l = K_l, eta = eta, overall_g = overall_g) # model define
    model.logdir = logdir
    if run_train:   # Training
        best_ckpt = os.path.join(logdir, "best.ckpt")
        if os.path.exists(best_ckpt):
            os.remove(best_ckpt)
        version0 = os.path.join(logdir, exp_name, "version_0")
        if os.path.exists(version0):
            shutil.rmtree(version0, True)

        trainer.fit(model, data_module, ckpt_path=ckpt_path)

    if run_eval:    # Evaluation
        ckpt_path = (
            f"{logdir}/last.ckpt"
            if dataset_name != 'single_image_non_ref'
            else f"{logdir}/last.ckpt"
        )
        print('the checkpoint path is:', ckpt_path)
        trainer.test(model, data_module, ckpt_path=ckpt_path)

    if run_render:  # Rendering
        print('rendering')
        ckpt_path = (
            f"{logdir}/last.ckpt"
            #if model_name != "mipnerf360" or dataset_name != 'single_image_non_ref'
            if dataset_name != 'single_image_non_ref'
            else f"{logdir}/last.ckpt"
        )
        print('the checkpoint path is:', ckpt_path)
        trainer.predict(model, data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ginc",
        action="append",
        help="gin config file",
    )
    parser.add_argument(
        "--ginb",
        action="append",
        help="gin bindings",
    )
    parser.add_argument(
        "--resume_training",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="gin bindings",
    )
    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="path to checkpoints"
    )
    parser.add_argument("--K_l", type=float, default=0.3, help="Local Concealing Range")
    parser.add_argument("--K_g", type=float, default=0.3, help="Global Concealing Initial Value")
    parser.add_argument("--eta", type=float, default=0.1, help="Concealing Degree Control")
    parser.add_argument("--overall_g", type=float, default=1.0, help="Overall Gain for Final Enhancement, default 1.0")
    parser.add_argument("--seed", type=int, default=220901, help="seed to use")
    parser.add_argument("--logbase", type=str, default="./logs", help="seed to use")
    args = parser.parse_args()

    ginbs = []
    if args.ginb:
        ginbs.extend(args.ginb)

    logging.info(f"Gin configuration files: {args.ginc}")
    logging.info(f"Gin bindings: {ginbs}")

    gin.parse_config_files_and_bindings(args.ginc, ginbs)
    run(
        logbase=args.logbase,
        ginc=args.ginc,
        ginb=ginbs,
        K_l=args.K_l,
        K_g=args.K_g,
        eta=args.eta,
        overall_g=args.overall_g,
        resume_training=args.resume_training,
        ckpt_path=args.ckpt_path,
        seed=args.seed,
    )
