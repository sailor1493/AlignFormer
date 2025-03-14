import importlib
import mmcv
import torch
import math
from collections import OrderedDict
from copy import deepcopy
import os
from os import path as osp
import numpy as np
import torch.nn.utils as tutils
from functools import partial

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
from basicsr.utils import (
    get_root_logger,
    tensor2img,
    tensor2npy,
)


@MODEL_REGISTRY.register()
class DISCNetModel(BaseModel):
    """Base model for PSF-aware restoration."""

    def __init__(self, opt):
        super().__init__(opt)

        # define network
        self.net_g = build_network(deepcopy(opt["network_g"]))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt["path"].get("pretrain_model_g", None)
        if load_path is not None:
            self.load_network(self.net_g, load_path, self.opt["path"]["strict_load"])
        self.clamp_function = lambda x: x
        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt["train"]

        # define losses
        if train_opt.get("pixel_opt"):
            self.cri_pix = build_loss(train_opt["pixel_opt"]).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get("perceptual_opt"):
            self.cri_perceptual = build_loss(train_opt["perceptual_opt"]).to(
                self.device
            )
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError("Both pixel and perceptual losses are None.")

        clip = train_opt.get("clip")
        if clip:
            self.clip_function = partial(
                tutils.clip_grad_norm_, max_norm=clip, foreach=True
            )
        else:
            self.clip_function = lambda x: x

        clamp = train_opt.get("clamp")
        if clamp:
            self.clamp_function = partial(torch.clamp, min=0.0, max=1.0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt["train"]
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f"Params {k} will not be optimized.")

        optim_type = train_opt["optim_g"].pop("type")
        if optim_type == "Adam":
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt["optim_g"])
        else:
            raise NotImplementedError(f"optimizer {optim_type} is not supperted yet.")
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data["lq"].to(self.device)
        if "gt" in data:
            self.gt = data["gt"].to(self.device)
        if "ref" in data:
            self.mask = data["ref"].to(self.device)
        if "psf_code" in data:
            self.psf_code = data["psf_code"].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq, self.psf_code)
        self.output = self.clamp_function(self.output)

        l_total = 0
        loss_dict = OrderedDict()
        mask = self.mask if hasattr(self, "mask") else None
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt, weight=mask)
            l_total += l_pix
            loss_dict["l_pix"] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt, weight=mask)
            if l_percep is not None:
                l_total += l_percep
                loss_dict["l_percep"] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict["l_style"] = l_style

        l_total.backward()
        self.clip_function(self.net_g.parameters())
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            N, C, H, W = self.lq.shape
            if H > 2000 or W > 2000:
                self.output = self.test_crop9()
            else:
                self.output = self.net_g(self.lq, self.psf_code)
                self.output = self.clamp_function(self.output)
        self.net_g.train()

    def test_crop9(self):
        N, C, H, W = self.lq.shape
        h, w = math.ceil(H / 3), math.ceil(W / 3)
        rf = 30
        imTL = self.net_g(self.lq[:, :, 0 : h + rf, 0 : w + rf], self.psf_code)[
            :, :, 0:h, 0:w
        ]
        imML = self.net_g(
            self.lq[:, :, h - rf : 2 * h + rf, 0 : w + rf], self.psf_code
        )[:, :, rf : (rf + h), 0:w]
        imBL = self.net_g(self.lq[:, :, 2 * h - rf :, 0 : w + rf], self.psf_code)[
            :, :, rf:, 0:w
        ]
        imTM = self.net_g(
            self.lq[:, :, 0 : h + rf, w - rf : 2 * w + rf], self.psf_code
        )[:, :, 0:h, rf : (rf + w)]
        imMM = self.net_g(
            self.lq[:, :, h - rf : 2 * h + rf, w - rf : 2 * w + rf], self.psf_code
        )[:, :, rf : (rf + h), rf : (rf + w)]
        imBM = self.net_g(
            self.lq[:, :, 2 * h - rf :, w - rf : 2 * w + rf], self.psf_code
        )[:, :, rf:, rf : (rf + w)]
        imTR = self.net_g(self.lq[:, :, 0 : h + rf, 2 * w - rf :], self.psf_code)[
            :, :, 0:h, rf:
        ]
        imMR = self.net_g(
            self.lq[:, :, h - rf : 2 * h + rf, 2 * w - rf :], self.psf_code
        )[:, :, rf : (rf + h), rf:]
        imBR = self.net_g(self.lq[:, :, 2 * h - rf :, 2 * w - rf :], self.psf_code)[
            :, :, rf:, rf:
        ]

        imT = torch.cat((imTL, imTM, imTR), 3)
        imM = torch.cat((imML, imMM, imMR), 3)
        imB = torch.cat((imBL, imBM, imBR), 3)
        output_cat = torch.cat((imT, imM, imB), 2)
        output_cat = self.clamp_function(output_cat)
        return output_cat

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        logger = get_root_logger()
        logger.info("Only support single GPU validation.")
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt["name"]
        with_metrics = self.opt["val"].get("metrics") is not None

        if save_img and not self.opt["is_train"]:
            img_dirpath = osp.join(self.opt["path"]["visualization"], dataset_name)
            os.makedirs(img_dirpath, exist_ok=True)

        if with_metrics:
            if not hasattr(self, "metric_results"):  # only execute in the first run
                self.metric_results = {
                    metric: 0 for metric in self.opt["val"]["metrics"].keys()
                }
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)

        _logger = get_root_logger()

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data["lq_path"][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals["result"]])
            if "gt" in visuals:
                # gt_img = tensor2raw([visuals['gt']]) # replace for raw data.
                gt_img = tensor2img([visuals["gt"]])
                del self.gt
            if "mask" in visuals:
                mask = tensor2img([visuals["mask"]])
                del self.mask

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            image_metric = {}

            if with_metrics:
                metric_data = {"img1": sr_img, "img2": gt_img, "mask": mask}
                # calculate metrics
                for name, opt_ in self.opt["val"]["metrics"].items():
                    _metric = calculate_metric(metric_data, opt_)
                    self.metric_results[name] += _metric
                    image_metric[name] = _metric

            metric_message = (
                f"{img_name}_{image_metric.get('psnr', 0)}_{image_metric.get('ssim',0)}"
            )
            _logger.info(metric_message)

            if save_img:
                if self.opt["is_train"]:
                    imgdir = osp.join(self.opt["path"]["visualization"], img_name)
                    os.makedirs(imgdir, exist_ok=True)
                    save_img_path = osp.join(imgdir, f"{img_name}_{current_iter}.png")

                else:
                    if self.opt["val"]["suffix"]:
                        save_img_path = osp.join(
                            img_dirpath,
                            f'{img_name}_{self.opt["val"]["suffix"]}.png',
                        )
                    else:
                        save_img_path = osp.join(
                            img_dirpath, f'{img_name}_{self.opt["name"]}.png'
                        )
                # np.save(save_img_path.replace('.png', '.npy'), sr_img) # replace for raw data.
                mmcv.imwrite(sr_img, save_img_path)
                # mmcv.imwrite(gt_img, save_img_path.replace('syn_val', 'gt'))

            save_npy = self.opt["val"].get("save_npy", None)
            if save_npy:
                if self.opt["is_train"]:
                    save_img_path = osp.join(
                        self.opt["path"]["visualization"],
                        img_name,
                        f"{img_name}_{current_iter}.npy",
                    )
                else:
                    if self.opt["val"]["suffix"]:
                        save_img_path = osp.join(
                            self.opt["path"]["visualization"],
                            dataset_name,
                            f'{img_name}_{self.opt["val"]["suffix"]}.npy',
                        )
                    else:
                        save_img_path = osp.join(
                            self.opt["path"]["visualization"],
                            dataset_name,
                            f"{img_name}.npy",
                        )

                np.save(
                    save_img_path, tensor2npy([visuals["result"]])
                )  # saving as .npy format.

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= idx + 1
                # update the best metric result
                self._update_best_metric_result(
                    dataset_name, metric, self.metric_results[metric], current_iter
                )

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f"Validation {dataset_name}\n"
        for metric, value in self.metric_results.items():
            log_str += f"\t # {metric}: {value:.4f}\n"
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f"metrics/{metric}", value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict["lq"] = self.lq.detach().cpu()
        out_dict["result"] = self.output.detach().cpu()
        if hasattr(self, "gt"):
            out_dict["gt"] = self.gt.detach().cpu()
        if hasattr(self, "mask"):
            out_dict["mask"] = self.mask.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, "net_g", current_iter)
        self.save_training_state(epoch, current_iter)
