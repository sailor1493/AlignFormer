from basicsr.utils import imfrombytes
import torch
import os
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from tqdm import tqdm


lq_dir = "datasets/iphone_dataset/lq/test_sub"
gt_dir = "datasets/iphone_dataset/AlignFormer/test_sub"
lq_files = os.listdir(lq_dir)
lq_files.sort(key=lambda x: int(x.split(".")[0]))
psnr = PeakSignalNoiseRatio().to(0)
ssim = StructuralSimilarityIndexMeasure().to(0)

psnr_sum = 0
ssim_sum = 0

with open("img_log.csv", "w") as f:
    f.write("img,psnr,ssim\n")
    for lq_file in tqdm(lq_files):
        lq_path = os.path.join(lq_dir, lq_file)
        gt_path = os.path.join(gt_dir, lq_file)
        lq_bytes = open(lq_path, "rb").read()
        gt_bytes = open(gt_path, "rb").read()
        lq_img = imfrombytes(lq_bytes, float32=True) / 255.0
        gt_img = imfrombytes(gt_bytes, float32=True) / 255.0
        lq_img = torch.from_numpy(lq_img).to(0).permute(2, 0, 1).unsqueeze(0)
        gt_img = torch.from_numpy(gt_img).to(0).permute(2, 0, 1).unsqueeze(0)
        psnr_value = psnr(lq_img, gt_img)
        ssim_value = ssim(lq_img, gt_img)
        lq_filename = lq_file.split(".")[0]
        f.write(f"{lq_filename},{psnr_value.item()},{ssim_value.item()}\n")
        psnr_sum += psnr_value.item()
        ssim_sum += ssim_value.item()

print(f"PSNR: {psnr_sum / len(lq_files)}")
print(f"SSIM: {ssim_sum / len(lq_files)}")
