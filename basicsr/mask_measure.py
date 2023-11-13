from basicsr.utils import imfrombytes
import torch
import os
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from tqdm import tqdm
import numpy as np
import multiprocessing as mp


dir = "/home/n2/chanwoo/benchmark-alignformer/AlignFormer/datasets/train/mask"
files = os.listdir(dir)

with open("log.txt", "w") as f:
    f.write("")
count = len(files)
small = 0
bucket = np.zeros(count)
for i, file in tqdm(enumerate(files), total=count):
    path = os.path.join(dir, file)
    bts = open(path, "rb").read()
    arr = imfrombytes(bts, float32=True)

    sum_val = arr.sum()
    num_val = arr.shape[0] * arr.shape[1] * arr.shape[2]
    avg_val = sum_val / (arr.shape[0] * arr.shape[1] * arr.shape[2])
    bucket[i] = avg_val
    if avg_val < 1e-10:
        with open("log.txt", "a") as f:
            f.write(f"{file}\n")

crits = [0.5, 0.25]
for i in range(1, 7):
    crits.append(10 ** (-i))
crits.append(1e-10)
crits.append(0)

for crit in crits:
    # count elements smaller than crit
    small = np.sum(bucket <= crit)
    print(f"crit: {crit}, small: {small}, ratio: {100 *small / count:.4f}%")

print("Random sampling")
REPS = 1000
MAX_RETRY = 100
bucket = np.zeros(count * REPS)


def handle_file(path):
    bts = open(path, "rb").read()
    arr = imfrombytes(bts, float32=True)
    bucket = np.zeros(REPS)
    tries = np.zeros(REPS)

    H, W, _ = arr.shape
    patchsize = 256
    for j in range(REPS):
        retry = 0
        while True:
            retry += 1
            x = np.random.randint(0, W - patchsize)
            y = np.random.randint(0, H - patchsize)
            patch = arr[y : y + patchsize, x : x + patchsize, :]
            sum_cal = patch.sum()
            if sum_cal < 1e-10:
                if retry < MAX_RETRY:
                    continue
            break
        num_cal = patchsize * patchsize * 3
        avg_val = sum_cal / num_cal
        bucket[j] = avg_val
        tries[j] = retry
    return bucket, tries


with mp.Pool(mp.cpu_count() // 2) as pool:
    res = pool.map(handle_file, [os.path.join(dir, file) for file in files])
    bucket = np.concatenate([r[0] for r in res])
    tries = np.concatenate([r[1] for r in res])
pool.close()
pool.join()


for crit in crits:
    # count elements smaller than crit
    small = np.sum(bucket <= crit)
    print(f"crit: {crit}, small: {small}, ratio: {100 *small / bucket.shape[0]:.4f}%")

print("Retry statistics")
for crit in [1, 5, 10, 20, 50, 100]:
    # count elements smaller than crit
    small = np.sum(tries > crit)
    print(f"crit: {crit}, small: {small}, ratio: {100 *small / tries.shape[0]:.4f}%")
