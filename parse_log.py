import re
from argparse import ArgumentParser

# 2023-11-12 22:55:33,076 INFO: 156_22.014951640336843_0.8612793415826284
#
pattern = re.compile(r"INFO: (\d+)_(\d+\.\d+)_(\d+\.\d+)")


def get_pattern(line):
    match = pattern.search(line)
    if match:
        img_no = int(match.group(1))
        psnr = float(match.group(2))
        ssim = float(match.group(3))
        return img_no, psnr, ssim
    else:
        return None


parser = ArgumentParser()
parser.add_argument("--log", type=str, help="log file")
parser.add_argument("--out", type=str, help="output file")

args = parser.parse_args()
with open(args.log, "r") as f:
    lines = f.readlines()

with open(args.out, "w") as f:
    f.write("img,psnr,ssim\n")
    for line in lines:
        res = get_pattern(line)
        if res:
            f.write("{},{},{}\n".format(res[0], res[1], res[2]))
