import numpy as np
from glob import glob
from tqdm import tqdm
root_dir = "/131_data/jihwan/data/minecraft_lmdb/train"

files = sorted(glob(f"{root_dir}/*.npy"))

avgs = []
lens = []
for file in tqdm(files):
    arr = np.load(file)

    lens.append(len(arr))
    avgs.append(np.mean(arr, axis=0))


total_len = sum(lens)

total_avg = np.zeros_like(avgs[0])
for avg, length in zip(avgs, lens):
    total_avg += avg * length

total_avg /= total_len

print(total_avg)
print(total_avg.shape)

np.save(root_dir + "_avg.npy", total_avg)