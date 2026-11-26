"""Write averaged-prob ensembles to cache (fp16) so seg26_paired.py can compare them per image."""
import sys
from pathlib import Path

import numpy as np

cache = Path(sys.argv[1])
out = sys.argv[2]
members = sys.argv[3:]
acc = None
for m in members:
    p = np.load(cache / m, mmap_mode="r")
    for s in range(0, p.shape[0], 32):
        chunk = np.asarray(p[s:s + 32], dtype=np.float32)
        if acc is None:
            acc = np.zeros(p.shape, dtype=np.float32)
        acc[s:s + 32] += chunk
acc /= len(members)
np.save(cache / out, acc.astype(np.float16))
print("wrote", out, acc.shape)
