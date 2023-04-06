import pathlib
import numpy as np
from collections import defaultdict
from itertools import islice
from binascii import hexlify


def cluster(a, clusters=None, steps=None):
    if clusters is None:
        clusters = defaultdict(list)
    if steps is None:
        steps = np.floor(np.linspace(0, a.shape[1]-5, num=11)).astype(np.uint32)[[1, 5, 9]]
    v = np.empty((a.shape[0], 4), dtype=np.uint8)
    v[:, 0] = np.floor(np.mean(np.abs(np.diff(a.astype(np.int16), axis=1)), axis=1)).astype(np.uint8)
    v[:, 1:] = np.sort(a, axis=1)[:, steps] >> 4
    for vv, aa in zip(v, a):
        clusters[vv.tobytes()].append(aa)
    return clusters


def batched(iterable, n):
    "Batch data into lists of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


def batch_cluster(chunks, output):
    output = pathlib.Path(output)

    for batch in batched(chunks, 10000):
        a = np.stack(list(np.frombuffer(i[1], dtype=np.uint8) for i in batch))
        clusters = cluster(a)
        for k, v in clusters.items():
            p = output / f'{hexlify(k).decode("utf8")}.vbi'
            with p.open('ab') as f:
                for l in v:
                    f.write(l.tobytes())


