import pathlib
import numpy as np
from collections import defaultdict
from itertools import islice
from binascii import hexlify


def cluster(a, l, clusters=None, steps=None):
    if clusters is None:
        clusters = defaultdict(list)
    if steps is None:
        steps = np.floor(np.linspace(0, a.shape[1]-5, num=11)).astype(np.uint32)[[1, 5, 10]]
    v = np.empty((a.shape[0], 5), dtype=np.uint8)
    v[:, 0] = l
    v[:, 1] = np.floor(np.mean(np.abs(np.diff(a.astype(np.int16), axis=1)), axis=1)).astype(np.uint8)
    v[:, 2:] = np.diff(np.sort(a, axis=1)[:, steps] >> 4, axis=1, prepend=0)
    for vv, aa in zip(v, a):
        clusters[vv.tobytes()].append(aa)
    return v, clusters


def batched(iterable, n):
    "Batch data into lists of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


def batch_cluster(chunks, output, prefix="", lpf=32):
    output = pathlib.Path(output)

    with (output / f'{prefix}map.bin').open('wb') as mapfile:

        for batch in batched(chunks, 10000):
            a = np.stack(list(np.frombuffer(i[1], dtype=np.uint8) for i in batch))
            l = np.array(list(i[0] for i in batch)) % lpf
            map, clusters = cluster(a, l)
            mapfile.write(map.tobytes())
            for k, v in clusters.items():
                p = output / f'{prefix}{hexlify(k).decode("utf8")}.vbi'
                with p.open('ab') as f:
                    for l in v:
                        f.write(l.tobytes())


def rendermap(config, map, output):
    from PIL import Image
    import math
    a = np.fromfile(map, dtype=np.uint8).reshape(-1, config.frame_lines, 5)
    rows = []
    frames = 25 * 60
    for n in range(0, a.shape[0], frames):
        r = a[n:n+frames]
        if r.shape[0] < frames:
            r = np.concatenate([r, np.zeros((frames-r.shape[0], config.frame_lines, 5), dtype=np.uint8)], axis=0)
        r = np.swapaxes(r, 0, 1)
        rows.append(r)
    i = np.concatenate(rows, axis=0) * 20
    i = Image.fromarray(i[:,:,[1, 3, 4]], mode="RGB")
    i.save(output)

