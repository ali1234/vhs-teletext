import numpy as np
import matplotlib.pyplot as plt


def plot(packets):
    """Plot data from CELP packets. Experimental code."""

    datas = []
    for p in packets:
        datas.append(p._array[4:])

    datas = np.concatenate(datas)
    data = np.unpackbits(datas.reshape(-1, 2, 19), bitorder='little').reshape(-1, 2, 152)
    frame0 = data[:, 0, :]
    frame1 = data[:, 1, :]
    d1 = np.sum(data, axis=0)

    p = np.arange(152)
    widths = np.array([
        0,
        3, 4, 4, 4, 4, 4, 4, 4, 3, 3, # 37 bytes - 10 x LPC params of (unknown?) variable size
        5, 5, 5, 5,             # 4x5 = 20 bytes - pitch gain (LTP gain)
        5, 5, 5, 5,             # 4x5 = 20 bytes - vector gain
        7, 7, 7, 7,             # 4x7 = 28 bytes - pitch index (LTP lag)
        8, 8, 8, 8,             # 4x8 = 32 bytes - vector index
        3, 3, 3, 3,             # 4x3 = 12 bytes - error correction for vector gains?
        3,                      # 3 bytes - always zero (except for recovery errors)
    ])
    g = np.cumsum(widths)
    print(sum(widths))

    fig, ax = plt.subplots(5, 2)
    for n in range(g.shape[0]-1):
        ax[0][0].bar(p[g[n]:g[n+1]], d1[0][g[n]:g[n+1]], 0.8)
        ax[0][1].bar(p[g[n]:g[n + 1]], d1[1][g[n]:g[n + 1]], 0.8)

    a = np.packbits(frame0[:,37:37+20].reshape(-1, 5), axis=-1, bitorder='little').flatten()
    b = np.packbits(frame0[:,37+20:37+20+20].reshape(-1, 5), axis=-1, bitorder='little').flatten()
    c = np.packbits(frame0[:,37+20+20:37+20+20+28].reshape(-1, 7), axis=-1, bitorder='little').flatten()
    d = np.packbits(frame0[:,37+20+20+28:37+20+20+28+32].reshape(-1, 8), axis=-1, bitorder='little').flatten()

    ax[1][0].plot(a[:10000], linewidth=0.5)
    ax[2][0].plot(b[:10000], linewidth=0.5)
    ax[3][0].plot(c[:10000], linewidth=0.5)
    ax[4][0].plot(d[:10000], linewidth=0.5)

    a = np.packbits(frame1[:,37:37+20].reshape(-1, 5), axis=-1, bitorder='little').flatten()
    b = np.packbits(frame1[:,37+20:37+20+20].reshape(-1, 5), axis=-1, bitorder='little').flatten()
    c = np.packbits(frame1[:,37+20+20:37+20+20+28].reshape(-1, 7), axis=-1, bitorder='little').flatten()
    d = np.packbits(frame1[:,37+20+20+28:37+20+20+28+32].reshape(-1, 8), axis=-1, bitorder='little').flatten()

    ax[1][1].plot(a[:10000], linewidth=0.5)
    ax[2][1].plot(b[:10000], linewidth=0.5)
    ax[3][1].plot(c[:10000], linewidth=0.5)
    ax[4][1].plot(d[:10000], linewidth=0.5)

    fig.tight_layout()

    plt.show()


def play(packets):
    for p in packets:
        pass
