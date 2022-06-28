import numpy as np
import matplotlib.pyplot as plt

from teletext.coding import hamming8_decode


def celp_plot(data):
    """Plot data from CELP packets. Experimental code."""
    data = np.unpackbits(np.fromfile(data, dtype=np.uint8).reshape(-1, 2, 19), bitorder='little').reshape(-1, 2, 152)
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


def celp_play(data):
    import miniaudio
    import time

    sample_rate = 8000

    def stream_pcm(data, device):
        data = np.unpackbits(np.fromfile(data, dtype=np.uint8).reshape(-1, 2, 19),
                             bitorder='little').reshape(-1, 2, 152)
        #data = data[:, 0, :]
        # data = data[:, 1, :]
        data = data.reshape(-1, 152)

        a = np.packbits(data[:, 37:37 + 20].reshape(-1, 5), axis=-1, bitorder='little').flatten()
        b = np.packbits(data[:, 37 + 20:37 + 20 + 20].reshape(-1, 5), axis=-1, bitorder='little').flatten().astype(np.int8)
        b -= 16
        c = np.packbits(data[:, 37 + 20 + 20:37 + 20 + 20 + 28].reshape(-1, 7), axis=-1, bitorder='little').flatten()
        d = np.packbits(data[:, 37 + 20 + 20 + 28:37 + 20 + 20 + 28 + 32].reshape(-1, 8), axis=-1, bitorder='little').flatten()

        subframes = a.shape[0]
        samples = subframes * 40

        wave = (((np.sin(np.linspace(0, 200*2*3.14159, 8000)) > 0) * 2) - 1) * 2
        #wave = np.sin(np.linspace(0, 55*2*3.14159, 8000))
        wave = np.sin(np.linspace(0, 100*2*3.14159, 8000))
        wave = np.random.normal(loc=0.0, scale=1.0, size=(8000, ))

        pos = 0
        required_frames = yield b""  # generator initialization
        while pos < samples:
            chunk = np.empty((required_frames, ), dtype=np.int16)
            for n in range(required_frames):
                pn = pos + n
                sf = pn//40
                s = wave[pn%wave.shape[0]] * (1.5**np.abs(b[sf])) * 8
                if abs(s) > 32767:
                    print("clip!")
                chunk[n] = s
            #print(np.max(chunk))
            required_frames = yield chunk
            pos += required_frames

        device.__running = False



    with miniaudio.PlaybackDevice(output_format=miniaudio.SampleFormat.SIGNED16,
                                  nchannels=1, sample_rate=sample_rate) as device:
        device.__running = True
        stream = stream_pcm(data, device)
        next(stream)  # start the generator
        device.start(stream)
        while device.__running:
            time.sleep(0.1)
