import numpy as np
import matplotlib.pyplot as plt

from teletext.coding import hamming8_decode

from tqdm import tqdm

from spectrum.linear_prediction import lsf2poly
from scipy.signal import lfilter, filtfilt, sawtooth, square


def celp_print(packets, rows, o):
    """Dump CELP packets from data channels 4 and 12. We don't know how to decode all of these."""

    dblevels = [0, 4, 8, 12, 18, 24, 30, 0]

    servicetypes = [
        'Single-channel mode using 1 VBI line per frame',
        'Single-channel mode using 2 VBI lines per frame',
        'Single-channel mode using 3 VBI lines per frame',
        'Single-channel mode using 4 VBI lines per frame',
        'Mute Channel 1',
        'Two-channel Mode using 2 VBI lines per frame',
        'Mute Channel 2',
        'Two-channel Mode using 4 VBI lines per frame',
    ]

    for p in packets:
        if p.mrag.magazine == 4 and p.mrag.row in rows:
            dcn = p.mrag.magazine + ((p.mrag.row & 1) << 3)
            control = hamming8_decode(p._array[2])
            service = hamming8_decode(p._array[3])

            frame0 = p._array[4:23]
            frame1 = p._array[23:42]
            if o is None:
                print(f'DCN: {dcn} ({p.mrag.magazine}/{p.mrag.row})', end=' ')
                if dcn == 4:
                    print('Programme-related audio.', end=' ')
                    print('Service:', 'AUDETEL' if service == 0 else hex(service), end=' ')
                    print('Control:', hex(control), dblevels[control & 0x7], 'dB',
                          '(muted)' if control & 0x8 else '')
                elif dcn == 12:
                    print('Programme-independant audio.', end=' ')
                    if service & 0x8:
                        print('User-defined service', hex(service & 0x7), hex(p._array[3]))
                    else:
                        print(servicetypes[service], f'Control: {hex(control)}' if control else '')
                print(frame0.tobytes().hex(), frame1.tobytes().hex())
            else:
                o.write(frame0.tobytes())
                o.write(frame1.tobytes())


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


def celp_generate_audio(data, frame=None, sample_rate=8000):

    subframe_length = sample_rate // 200

    data = np.unpackbits(np.fromfile(data, dtype=np.uint8).reshape(-1, 2, 19),
                         bitorder='little').reshape(-1, 2, 152)
    if frame == 0:
        data = data[:, 0, :]
    elif frame == 1:
        data = data[:, 1, :]
    else:
        data = data.reshape(-1, 152)

    # parameter positions in the frame data
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

    # Speech Coding in Private and Broadcast Networks, Suddle, p121
    # note the transposition of first 8 values in column 7/8
    lsf_vector_quantizers = np.array([
        [ 143,  182,  214,  246,  284,  329,  389,  475,    0,    0,    0,    0,    0,    0,    0,    0,],
        [ 211,  252,  285,  317,  349,  383,  419,  458,  503,  554,  608,  665,  731,  809,  912, 1072,],
        [ 402,  470,  522,  571,  621,  671,  724,  778,  835,  902,  979, 1065, 1147, 1241, 1357, 1517,],
        [ 617,  732,  819,  885,  944, 1001, 1060, 1121, 1186, 1260, 1342, 1425, 1514, 1613, 1723, 1885,],
        [ 981, 1081, 1172, 1254, 1329, 1403, 1473, 1539, 1609, 1679, 1753, 1826, 1908, 1998, 2106, 2236,],
        [1334, 1446, 1539, 1626, 1697, 1763, 1828, 1890, 1954, 2019, 2087, 2160, 2238, 2328, 2420, 2526,],
        [1830, 1959, 2056, 2134, 2198, 2254, 2303, 2349, 2397, 2448, 2500, 2560, 2632, 2715, 2823, 2966,],
        [2247, 2361, 2434, 2496, 2550, 2600, 2647, 2694, 2742, 2791, 2846, 2904, 2966, 3049, 3155, 3256,],
        [2347, 2481, 2583, 2674, 2767, 2874, 3005, 3202,    0,    0,    0,    0,    0,    0,    0,    0,],
        [3140, 3246, 3326, 3395, 3458, 3524, 3601, 3709,    0,    0,    0,    0,    0,    0,    0,    0,],
    ]) * 2 * np.pi / sample_rate # convert to radians per sample

    vec_gain_quantization = np.array([
        -1100,  -850,  -650,  -510,  -415,  -335,  -275,  -220,
         -175,  -135,   -98,   -65,   -35,   -12,    -3,    -1,
            1,     3,    12,    35,    65,    98,   135,   175,
          220,   275,   335,   415,   510,   650,   850,  1100,
    ])

    ltp_gain_quantization = np.array([
        -0.993, -0.831, -0.693, -0.555, -0.414, -0.229, 0.0, 0.193,
        0.255, 0.368, 0.457, 0.531, 0.601, 0.653, 0.702, 0.745,
        0.780, 0.816, 0.850, 0.881, 0.915, 0.948, 0.983, 1.020,
        1.062, 1.117, 1.193, 1.289, 1.394, 1.540, 1.765, 1.991,
    ])

    # wave we're going to play through the filter
    #sq = (((np.sin(np.linspace(0, 1000*2*3.14159, 8000)) > 0) * 2) - 1)

    #sn2 = np.sin(np.linspace(0, 273*2*3.14159, 8000))
    wnu = np.random.uniform(-1, 1, size=(sample_rate, ))
    wnn = np.random.normal(loc=0.0, scale=1.0, size=(sample_rate, ))
    sw1 = sawtooth(np.linspace(0, 100, sample_rate) * 2 * np.pi, width=1) * 5
    sw2 = sawtooth(np.linspace(0, 133, sample_rate) * 2 * np.pi, width=1) * 5
    sw3 = sawtooth(np.linspace(0, 60, sample_rate) * 2 * np.pi, width=1) * 5
    sq = square(np.linspace(0, 2000, sample_rate))
    wave = (sq + (wnu*0.01)) * 0.00005

    pos = 0
    prev = None

    count, err = 0, 0

    for n in tqdm(range(0,data.shape[0])): # frames
        raw_frame = data[n]
        decoded_frame = np.empty((30, ), dtype=np.int32)
        for n in range(len(g)-2):
            slice = raw_frame[g[n]:g[n+1]]
            decoded_frame[n] = np.packbits(slice, bitorder='little')
        lsf_q = decoded_frame[:10]
        pitch_gain = ltp_gain_quantization[decoded_frame[10:14]]
        vector_gain = vec_gain_quantization[decoded_frame[14:18]]
        pitch_idx = decoded_frame[18:22]
        vector_idx = decoded_frame[22:26]
        frame = np.empty((subframe_length*4,), dtype=np.double)
        for subframe in range(4):
            gain = vector_gain[subframe]
            #gain = np.mean(np.abs(vector_gain))
            for n in range(subframe_length):
                pos += 1
                frame[(subframe*subframe_length) + n] = wave[pos % wave.shape[0]] * gain


        lsf = lsf_vector_quantizers[np.arange(10), lsf_q]
        #print(lsf)

        #simple way to make sure the lsf is valid: just sort it
        lsf = sorted(lsf)

        count += 1
        x = np.diff(lsf) <= 0
        if any(x):
            err += 1
            # if the parameters are not monotonic, use the ones from previous frame
            if prev is None:
                continue
            else:
                lsf = prev
        else:
            prev = lsf

        a = lsf2poly(lsf)

        filt = filtfilt(a, [1], frame)

        if np.max(filt) > 1:
            print("NOO", np.max(filt))

        result = np.clip((filt * 32000), -32767, 32767)

        yield result.astype(np.int16)


def celp_to_raw(data, output):
    if output is None:
        import subprocess
        ps = subprocess.Popen(['play', '--buffer', '4000', '-t', 'raw', '-r', '8000', '-e', 'signed', '-b', '16', '-c', '1', '-'], stdin=subprocess.PIPE)
        output = ps.stdin
    for subframe in celp_generate_audio(data, sample_rate=8000):
        output.write(subframe.tobytes())
