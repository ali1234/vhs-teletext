import numpy as np
import matplotlib.pyplot as plt
from spectrum import lsf2poly
from scipy.signal import lfilter


class CELPDecoder:
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
    offsets = np.cumsum(widths)

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
    ]) #* 2 * np.pi / sample_rate # convert to radians per sample

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

    def __init__(self, sample_rate=8000):
        self.sample_rate = sample_rate
        self.subframe_length = sample_rate // 200
        self.pos = 0

    @classmethod
    def decode_params(cls, raw_frame):
        bits = np.unpackbits(raw_frame, bitorder='little')
        decoded_frame = np.empty((30, ), dtype=np.int32)
        for n in range(len(cls.offsets)-2):
            slice = bits[cls.offsets[n]:cls.offsets[n+1]]
            decoded_frame[n] = np.packbits(slice, bitorder='little')
        lsf = cls.lsf_vector_quantizers[np.arange(10), decoded_frame[:10]]
        pitch_gain = cls.ltp_gain_quantization[decoded_frame[10:14]]
        vector_gain = cls.vec_gain_quantization[decoded_frame[14:18]]
        pitch_idx = decoded_frame[18:22]
        vector_idx = decoded_frame[22:26]
        return lsf, pitch_gain, vector_gain, pitch_idx, vector_idx

    # wave we're going to play through the filter
    wave = np.random.uniform(-1, 1, size=(8000, ))

    def apply_lpc_filter(self, lsf, signal):
        a = lsf2poly(sorted(lsf * 2 * np.pi / self.sample_rate))
        return lfilter([1], a, signal)

    def generate_audio(self, raw_frame):
        lsf, pitch_gain, vector_gain, pitch_idx, vector_idx = self.decode_params(raw_frame)

        frame = np.empty((self.subframe_length*4,), dtype=np.double)
        for subframe in range(4):
            gain = vector_gain[subframe]
            for n in range(self.subframe_length):
                self.pos += 1
                frame[(subframe*self.subframe_length) + n] = self.wave[self.pos % self.sample_rate] * gain * 0.0001

        filtered = self.apply_lpc_filter(lsf, frame)
        return np.clip((filtered * 32767), -32767, 32767).astype(np.int16)

    def decode_packet_stream(self, packets, frame=None):
        for p in packets:
            if frame is None or frame == 0:
                yield self.generate_audio(p._array[4:23])
            if frame is None or frame == 1:
                yield self.generate_audio(p._array[23:42])

    def play(self, packets, output=None):
        if output is None:
            import subprocess
            ps = subprocess.Popen(['play', '--buffer', '4000', '-t', 'raw', '-r', '8000', '-e', 'signed', '-b', '16', '-c', '1', '-'], stdin=subprocess.PIPE)
            output = ps.stdin
        for subframe in self.decode_packet_stream(packets):
            output.write(subframe.tobytes())

    @classmethod
    def plot(cls, packets):
        datas = []
        for p in packets:
            datas.append(p._array[4:])
        datas = np.concatenate(datas)

        data = np.unpackbits(datas.reshape(-1, 2, 19), bitorder='little').reshape(-1, 2, 152)
        frame0 = data[:, 0, :]
        frame1 = data[:, 1, :]
        d1 = np.sum(data, axis=0)

        p = np.arange(152)

        fig, ax = plt.subplots(5, 2)
        for n in range(cls.offsets.shape[0]-1):
            ax[0][0].bar(p[cls.offsets[n]:cls.offsets[n+1]], d1[0][cls.offsets[n]:cls.offsets[n+1]], 0.8)
            ax[0][1].bar(p[cls.offsets[n]:cls.offsets[n + 1]], d1[1][cls.offsets[n]:cls.offsets[n + 1]], 0.8)

        for x, frame in enumerate([frame0, frame1]):
            for y, o in enumerate([10, 14, 18, 22], start=1):
                bits = frame[:,cls.offsets[o]:cls.offsets[o+4]].reshape(-1, cls.widths[o+1])
                a = np.packbits(bits, axis=-1, bitorder='little').flatten()
                ax[y][x].plot(a[:10000], linewidth=0.5)

        fig.tight_layout()

        plt.show()
