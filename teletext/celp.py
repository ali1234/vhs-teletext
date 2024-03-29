import numpy as np
import matplotlib.pyplot as plt
from spectrum import lsf2poly
import numpy as np
from scipy.signal import lfilter
from collections import deque
from tqdm import tqdm

hamming7_dec = np.array([
    [ 0x00, 0x10, 0x10, 0x14, 0x10, 0x11, 0x18, 0x12 ],
    [ 0x10, 0x11, 0x13, 0x19, 0x11, 0x01, 0x15, 0x11 ],
    [ 0x10, 0x1a, 0x13, 0x12, 0x16, 0x12, 0x12, 0x02 ],
    [ 0x13, 0x17, 0x03, 0x13, 0x1b, 0x11, 0x13, 0x12 ],
    [ 0x10, 0x14, 0x14, 0x04, 0x16, 0x1c, 0x15, 0x14 ],
    [ 0x1d, 0x17, 0x15, 0x14, 0x15, 0x11, 0x05, 0x15 ],
    [ 0x16, 0x17, 0x1e, 0x14, 0x06, 0x16, 0x16, 0x12 ],
    [ 0x17, 0x07, 0x13, 0x17, 0x16, 0x17, 0x15, 0x1f ],
    [ 0x10, 0x1a, 0x18, 0x19, 0x18, 0x1c, 0x08, 0x18 ],
    [ 0x1d, 0x19, 0x19, 0x09, 0x1b, 0x11, 0x18, 0x19 ],
    [ 0x1a, 0x0a, 0x1e, 0x1a, 0x1b, 0x1a, 0x18, 0x12 ],
    [ 0x1b, 0x1a, 0x13, 0x19, 0x0b, 0x1b, 0x1b, 0x1f ],
    [ 0x1d, 0x1c, 0x1e, 0x14, 0x1c, 0x0c, 0x18, 0x1c ],
    [ 0x0d, 0x1d, 0x1d, 0x19, 0x1d, 0x1c, 0x15, 0x1f ],
    [ 0x1e, 0x1a, 0x0e, 0x1e, 0x16, 0x1c, 0x1e, 0x1f ],
    [ 0x1d, 0x17, 0x1e, 0x1f, 0x1b, 0x1f, 0x1f, 0x0f ],
], dtype=np.uint8)

class LtpCodebook:
    def __init__(self, subframe_length):
        #self.buffer = np.zeros((147, subframe_length), dtype=np.double)
        #self.pos = 0
        self.buffer = deque(maxlen=147)

    def insert(self, subframe):
        self.buffer.extendleft(subframe)

    def get(self, lag):
        try:
            return self.buffer[lag + 20]
        except IndexError:
            return 0


class CELPStats:
    def __init__(self, decoder):
        self.decoder = decoder

    def __str__(self):
        result = f', L:{self.decoder.lsf_error:.0f}%, VG:{self.decoder.vector_gain_error:.0f}%'
        # reset the error counters
        self.decoder.lsf_errors = 0
        self.decoder.vector_gain_errors = 0
        self.decoder.subframes = 0
        return result


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


    lsf_vector_quantizers = {
        # Source Reliant Error Control For Low Bit Rate Speech Communications, Ong, p103
        'ong': np.array([
            [ 178,  218,  236,  267,  293,  332,  378,  420,    0,    0,    0,    0,    0,    0,    0,    0,],
            [ 210,  235,  265,  295,  325,  360,  400,  440,  480,  520,  560,  610,  670,  740,  810,  880,],
            [ 420,  460,  500,  540,  585,  640,  705,  775,  850,  950, 1050, 1150, 1250, 1350, 1450, 1550,],
            [ 752,  844,  910,  968, 1016, 1064, 1110, 1155, 1202, 1249, 1295, 1349, 1409, 1498, 1616, 1808,],
            [1041, 1174, 1274, 1340, 1407, 1466, 1514, 1559, 1611, 1658, 1714, 1773, 1834, 1906, 2008, 2166,],
            [1438, 1583, 1671, 1740, 1804, 1855, 1905, 1947, 1988, 2034, 2081, 2135, 2193, 2267, 2369, 2476,],
            [2005, 2115, 2176, 2222, 2260, 2297, 2333, 2365, 2394, 2427, 2463, 2501, 2551, 2625, 2728, 2851,],
            [2286, 2410, 2480, 2528, 2574, 2613, 2650, 2689, 2723, 2758, 2790, 2830, 2879, 2957, 3049, 3197,],
            [2775, 2908, 3000, 3086, 3159, 3234, 3331, 3453,    0,    0,    0,    0,    0,    0,    0,    0,],
            [3150, 3272, 3354, 3415, 3473, 3531, 3580, 3676,    0,    0,    0,    0,    0,    0,    0,    0,],
        ]),

        # Speech Coding in Private and Broadcast Networks, Suddle, p121
        # note the transposition of first 8 values in column 7/8
        'suddle': np.array([
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
        ]),
    }

    lsf_weights = np.array([1/8, 3/8, 5/8, 7/8])

    vec_gain_quantizers = {
        # Suddle, chapter 7
        'audetel': np.array([
            -1996, -1306, -990, -780, -628, -510, -418, -336,
            -268, -204, -148, -96, -54, -20, -6, -2,
            2, 6, 20, 54, 96, 148, 204, 268,
            336, 418, 510, 628, 780, 990, 1306, 1996,
        ]),

        # Suddle, chapter 5
        'unknown': np.array([
            -1100,  -850,  -650,  -510,  -415,  -335,  -275,  -220,
             -175,  -135,   -98,   -65,   -35,   -12,    -3,    -1,
                1,     3,    12,    35,    65,    98,   135,   175,
              220,   275,   335,   415,   510,   650,   850,  1100,
        ]),
    }

    ltp_gain_quantization = np.array([
        -0.993, -0.831, -0.693, -0.555, -0.414, -0.229,    0.0,  0.193,
         0.255,  0.368,  0.457,  0.531,  0.601,  0.653,  0.702,  0.745,
         0.780,  0.816,  0.850,  0.881,  0.915,  0.948,  0.983,  1.020,
         1.062,  1.117,  1.193,  1.289,  1.394,  1.540,  1.765,  1.991,
    ])

    def __init__(self, lsf_lut='suddle', vec_gain_lut='audetel', sample_rate=8000):
        self.lsf_lut = lsf_lut
        self.lsf_vector_quantization = self.lsf_vector_quantizers[lsf_lut]
        self.vec_gain_quantization = self.vec_gain_quantizers[vec_gain_lut]
        self.sample_rate = sample_rate
        self.subframe_length = sample_rate // 200
        self.pos = 0
        self.vector_gain_errors = 0
        self.lsf_errors = 0
        self.subframes = 0

    @property
    def lsf_error(self):
        return 100.0 * self.lsf_errors / self.subframes

    @property
    def vector_gain_error(self):
        return 100.0 * self.vector_gain_errors / self.subframes

    def stats(self):
        return CELPStats(self)

    def vector_parity(self, data, parity):
        hamm = hamming7_dec[data>>1, parity]
        self.vector_gain_errors += np.sum(hamm >> 4)
        return ((hamm&0xf)<<1)|(data&1)

    def decode_params(self, raw_frame):
        """Extracts the parameters from the raw packet according to offsets and widths."""
        bits = np.unpackbits(raw_frame, bitorder='little')
        decoded_frame = np.empty((30, ), dtype=np.uint8)
        for n in range(len(self.offsets)-2):
            slice = bits[self.offsets[n]:self.offsets[n+1]]
            decoded_frame[n] = np.packbits(slice, bitorder='little')
        lsf = self.lsf_vector_quantization[np.arange(10), decoded_frame[:10]]
        if np.any(np.diff(lsf) < 0):
            self.lsf_errors += 4
        pitch_gain = self.ltp_gain_quantization[decoded_frame[10:14]]
        #vector_gain = self.vec_gain_quantization[decoded_frame[14:18]]  # no hamming correction
        vector_gain = self.vec_gain_quantization[self.vector_parity(decoded_frame[14:18], decoded_frame[26:30])]
        pitch_idx = decoded_frame[18:22]
        vector_idx = decoded_frame[22:26]
        self.subframes += 4
        return lsf, pitch_gain, vector_gain, pitch_idx, vector_idx

    # wave we're going to play through the filter
    wave = np.random.uniform(-1, 1, size=(8000, ))

    def apply_lpc_filter(self, lsf, signal):
        """Convert line spectrum frequencies to a filter and apply to the signal."""
        a = lsf2poly(sorted(lsf * 2 * np.pi / self.sample_rate))
        result, self.last_z = lfilter([1], a, signal, zi=self.last_z)
        return result

    def generate_audio(self, raw_frame):
        """Generate an audio frame from a raw frame."""
        lsf, pitch_gain, vector_gain, pitch_idx, vector_idx = self.decode_params(raw_frame)

        # interpolate the LSFs
        if self.last_lsf is None:
            sub_lsf = np.repeat(lsf, 4).reshape(10, 4).T
        else:
            sub_lsf = (self.last_lsf[np.newaxis, :] * self.lsf_weights[::-1, np.newaxis]) + (lsf[np.newaxis, :] * self.lsf_weights[:, np.newaxis])

        frame = np.empty((self.subframe_length * 4,), dtype=np.double)
        subframe_buf = np.empty((self.subframe_length), dtype=np.double)
        for subframe in range(4):
            for n in range(self.subframe_length):
                self.pos += 1
                subframe_buf[n] = (self.wave[self.pos % self.sample_rate] * vector_gain[subframe]) + (self.ltp_codebook.get(pitch_idx[subframe] - n) * pitch_gain[subframe])
            self.ltp_codebook.insert(subframe_buf)
            p = subframe * self.subframe_length
            frame[p:p + self.subframe_length] = self.apply_lpc_filter(sub_lsf[subframe], subframe_buf)

        self.last_lsf = lsf
        return np.clip(frame * 0.5, -32767, 32767).astype(np.int16)

    def decode_packet_stream(self, packets, frame=None):
        """Decode an entire packet stream, yielding audio frames."""
        self.last_lsf = None
        self.last_z = np.zeros((10, ))
        self.ltp_codebook = LtpCodebook(self.subframe_length)
        for p in packets:
            if frame is None or frame == 0:
                yield self.generate_audio(p._array[4:23])
            if frame is None or frame == 1:
                yield self.generate_audio(p._array[23:42])

    def stream_pcm(self, packets, frame, device):
        src = self.decode_packet_stream(packets, frame)
        buf = []
        buflen = 0
        required_samples = yield b""  # generator initialization
        for frame in src:
            buf.append(frame)
            buflen += frame.shape[0]
            if buflen >= required_samples:
                tmp = np.concatenate(buf)
                buf = [tmp[required_samples:]]
                buflen = buf[0].shape[0]
                required_samples = yield tmp[:required_samples].tobytes()
        device.__running = False

    def play(self, packets, frame=None):
        """Play a packet stream."""
        import miniaudio
        import time

        with miniaudio.PlaybackDevice(output_format=miniaudio.SampleFormat.SIGNED16,
                                      nchannels=1, sample_rate=self.sample_rate,
                                      buffersize_msec=500) as device:
            device.__running = True
            stream = self.stream_pcm(packets, frame, device)
            next(stream)  # start the generator
            device.start(stream)
            while device.__running:
                time.sleep(0.1)

    def convert(self, output, packets, frame=None):
        import wave
        wf = wave.open(output, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        for frame in self.decode_packet_stream(packets, frame):
            wf.writeframes(frame.tobytes())
        wf.close()

    @classmethod
    def plot(cls, packets):
        """Plot statistics of the raw packets."""
        datas = []
        for p in packets:
            datas.append(p._array[4:])
        datas = np.concatenate(datas)

        data = np.unpackbits(datas.reshape(-1, 2, 19), bitorder='little').reshape(-1, 2, 152)
        d1 = np.sum(data, axis=0)
        p = np.arange(152)

        fig, ax = plt.subplots(6, 2)

        # plot the bit counts (top plots)
        for n in range(cls.offsets.shape[0]-1):
            s = slice(cls.offsets[n], cls.offsets[n + 1])
            ax[0][0].bar(p[s], d1[0][s], 0.8)
            ax[0][1].bar(p[s], d1[1][s], 0.8)

        # plot vector and pitch parameters over time
        for x in range(2):
            frame = data[:, x, :]
            for y, o in enumerate([10, 14, 18, 22], start=2):
                bits = frame[:,cls.offsets[o]:cls.offsets[o+4]].reshape(-1, cls.widths[o+1])
                a = np.packbits(bits, axis=-1, bitorder='little').flatten()
                ax[y][x].plot(a[:10000], linewidth=0.1)
                if y == 3:
                    hm = frame[:, cls.offsets[26]:cls.offsets[30]].reshape(-1, cls.widths[27])
                    hm = np.packbits(hm, axis=-1, bitorder='little').flatten()
                    for ham in range(8):
                        h = np.histogram(a[np.where(hm == ham)]>>1, np.arange(17))
                        ax[1][x].bar(np.arange(16), h[0])

        fig.tight_layout()
        plt.show()
