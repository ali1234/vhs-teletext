import numpy as np
import matplotlib.pyplot as plt
from spectrum import lsf2poly
import numpy as np
from scipy.signal import lfilter
from tqdm import tqdm

from teletext.coding import hamming8_decode

odd_parity = np.array([bin(n).count('1') & 1 for n in range(256)], dtype=bool)

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
    """Adaptive (long term prediction) codebook.

    Holds the most recent excitation samples. The vector for a lag L is the
    excitation from L samples ago. When L is shorter than the subframe, those
    L samples are repeated, as is usual in CELP decoders.
    """

    min_lag = 20
    max_lag = 147

    def __init__(self, subframe_length):
        self.subframe_length = subframe_length
        self.history = np.zeros((self.max_lag, ), dtype=np.double)

    def insert(self, subframe):
        self.history = np.concatenate((self.history, subframe))[-self.max_lag:]

    def get(self, lag):
        """Return the adaptive codebook vector for a lag in samples."""
        vector = np.empty((self.subframe_length, ), dtype=np.double)
        for n in range(self.subframe_length):
            vector[n] = self.history[n - lag] if n < lag else vector[n - lag]
        return vector


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

    # Suddle's table with the LSF 9 row from Ong. On real recordings Suddle's
    # LSF 9 row puts LSF 9 on average only 37 Hz above LSF 8 and 700 Hz below
    # LSF 10; Ong's row puts it in between, 440 Hz above LSF 8.
    lsf_vector_quantizers['suddle-ong9'] = lsf_vector_quantizers['suddle'].copy()
    lsf_vector_quantizers['suddle-ong9'][8] = lsf_vector_quantizers['ong'][8]

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

    # The real excitation codebook is not known. These are stand-ins:
    # 'noise' ignores the vector index and plays running random noise,
    # 'gaussian' is a fixed random codebook selected by the vector index,
    # 'clipped' is an overlapping centre-clipped random codebook and 'pame'
    # adds pitch-adaptive pulse trains in front of it, both as described by
    # Kondoz, Evans and Suddle (Eurospeech 93). The real codebook, once found,
    # can be added here as a (256, 40) array.
    @staticmethod
    def _clipped_codebook(size=256, length=40, shift=2, threshold=1.2):
        seq = np.random.default_rng(0).normal(0, 1, size=(size - 1) * shift + length)
        seq[np.abs(seq) < threshold] = 0
        book = np.lib.stride_tricks.sliding_window_view(seq, length)[::shift].copy()
        rms = np.sqrt(np.mean(book ** 2, axis=1, keepdims=True))
        return book / np.maximum(rms, 1e-9) / np.sqrt(3)

    excitation_codebooks = {
        'noise': None,
        'gaussian': np.random.default_rng(0).normal(0, 1 / np.sqrt(3), size=(256, 40)),
        'clipped': _clipped_codebook.__func__(),
        'pame': _clipped_codebook.__func__(),
    }

    def __init__(self, lsf_lut='suddle-ong9', vec_gain_lut='audetel', excitation='pame', sample_rate=8000):
        self.lsf_lut = lsf_lut
        self.excitation_type = excitation
        self.excitation_codebook = self.excitation_codebooks[excitation]
        self.lsf_vector_quantization = self.lsf_vector_quantizers[lsf_lut]
        self.vec_gain_quantization = self.vec_gain_quantizers[vec_gain_lut]
        self.sample_rate = sample_rate
        self.subframe_length = sample_rate // 200
        self.pos = 0
        self.vector_gain_errors = 0
        self.lsf_errors = 0
        self.subframes = 0
        self.text_frames = 0

    @property
    def lsf_error(self):
        return 100.0 * self.lsf_errors / max(self.subframes, 1)

    @property
    def vector_gain_error(self):
        return 100.0 * self.vector_gain_errors / max(self.subframes, 1)

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

    # noise played through the filter when no excitation codebook is used
    wave = np.random.default_rng(0).uniform(-1, 1, size=(8000, ))

    @staticmethod
    def pulse_vector(phase, spacing, length=40):
        """Unit pulses every spacing samples, starting at phase."""
        vec = np.zeros((length, ))
        vec[phase::spacing] = 1
        if not vec.any():
            return vec
        return vec / np.sqrt(np.mean(vec ** 2)) / np.sqrt(3)

    def excitation(self, vector_idx, lag=None):
        """Return the excitation vector for an index.

        For 'pame' the first indexes are pulse trains that depend on the
        pitch lag of the subframe: lag phases with pulses every lag samples,
        then, if the lag is longer than 40, lag/2 phases with pulses every
        lag/2 samples. The remaining indexes select the clipped codebook.
        """
        if self.excitation_codebook is None:
            pos = self.pos + np.arange(1, self.subframe_length + 1)
            self.pos += self.subframe_length
            return self.wave[pos % self.sample_rate]
        if self.excitation_type == 'pame' and lag is not None:
            if vector_idx < lag:
                return self.pulse_vector(vector_idx, lag, self.subframe_length)
            half = lag // 2 if lag > 2 * LtpCodebook.min_lag else 0
            if vector_idx < lag + half:
                return self.pulse_vector(vector_idx - lag, half, self.subframe_length)
            return self.excitation_codebook[vector_idx - lag - half]
        return self.excitation_codebook[vector_idx]

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
        for subframe in range(4):
            lag = pitch_idx[subframe] + LtpCodebook.min_lag
            subframe_buf = (self.excitation(vector_idx[subframe], lag) * vector_gain[subframe]) + \
                           (self.ltp_codebook.get(lag) * pitch_gain[subframe])
            self.ltp_codebook.insert(subframe_buf)
            p = subframe * self.subframe_length
            frame[p:p + self.subframe_length] = self.apply_lpc_filter(sub_lsf[subframe], subframe_buf)

        self.last_lsf = lsf
        return np.clip(frame * 0.5, -32767, 32767).astype(np.int16)

    @staticmethod
    def is_text(packet):
        """True if the packet holds a text message instead of audio.

        Test transmissions and talkback messages ("Audetel Test000001 Zero
        fade", "Test Talkback 0010 Address =88") are sent as odd parity text
        in the same packets. Random audio data almost never has odd parity
        in 34 or more of its 38 bytes.
        """
        return np.count_nonzero(odd_parity[packet._array[4:42]]) >= 34

    @staticmethod
    def is_two_channel(packet):
        """True if the packet's service byte selects a two channel mode.

        In two channel mode frame 0 of each packet belongs to channel 1 and
        frame 1 to channel 2. This was verified for service 5 on a recording;
        service 7 is assumed to work the same way.
        """
        if packet.mrag.row != 31:
            return False
        service = hamming8_decode(packet._array[2])
        return service in (5, 7)

    def decode_packet_stream(self, packets, frame=None):
        """Decode an entire packet stream, yielding audio frames.

        frame selects frame 0 or 1 of each packet. If it is None both frames
        are decoded in order, except in two channel mode, where only frame 0
        (channel 1) is decoded. Text packets are replaced with silence.
        """
        self.last_lsf = None
        self.last_z = np.zeros((10, ))
        self.ltp_codebook = LtpCodebook(self.subframe_length)
        silence = np.zeros((self.subframe_length * 4, ), dtype=np.int16)
        for p in packets:
            frames = (0, 1) if frame is None else (frame, )
            if frame is None and self.is_two_channel(p):
                frames = (0, )
            text = self.is_text(p)
            for f in frames:
                if text:
                    self.text_frames += 1
                    yield silence
                else:
                    yield self.generate_audio(p._array[4 + (f * 19):23 + (f * 19)])

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
