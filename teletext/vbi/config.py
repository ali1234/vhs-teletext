import math
import pathlib

import numpy as np

class Config(object):

    teletext_bitrate = 6937500.0
    gauss = 4.0
    std_thresh = 14

    sample_rate: float
    line_length: int
    line_start_range: tuple
    dtype: type
    field_lines: int
    field_range: range

    extra_roll: int = 0
    sample_rate_adjust: float = 0

    # Clock run-in and framing code. These bits are set at the start of every teletext packet.
    crifc = np.array((
        1, -1, 1, -1, 1, -1, 1, -1,
        1, -1, 1, -1, 1, -1, 1, -1,
        1, 1, 1, -1, -1, 1, -1, -1,
    ))

    observed_crifc = np.array([
        [133, 132, 129, 127, 124, 121, 119, 117],
        [116, 115, 115, 115, 116, 117, 118, 119],
        [120, 121, 121, 121, 121, 120, 119, 118],
        [118, 117, 116, 116, 116, 117, 117, 118],
        [119, 120, 120, 121, 121, 121, 120, 119],
        [119, 118, 117, 116, 116, 116, 116, 117],
        [118, 119, 120, 121, 122, 122, 122, 122],
        [121, 120, 119, 118, 117, 117, 117, 117],
        [118, 118, 119, 120, 121, 121, 121, 121],
        [121, 120, 119, 119, 118, 118, 117, 117],
        [118, 118, 119, 120, 121, 122, 122, 122],
        [122, 121, 120, 119, 118, 118, 117, 117],
        [117, 117, 118, 119, 120, 120, 121, 121],
        [122, 122, 122, 122, 121, 121, 121, 121],
        [120, 120, 119, 118, 116, 115, 113, 110],
        [108, 105, 104, 103, 104, 107, 112, 119],
        [128, 137, 147, 157, 166, 174, 179, 183],
        [184, 183, 181, 178, 175, 171, 168, 166],
        [164, 163, 162, 160, 159, 156, 153, 147],
        [141, 133, 124, 114, 104,  96,  88,  82],
        [ 78,  77,  79,  83,  90,  99, 108, 118],
        [127, 134, 140, 144, 146, 145, 141, 136],
        [128, 119, 110, 100,  91,  83,  76,  69],
        [ 65,  61,  59,  57,  57,  57,  57,  58]
    ], dtype=np.uint8)

    observed_crifc_gradient = np.gradient(observed_crifc[8:24,:].flatten())

    # Card specific default parameters:

    cards = {
        'bt8x8': {
            'sample_rate': 35468950.0,
            'line_length': 2048,
            'line_start_range': (60, 130),
            'dtype': np.uint8,
            'field_lines': 16,
            'field_range': range(0, 16),
        },
        'cx88': {
            'sample_rate': 35468950.0,
            'line_length': 2048,
            'line_start_range': (90, 150),
            'dtype': np.uint8,
            'field_lines': 18,
            'field_range': range(1, 17),
        },
        'tbc': { # ld-decode/vhs-decode tbc (full fields)
            'sample_rate': 17730000.0,
            'line_length': 1135,
            'line_start_range': (160, 190),
            'dtype': np.uint16,
            'field_lines': 313,
            'field_range': range(6, 22),
        },
        'tbc-vbi': {  # ld-decode/vhs-decode tbc (vbi only)
            'sample_rate': 17730000.0,
            'line_length': 1135,
            'line_start_range': (160, 190),
            'dtype': np.uint16,
            'field_lines': 16,
            'field_range': range(0, 16),
        },
        'saa7131': {
            'sample_rate': 27000000.0,
            'line_length': 2048,
            'line_start_range': (0, 60),
            'dtype': np.uint8,
            'field_lines': 17,
            'field_range': range(0, 16),
        },
        'cx23885': {
            'sample_rate': 27000000.0,
            'line_length': 1440,
            'line_start_range': (0, 60),
            'dtype': np.uint8,
            'field_lines': 18,
            'field_range': range(1, 17),
        },
    }

    def __init__(self, card='bt8x8', **kwargs):
        setattr(self, 'card', card)

        for k, v in self.cards[card].items():
            setattr(self, k, v)

        for k, v in kwargs.items():
            if v is not None:
                setattr(self, k, v)

        self.frame_lines = self.field_lines * 2

        self.sample_rate += self.sample_rate_adjust

        # width of a bit in samples (float)
        self.bit_width = self.sample_rate / self.teletext_bitrate

        results = []
        for pad in range(500):
            r = (self.line_length+pad) * 8 / self.bit_width
            rs = round(r)
            err = abs(r - rs)
            results.append((err, pad, rs))

        # resample params
        self.resample_pad, self.resample_tgt = min(results)[1:]
        self.resample_size = math.ceil(self.line_length * 8 / self.bit_width)

        # region of the original line where the CRI begins, in samples
        self.start_slice = slice(
            math.floor(self.line_start_range[0] * 8 / self.bit_width),
            math.ceil(self.line_start_range[1] * 8 / self.bit_width)
        )

        # last sample of original line where teletext may occur
        self.line_trim = self.start_slice.stop + math.ceil(8 * 45 * 8)

        # fft
        self.fftbins = [0, 47, 54, 97, 104, 147, 154, 197, 204]

    def __repr__(self):
        return f'{type(self).__name__}: {self.__dict__}'

    @property
    def line_bytes(self):
        return self.line_length * np.dtype(self.dtype).itemsize

    __datadir = pathlib.Path(__file__).parent.parent / 'vbi' / 'data'
    tape_formats = [f.name for f in __datadir.iterdir() if f.is_dir()]
