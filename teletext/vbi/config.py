import math

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

    # Clock run-in and framing code. These bits are set at the start of every teletext packet.
    crifc = np.array((
        1, -1, 1, -1, 1, -1, 1, -1,
        1, -1, 1, -1, 1, -1, 1, -1,
        1, 1, 1, -1, -1, 1, -1, -1,
    ))

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
        'ddd-tbc': { # domesday duplicator tbc (full fields)
            'sample_rate': 17712500.0,
            'line_length': 1135,
            'line_start_range': (160, 190),
            'dtype': np.uint16,
            'field_lines': 313,
            'field_range': range(6, 22),
        },
        'ddd-vbi': {  # domesday duplicator vbi (pre-sliced)
            'sample_rate': 17712500.0,
            'line_length': 1135,
            'line_start_range': (160, 190),
            'dtype': np.uint16,
            'field_lines': 16,
            'field_range': range(0, 16),
        },
        'saa7131': {
            'sample_rate': 27000000.0,
            'line_length': 1440,
            'line_start_range': (0, 20),
            'dtype': np.uint8,
            'field_lines': 16,
            'field_range': range(0, 16),
        },
    }

    def __init__(self, card='bt8x8', **kwargs):
        setattr(self, 'card', card)

        for k, v in self.cards[card].items():
            setattr(self, k, v)

        for k, v in kwargs.items():
            if v is not None:
                setattr(self, k, v)

        # width of a bit in samples (float)
        self.bit_width = self.sample_rate / self.teletext_bitrate

        # length of resampled line
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
