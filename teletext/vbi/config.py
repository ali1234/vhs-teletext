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

    extra_roll: int = 0
    sample_rate_adjust: float = 0

    # Clock run-in and framing code. These bits are set at the start of every teletext packet.
    crifc = np.array((
        1, -1, 1, -1, 1, -1, 1, -1,
        1, -1, 1, -1, 1, -1, 1, -1,
        1, 1, 1, -1, -1, 1, -1, -1,
    ))

    observed_crifc = np.array([
        [132, 130, 128, 125, 122, 120, 118, 116],
        [115, 115, 115, 115, 116, 117, 118, 120],
        [120, 121, 121, 121, 120, 120, 119, 118],
        [117, 117, 116, 116, 116, 117, 118, 119],
        [119, 120, 121, 121, 121, 121, 120, 119],
        [118, 117, 117, 116, 116, 116, 117, 117],
        [119, 120, 121, 121, 122, 122, 122, 121],
        [120, 119, 118, 117, 117, 117, 117, 117],
        [118, 119, 120, 120, 121, 121, 121, 121],
        [120, 120, 119, 118, 118, 117, 117, 118],
        [118, 119, 120, 121, 121, 122, 122, 122],
        [122, 121, 120, 119, 118, 117, 117, 117],
        [117, 117, 118, 119, 120, 121, 121, 121],
        [122, 122, 122, 121, 121, 121, 121, 120],
        [120, 119, 118, 117, 116, 114, 111, 109],
        [106, 104, 103, 103, 105, 109, 116, 123],
        [132, 142, 152, 162, 170, 177, 181, 183],
        [184, 182, 180, 177, 173, 170, 167, 165],
        [163, 162, 161, 160, 158, 155, 150, 144],
        [137, 128, 119, 109, 100,  92,  85,  80],
        [ 77,  78,  81,  86,  94, 103, 113, 122],
        [131, 138, 143, 145, 146, 143, 139, 132],
        [124, 115, 105,  96,  87,  79,  72,  67],
        [ 63,  60,  58,  57,  57,  57,  58,  58]
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
        'ddd-tbc': { # domesday duplicator tbc (full fields)
            'sample_rate': 17730000.0,
            'line_length': 1135,
            'line_start_range': (160, 190),
            'dtype': np.uint16,
            'field_lines': 313,
            'field_range': range(6, 22),
        },
        'ddd-vbi': {  # domesday duplicator vbi (pre-sliced)
            'sample_rate': 17730000.0,
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

        self.sample_rate += self.sample_rate_adjust

        print(self.sample_rate, self.sample_rate_adjust)

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
