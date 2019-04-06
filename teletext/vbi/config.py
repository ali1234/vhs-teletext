import math

import numpy as np

class Config(object):

    teletext_bitrate = 6937500.0
    gauss = 4.0
    std_thresh = 14

    sample_rate: float
    line_length: int
    line_start_range: tuple

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
        },
        'saa7131': {
            'sample_rate': 27000000.0,
            'line_length': 1440,
            'line_start_range': (0, 20),
        }
    }

    def __init__(self, card='bt8x8', **kwargs):
        for k, v in self.cards[card].items():
            setattr(self, k, v)

        for k, v in kwargs.items():
            if v is not None:
                setattr(self, k, v)

        # width of a bit in samples (float)
        self.bit_width = self.sample_rate / self.teletext_bitrate

        # region of the original line where the CRI begins, in samples
        self.start_slice = slice(*self.line_start_range)

        # last sample of original line where teletext may occur
        self.line_trim = self.start_slice.stop + math.ceil(self.bit_width * 45 * 8)

        # first sample for each bit in the rolled arrays
        self.bits = np.array([round(self.start_slice.start + (x * self.bit_width)) for x in range((45 * 8) + 9)])
        # number of samples in each bit
        self.bit_lengths = (self.bits[1:] - self.bits[:-1])

        # fft
        self.fftbins = [0, 47, 54, 97, 104, 147, 154, 197, 204]
