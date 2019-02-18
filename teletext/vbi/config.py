import math

import numpy as np

class Config(object):

    teletext_bitrate = 6937500.0
    gauss = 4.0
    std_thresh = 14

    sample_rate: float
    line_length: int
    line_start_range: tuple

    cards = {
        'bt8x8': {
            'sample_rate': 35468950.0,
            'line_length': 2048,
            'line_start_range': (60, 120),
        },
        'saa7131': {
            'sample_rate': 27000000.0,
            'line_length': 1600,
            'line_start_range': (0, 15),
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
        self.line_trim = self.start_slice.stop + math.ceil(self.bit_width * 46 * 8)

        # region immediately before the CRI in the rolled arrays
        self.pre_slice = slice(
            max(0, int(self.start_slice.start - (self.bit_width * 9))),
            max(1, int(self.start_slice.start - (self.bit_width * 2)))
        )

        # region immediately after the beginning of the CRI in the rolled arrays
        self.post_slice = slice(
            int(self.start_slice.start + (self.bit_width * 2)),
            int(self.start_slice.start + (self.bit_width * 9))
        )

        # region approximately containing framing code in the rolled arrays
        self.frcmrag_slice = slice(
            int(self.start_slice.start + (self.bit_width * 16)),
            int(self.start_slice.start + (self.bit_width * 24))
        )

        # first sample for each bit in the rolled arrays
        self.bits = np.array([round(self.start_slice.start + (x * self.bit_width)) for x in range((45 * 8) + 9)])
        # number of samples in each bit
        self.bit_lengths = (self.bits[1:] - self.bits[:-1])
