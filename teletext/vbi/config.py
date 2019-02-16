import numpy as np

class Config(object):

    teletext_bitrate = 6937500.0
    gauss = 3.0
    std_thresh = 14
    mdiff_thresh = 45

    sample_rate: float
    line_length: int
    line_trim: int
    line_start_range: tuple

    cards = {
        'bt8x8': {
            'sample_rate': 35468950.0,
            'line_length': 2048,
            'line_trim': 1960,
            'line_start_range': (60, 120),
        },
        'saa7131': {
            'sample_rate': 27000000.0,
            'line_length': 1600,
            'line_trim': 1440,
            'line_start_range': (0, 15),
        }
    }

    def __init__(self, card='bt8x8', **kwargs):
        for k, v in self.cards[card].items():
            setattr(self, k, v)

        for k, v in kwargs.items():
            if v is not None:
                setattr(self, k, v)

        self.bit_width = self.sample_rate / self.teletext_bitrate

        self.start_slice = slice(*self.line_start_range)

        self.pre_slice = slice(
            max(0, int(self.start_slice.start - (self.bit_width * 15))),
            max(1, int(self.start_slice.start - (self.bit_width * 2)))
        )

        self.post_slice = slice(
            int(self.start_slice.start + (self.bit_width * 2)),
            int(self.start_slice.start + (self.bit_width * 15))
        )

        self.frcmrag_slice = slice(
            int(self.start_slice.start + (self.bit_width * 17)),
            int(self.start_slice.start + (self.bit_width * 40))
        )

        self.bits = np.array([int(self.start_slice.start + (x * self.bit_width)) for x in range((45 * 8) + 9)])
        self.bit_lengths = (self.bits[1:] - self.bits[:-1])
        self.bit_pairs = [x for x in zip(self.bits[:-1], self.bits[1:])]
