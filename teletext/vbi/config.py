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

        line_start_mid = (sum(self.line_start_range) / 2) + (self.bit_width * 1.5)

        self.line_start_shift = ((self.line_start_range[1] - self.line_start_range[0]) / 2)

        self.line_start_slice = slice(*self.line_start_range)

        self.line_start_pre = slice(max(0, int(line_start_mid - (self.bit_width * 15))), max(1, int(line_start_mid - (self.bit_width * 2))))
        self.line_start_post = slice(int(line_start_mid + (self.bit_width * 2)), int(line_start_mid + (self.bit_width * 15)))
        self.line_start_frcmrag = slice(int(line_start_mid + (self.bit_width * 17)), int(line_start_mid + (self.bit_width * 40)))

        self.bits = np.array([int(line_start_mid + (x * self.bit_width)) for x in range((45 * 8) + 9)])
        self.bit_lengths = (self.bits[1:] - self.bits[:-1])
        self.bit_pairs = [x for x in zip(self.bits[:-1], self.bits[1:])]
