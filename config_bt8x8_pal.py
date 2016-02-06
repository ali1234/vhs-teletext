import numpy
from teletext.raw.pattern import Pattern

# Tunable values

sample_rate = 35468950.0     # sample rate of BT8x8 PAL VBI
teletext_bitrate = 6937500.0 # bit rate of PAL teletext

line_start_range = [60, 120] # range covering start position of clock run-in, in samples

line_length = 2048           # length of raw lines in samples
line_trim = 1960             # ignore raw samples after this point

# Calculated values

bit_width = sample_rate/teletext_bitrate

line_start = ((line_start_range[1] + line_start_range[0]) / 2) + (bit_width*1.5)

line_start_shift = ((line_start_range[1] - line_start_range[0]) / 2)

line_start_pre = [line_start - 70, line_start - 10]
line_start_post = [line_start + 10, line_start + 70]

bits = numpy.array([int(line_start + (x*bit_width)) for x in range((45*8)+3)])
bit_lengths = (bits[1:] - bits[:-1])
bit_pairs = [x for x in zip(bits[:-1], bits[1:])]

# tmp

fr = numpy.array([0xf4, 0xff, 0xcf, 0x54, 0x43, 0xaa, 0x0d, 0x00])

m = Pattern('mrag_patterns')
p = Pattern('parity_patterns')