import numpy

# Tunable values

sample_rate = 35468950.0     # sample rate of BT8x8 PAL VBI
teletext_bitrate = 6937500.0 # bit rate of PAL teletext

line_start_range = [60, 120] # range covering start position of clock run-in, in samples

line_length = 2048           # length of raw lines in samples
line_trim = 1960             # ignore raw samples after this point

gauss = 3.0                  # amount of gaussian blurring to use when finding CRI
std_thresh = 14              # maximum standard deviation of samples before CRI
mdiff_thresh = 45            # maximum standard deviation of sample during CRI

# Calculated values

bit_width = sample_rate/teletext_bitrate

line_start = ((line_start_range[1] + line_start_range[0]) / 2) + (bit_width*1.5)

line_start_shift = ((line_start_range[1] - line_start_range[0]) / 2)

line_start_pre = [int(line_start - (bit_width * 15)), int(line_start - (bit_width * 2))]
line_start_post = [int(line_start + (bit_width * 2)), int(line_start + (bit_width * 15))]
line_start_frcmrag = [int(line_start + (bit_width * 17)), int(line_start + (bit_width * 40))]

bits = numpy.array([int(line_start + (x*bit_width)) for x in range((45*8)+9)])
bit_lengths = (bits[1:] - bits[:-1])
bit_pairs = [x for x in zip(bits[:-1], bits[1:])]

