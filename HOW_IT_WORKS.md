HOW IT WORKS
------------

Teletext is encoded as a non-return-to-zero signal with two levels representing
one and zero. This is a fancy way of saying that a line of teletext data is
a sequence of black and white "pixels" in the TV signal. Of course, since the
signal is analogue there are no individual pixels, the signal is continuous.
But you can imagine that there are pixels in the idealized "perfect" signal.

The problem of decoding teletext from a VHS recording is that VHS bandwidth is
lower than teletext bandwidth. This means that the signal is effectively low
pass filtered, which in terms of an image is equivalent to gaussian blurring.

There are methods for reversing gaussian blur, but they are designed to work
with general image data. In the case of teletext we only have black or white
levels, so these methods are not optimal. We can exploit the limitations on
the input in order to get a better result. We can also exploit information
about the protocol to further improve efficiency and accuracy.

When the black and white signal is blurred, the individual pixels are blurred
in to each other. This makes the signal unreadable using normal methods, because
instead of a clean sequence like "1010" you get something close to "0.5 0.5 0.5 0.5".
But all is not lost, because a sequence like "1111" or "0000" will be the same
after blurring. So if you see a signal like "0.5 0.7 1.0 1.0" you can guess that
the original was probably "0 1 1 1" or "0 0 1 1".

There are 45 bytes in each teletext line, so the space of possible guesses is
2^(45*8) which is a very big number, which makes trying every guess completely
impractical. However there are ways to reduce this number:


FOUR RULES
----------

1. Nearly all bytes have a parity bit which means there are only 128 possible
combinations instead of 256.

2. Some bytes are hamming encoded. These have even fewer possible combinations
(16 valid codes).

3. The first three bytes in the signal are always the same: two bytes of clock
run-in (alternating ones and zeros) and the framing code. We can use this to
find the start of the signal in the sample data (it moves a bit in each line,
but the width is always the same.) These bytes are not stored in the output;
the remaining 42 bytes of each line make up one packet in a `.t42` file.

4. The protocol itself defines rules about which bytes are allowed in which
positions, reducing the problem space further.


FINDING THE SIGNAL
------------------

Each captured VBI line is processed independently (`teletext/vbi/line.py`):

1. The line is resampled so that there are exactly 8 samples per teletext bit.
The teletext bit rate is 6.9375 Mbit/s, so the resampling factor depends on the
capture card's sample rate (see the card table in the README).

2. The line is checked for a teletext signal. It is rejected if the signal is
too weak, if the noise floor before the expected start is too high, or if its
spectrum does not contain the harmonics expected from the teletext bit rate.
The percentage of rejected lines is shown in the progress bar.

3. The approximate start is found from the steepest rise in the expected start
region (`line_start_range` of the card). It is then refined twice: first by
looking for the distinctive end of the clock run-in and start of the framing
code, then by matching against an averaged recording of the real clock run-in
and framing code as it looks after passing through a VCR.

4. The samples for each bit are averaged, giving one value per bit, and the
line is normalised to the range 0-255.


DECONVOLUTION
-------------

The convolved training data can be compared to recorded tapes in order to
determine what the data originally was. Each byte of the line is compared
against the pattern tables, including a few bits before and after it. The
closest match (smallest sum of squared differences) is the most likely original
byte.

There are three pattern tables for each tape format, one for each type of
byte coding (see rules 1 and 2):

* `hamming.dat` - hamming 8/4 coded bytes
* `parity.dat` - 7 bit data with odd parity
* `full.dat` - unrestricted 8 bit data

The decoder first matches the magazine/row address and designation code with
the hamming table. It then knows which kind of packet it is looking at, and
decodes the rest of the line with the appropriate table:

| Packet                    | Coding                                                        |
|---------------------------|---------------------------------------------------------------|
| Row 0 (page header)       | page/subpage/control bytes hamming, header text parity        |
| Rows 1-25 (page content)  | parity, or full with `--eight-bit`                            |
| Row 27 (fastext links)    | hamming, plus full for the CRC (designation codes 0-3)        |
| Magazine 8, row 30 (BSDP) | hamming initial page, hamming or full data, parity status text |
| Other rows                | full                                                          |

Lines whose magazine or row are excluded with `-m`/`-r` are not decoded
further, which saves time.

The matching can be performed in parallel using CUDA or OpenCL, which allows
deconvolution to run in near realtime. On the CPU, lines are spread over
multiple processes (`-t`).

There is also a simple "slice" mode (`-M slice`) which recovers bits with a
threshold and differential instead of pattern matching. It needs no training
data but only works on clean signals.

See TRAINING.md for how the pattern tables are made.


SQUASHING
---------

The algorithm outputs lots of teletext packets, but they will still not be
perfect (even though they may be valid, they aren't necessarily correct.)

Since the teletext pages are broadcast on a loop, any recording of more than a
few minutes will have multiple copies of every packet. This means, if two
packets are received that only differ at a couple of bytes, they can be assumed
to be the same.

The stream is first "paginated", ie split in to subpages. Each page header
starts a new subpage in its magazine, and all following rows from that
magazine belong to it.

All versions of the same subpage (same magazine, page and subpage number) are
grouped together. With `--threshold`, versions which differ too much are put
into separate groups instead, so that different content sharing one subpage
number is not mixed. Groups with fewer than `--min-duplicates` versions are
discarded.

Within each group, for each byte, the most frequent decoding is used. So for
example if you had these inputs:

    HELLO
    HELLP
    MELLO

Then the result "HELLO" would be decoded, since those are the most frequent
bytes in this position. Rows 26 and 28 carry hamming 24/18 coded triplets of
three bytes each, so for those the most frequent whole triplet is used instead
of individual bytes. For this to
work well, you need a lot of copies of every packet.
