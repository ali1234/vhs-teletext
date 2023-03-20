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
instead of a clean sequence like "1010" you something close to "0.5 0.5 0.5 0.5".
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

2. Some bytes are hamming encoded. These have even fewer possible combinations.

3. The first three bytes in the signal are always the same. We can use this
to find the start of the signal in the sample data (it moves a bit in each
line, but the width is always the same.)

4. The protocol itself defines rules about which bytes are allowed in which
positions, reducing the problem space further.


TRAINING
--------

A known signal is recorded to a tape using a Raspberry Pi with rpi-teletext.
This signal is played back into the computer, which builds a table of convolved
-> original sequences.


DECONVOLUTION
-------------

The convolved training data can be compared to recorded tapes in order to determine
what the data originally was. The line is first resampled to 1 sample per "bit". 
Then it is divided into "bytes". Each one is compared against the training
tables, including a few bits before and after. The closest match is the most
likely original signal.

This algorithm can be performed in parallel using CUDA or OpenCL. This allows
deconvolution to run in near realtime with a GTX 780.

See TRAINING.md for more.


SQUASHING
---------

The algorithm outputs lots of teletext packets, but they will still not be
perfect (even though they may be valid, they aren't necessarily correct.)

Since the teletext pages are broadcast on a loop, any recording of more than a
few minutes will have multiple copies of every packet. This means, if two packets
are received that only differ at a couple of bytes, they can be assumed to be the same.

The stream is first "paginated", ie split in to subpages.

All versions of the same subpage are compared, and for each byte, the most
frequent decoding is used so for example if you had these inputs:

HELLO
HELLP
MELLO

Then the result "HELLO" would be decoded, since those are the most frequent bytes
in this position. For this to work well, you need a lot of copies of every packet.
