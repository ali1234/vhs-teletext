Training
--------

1. Record the training signal to a tape:

```
teletext training | raspi-teletext -
```

2. Record it back to `training.vbi`.

3. Run the following script to process the training file into patterns:

```
#!/bin/sh
SPLITS=$(mktemp -d -p .)
teletext training split $SPLITS training.vbi
teletext training squash $SPLITS training.dat
teletext training build -m full -b 4 20 training.dat full.dat
teletext training build -m parity -b 6 20 training.dat parity.dat
teletext training build -m hamming -b 1 20 training.dat hamming.dat
cp full.dat parity.dat hamming.dat ~/Source/vhs-teletext/teletext/vbi/data/
echo $SPLITS
```

Theory
------

The idea behind training is to record a known teletext signal on to
tape and then play it back into the computer in the same way as you
would when recovering a tape. Then the original and observed signal
can be compared to build a database of patterns.

To make sure we can identify the degraded training packets, each one
has an ID and checksum. These are encoded so that each bit of data is
three bits wide in the output. This makes recovery of the original
trivial. There are also fixed bytes which can be used to help with
alignment.

We want to fit the most possible patterns into the least possible
tape. A De Bruijn sequence is used to do this. This is defined as the
shortest possible sequence which contains every sequence of input
characters (0 and 1 in this case) up to length N.

We use the De Bruijn sequence [2, 24]. This means it contains every
possible sequence of 1 or 0 of length 24 bits, which is about 16
million patterns. The ID field stores an offset into this sequence.

When recording it is possible for a run of whole frames to be lost,
so we do not simply display the whole De Bruijn sequence from start
to finish. Instead, for each packet, we add a prime number to the
offset and modulo the sequence length. This way every part of the
sequence is shown multiple times, and even a long run of frame drops
is unlikely to cause total loss of any part of the pattern.

After recording the signal back into the computer it is sliced into
patterns representing 24 bits of data. For a specific 24 bit pattern
there will be multiple slices in the signal. An average is taken of
every occurence and saved along with the original data it represents.
This is the intermediate training data.

Finally the pattern files are built. A pattern is describe like this:

 1. Number of bits to match before.
 2. Set of possible bytes to match.
 3. Number of bits to match after.

So for example, the parity data file is like this:

 build_pattern(args.parity, 'parity.dat', 4, 18, parity_set)

 Means: 

 1. Match 4 bits before.
 2. Match any byte with odd parity. (128 possibilities/7 bits)
 3. Match 3 bits after.

giving 14 bits total, or 16384 patterns.

To build the pattern data the intermediate data is processed and any
pattern which matches the criteria is added into a list. Then the
average for each list is taken. That is the final pattern we will 
match against.



