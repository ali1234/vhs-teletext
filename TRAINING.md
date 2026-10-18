Training
--------

The decoder matches recorded signals against pattern tables (see
HOW_IT_WORKS.md). Tables for `vhs`, `betamax` and `grundig_2x4` recordings are
included in `teletext/vbi/data/`. You can build your own tables for a specific
VCR, tape speed or capture setup as follows.

1. Generate the training signal and record it to a tape with a Raspberry Pi
running [raspi-teletext](https://github.com/ali1234/raspi-teletext):

```
teletext training generate | raspi-teletext -
```

2. Play the tape back and capture it to `training.vbi`, exactly as you would
capture a tape you want to recover (e.g. `teletext record > training.vbi`).

3. Process the capture into pattern tables. Use the same `-c/--card` (and other
capture options) for `split` as you would use for `deconvolve`:

```
#!/bin/sh
set -e
FORMAT=my_vcr
SPLITS=$(mktemp -d -p .)
teletext training split $SPLITS training.vbi
teletext training squash $SPLITS training.dat
teletext training build -m full -b 4 20 training.dat full.dat
teletext training build -m parity -b 6 20 training.dat parity.dat
teletext training build -m hamming -b 1 20 training.dat hamming.dat
mkdir -p teletext/vbi/data/$FORMAT
cp full.dat parity.dat hamming.dat teletext/vbi/data/$FORMAT/
echo $SPLITS
```

Run this from the root of the repo with the package installed in editable mode
(`pip3 install -e .`). Every directory in `teletext/vbi/data/` automatically
becomes a choice for `-f/--tape-format`, so the new tables can then be used
with:

```
teletext deconvolve -f my_vcr capture.vbi > stream.t42
```

The intermediate split directory is large and can be deleted once `training.dat`
has been written.


Commands
--------

| Command                     | Description                                                    |
|-----------------------------|----------------------------------------------------------------|
| `training generate`         | Write the training packet stream to stdout (t42).              |
| `training split DIR INPUT`  | Decode a training capture into 256 intermediate bins in DIR.   |
| `training squash DIR OUT`   | Average the bins into one intermediate training file.          |
| `training build IN OUT`     | Build a pattern table from the intermediate file.              |
| `training showbin INPUT`    | Visually display an intermediate bin or training file.         |
| `training crifc INPUT`      | Print the averaged clock run-in and framing code of a capture. |

The output of `training crifc` has the same form as `observed_crifc` in
`teletext/vbi/config.py`, which the decoder uses to lock on to the start of
each line.


Theory
------

The idea behind training is to record a known teletext signal on to
tape and then play it back into the computer in the same way as you
would when recovering a tape. Then the original and observed signal
can be compared to build a database of patterns.

Each training packet contains 27 bytes of pattern data, an ID with a
checksum, and fixed bytes which can be used to help with alignment.
To make sure we can identify the degraded training packets, the ID and
checksum are encoded so that each bit of data is three bits wide in
the output. This makes recovery of the original trivial.

We want to fit the most possible patterns into the least possible
tape. A De Bruijn sequence is used to do this. This is defined as the
shortest possible sequence which contains every sequence of input
characters (0 and 1 in this case) up to length N.

We use the De Bruijn sequence B(2, 24). This means it contains every
possible sequence of 1 or 0 of length 24 bits, which is about 16
million patterns. It is stored in `teletext/vbi/data/debruijn.dat`.
The ID field stores an offset into this sequence.

When recording it is possible for a run of whole frames to be lost,
so we do not simply display the whole De Bruijn sequence from start
to finish. Instead, for each packet, we add a prime number (65521) to
the offset and take it modulo the sequence length. This way every part
of the sequence is shown multiple times, and even a long run of frame
drops is unlikely to cause total loss of any part of the pattern.

After recording the signal back into the computer, `split` finds each
training packet, reads its ID, and slices the pattern data into windows
representing 24 bits, stored alongside the original 24 bits they
represent. The windows are sorted into 256 bins by their first byte.
For a specific 24 bit pattern there will be multiple slices in the
signal. `squash` takes the average of every occurrence of each pattern
and saves it along with the original data it represents. This is the
intermediate training data (`training.dat`), with 27 bytes per entry:
3 bytes of original data followed by 24 averaged samples, one per bit.

Finally the pattern tables are built. Each entry in a table describes
one byte in the middle of the 24 bit window (bits 8-15), plus some
context bits before and after it. `build` takes:

 1. `-m/--mode` - the set of possible bytes to match: `full` (all 256),
    `parity` (the 128 bytes with odd parity) or `hamming` (the 16 valid
    hamming 8/4 codes).
 2. `-b/--bits START END` - the part of the 24 bit window to match.
    Bits START to 7 are context before the byte, bits 16 to END-1 are
    context after it.

So for example:

    teletext training build -m parity -b 6 20 training.dat parity.dat

Means:

 1. Match 2 bits before (bits 6 and 7).
 2. Match any byte with odd parity (128 possibilities).
 3. Match 4 bits after (bits 16 to 19).

giving 4 * 128 * 16 = 8192 patterns, each compared over 14 samples.

The tables shipped for `betamax` and `grundig_2x4` were built with the
options in the script above. The `vhs` tables use `-b 3 19` for full,
`-b 4 18` for parity and `-b 1 20` for hamming. More context bits make
the matching more accurate but need more training data to fill every
pattern.

To build a table, the intermediate data is processed and every entry
whose middle byte is in the chosen set is added to a list for its key
(the context bits before, the byte, and the context bits after). Then
the average for each list is taken. That is the final pattern we will
match against.

Each table file starts with a 14 byte header (big endian: pattern
length, output length, number of patterns, start bit, end bit),
followed by all the patterns and then the byte each pattern decodes to.
