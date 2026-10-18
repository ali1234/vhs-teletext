This is a suite of tools for processing teletext signals recorded on VHS (and
other analogue tape formats), as well as tools for processing teletext packet
streams. It can recover teletext from raw VBI captures, clean up the result,
and turn it into browsable pages, images, HTML or a re-broadcastable service.

Supported capture sources include bt8x8, cx88 and saa7131 cards, as well as
TBC files produced by ld-decode / vhs-decode. Other VBI capture hardware should
work with appropriate configuration (see "Capture devices" below).

This is the third major version of the software. The old versions are still
available in the `v1` and `v2` branches of this repo, or from the releases page.

You can see my collection of pages recovered with this software at:

https://al.zerostem.io/~al/teletext/

And more at:

http://www.teletextarchive.com

And:

http://archive.teletextart.co.uk/

See [HOW_IT_WORKS.md](HOW_IT_WORKS.md) for a description of the decoding
algorithm and [TRAINING.md](TRAINING.md) for how the pattern tables used by
the decoder are created. [CELP.md](CELP.md) describes the CELP audio tools.


INSTALLATION
------------

Python 3 is required. Install in editable mode from a checkout of this repo:

    pip3 install -e .

Optional features are available as extras. Add the ones you need, for example:

    pip3 install -e .[CUDA,spellcheck,viewer]

| Extra        | Package(s)            | Enables                                          |
|--------------|-----------------------|--------------------------------------------------|
| `CUDA`       | pycuda                | GPU deconvolution with CUDA                      |
| `OpenCL`     | pyopencl              | GPU deconvolution with OpenCL                    |
| `spellcheck` | pyenchant             | `teletext spellcheck`                            |
| `viewer`     | PyOpenGL              | `teletext vbiview`                               |
| `qt`         | PyQt5                 | `ttviewer` and the Qt based tools                |
| `audio`      | spectrum, miniaudio   | `teletext celp` (CELP audio packet analysis)     |
| `profiler`   | plop                  | Profiling support                                |

If an extra is not available for your platform simply omit it from the
install command. The commands that need it will report that it is missing.

`teletext images` and `teletext vbi histogram` additionally need Pillow
(`pip3 install pillow`).

### GPU decoding

CUDA decoding requires the Nvidia proprietary driver. OpenCL decoding requires
pyopencl and an OpenCL runtime for your card. `deconvolve` tries CUDA first,
then OpenCL, then falls back to the CPU. Use `-O` to prefer OpenCL over CUDA
and `-C` to force CPU decoding.

### Displaying teletext in a terminal

Commands that print packets to a terminal (for example `filter` or
`deconvolve` without an output file) use teletext mosaic characters. To render
them correctly you need a teletext font and a suitable terminal:

    sudo apt-get install tv-fonts rxvt-unicode

Then enable bitmap fonts in your X server:

    cd /etc/fonts/conf.d
    sudo rm 70-no-bitmaps.conf
    sudo ln -s ../conf.avail/70-yes-bitmaps.conf .

After doing this you may need to rehash:

    xset fp rehash

Finally open a terminal with the required font:

    urxvt -fg white -bg black -fn teletext -fb teletext -geometry 41x25 +sb &

Alternatively, pass `-u` before the command (e.g. `teletext -u interactive`)
to use the Unicode 13.0 "Symbols for Legacy Computing" block for mosaic
graphics, which works with any terminal font that supports it.

TrueType teletext fonts and CSS for the HTML output are in `misc/`.


QUICK START
-----------

First capture VBI from VHS:

    teletext record -d /dev/vbi0 > capture.vbi

Deconvolve the recording into a stream of teletext packets (t42 format):

    teletext deconvolve capture.vbi > stream.t42

Examine the headers to find the services recorded on the tape:

    teletext filter -r 0 stream.t42

Or get a compact overview of headers and broadcast service data, one line
per 250 frames (10 seconds):

    teletext scan stream.t42

Split the capture into services, using packet numbers found above:

    teletext filter --start <N> --stop <N> stream.t42 > service-1.t42

List the pages present in a stream:

    teletext list -c service-1.t42

Display all copies of a page:

    teletext filter -p 100 service-1.t42

Squash duplicate subpages, which reduces errors:

    teletext squash service-1.t42 > squashed.t42

Generate HTML pages from a stream:

    teletext html output/ squashed.t42

Interactively view the pages in a stream:

    teletext service squashed.t42 | teletext interactive

Run any command with `--help` for a complete list of its options, or
`teletext --help-all` to show help for every command.


CAPTURE DEVICES
---------------

Commands that read raw VBI samples (`record`, `deconvolve`, `vbiview`, the
`training` and `vbi` tools) take `-c/--card` to select the sample format:

| Card      | Source                                         | Sample rate  | Samples/line | Sample type |
|-----------|------------------------------------------------|--------------|--------------|-------------|
| `bt8x8`   | bt848/bt878 VBI device (default)               | 35.46895 MHz | 2048         | 8 bit       |
| `cx88`    | cx2388x VBI device                             | 35.46895 MHz | 2048         | 8 bit       |
| `saa7131` | saa713x VBI device                             | 27 MHz       | 1440         | 8 bit       |
| `tbc`     | ld-decode / vhs-decode `.tbc`, full fields     | 17.73 MHz    | 1135         | 16 bit      |
| `tbc-vbi` | ld-decode / vhs-decode `.tbc`, VBI lines only  | 17.73 MHz    | 1135         | 16 bit      |

If your hardware is not listed, the defaults can be adjusted with
`--line-length`, `--sample-rate`, `--sample-rate-adjust`,
`--line-start-range` and `--extra-roll`. `teletext vbiview` and
`teletext vbi plot` are useful for checking that the settings are right.

`deconvolve` and `vbiview` also take `-f/--tape-format`, which selects the set
of pattern tables trained for a particular kind of recording. The available
choices are the directories in `teletext/vbi/data/`: currently `vhs`
(default), `betamax` and `grundig_2x4`. Training your own tables for a
specific VCR can improve results; see TRAINING.md.

`deconvolve -M slice` uses a simple threshold slicer instead of pattern
matching. It is only useful for good quality signals, such as off-air
captures.


COMMANDS
--------

### Capture and decoding

| Command      | Description                                                       |
|--------------|-------------------------------------------------------------------|
| `record`     | Record raw VBI samples from a capture device.                     |
| `deconvolve` | Turn raw VBI samples into teletext packets.                       |
| `vbiview`    | Display raw VBI samples with OpenGL.                              |
| `vbi`        | Tools for analysing raw VBI (histogram, plot, cluster, linesplit). |
| `training`   | Tools for creating pattern tables. See TRAINING.md.               |

### Packet stream processing

| Command      | Description                                                       |
|--------------|-------------------------------------------------------------------|
| `filter`     | Select and display packets by magazine, row, page and subpage.    |
| `grep`       | Select packets or pages matching a regular expression.            |
| `list`       | List the pages (and optionally subpages) in a stream.             |
| `scan`       | Summarise headers and broadcast service data over time.           |
| `split`      | Split a stream into one file per page/subpage.                    |
| `squash`     | Reduce errors by combining duplicate copies of each subpage.      |
| `finders`    | Fix up common header formats using known patterns.                |
| `spellcheck` | Correct errors using a spelling checker (needs pyenchant).        |
| `diff`       | Show the differences between two streams side by side.            |

### Output

| Command       | Description                                                      |
|---------------|------------------------------------------------------------------|
| `interactive` | Interactive teletext viewer in the terminal.                     |
| `html`        | Generate a browsable set of HTML pages.                          |
| `images`      | Render every subpage to PNG (animated if it uses flash).         |
| `urls`        | Print an edit.tf editor URL for every subpage.                   |
| `service`     | Build a looping broadcast carousel from a stream.                |
| `servicedir`  | Build a carousel from a directory of t42 files, live reloaded.   |
| `serial`      | Send packets to a USB serial teletext inserter.                  |
| `celp`        | Analyse/play CELP audio packets (needs the `audio` extra).       |

`ttviewer` is a separate Qt based viewer/editor (needs the `qt` extra).

### Common options

Most commands that read a t42 packet stream share these options:

* `INPUT` - input file, or stdin if omitted.
* `--start`, `--stop`, `--step`, `--limit` - select a range of packets (or
  lines, for raw VBI input) by position in the file.
* `-m/--mag`, `-r/--row` - only keep packets from these magazines/rows. Can be
  given multiple times.
* `-p/--page`, `-s/--subpage` - only keep these pages/subpages (in hex). Can
  be given multiple times. Implies pagination.
* `-P/--paginate` - sort rows into complete pages.
* `--wst` - input uses 43 byte packets (WST capture card format).
* `--ts PID` - input is an MPEG transport stream; extract teletext from PID.

Commands that write packets take `-o/--output FORMAT FILE`, which can be
given multiple times. Formats are `auto`, `text`, `ansi`, `debug`, `bar`,
`bytes`, `hex` and `vbi`. `auto` writes `debug` to a terminal and raw t42
`bytes` to anything else. For example, to save a stream and watch it at the
same time:

    teletext deconvolve capture.vbi -o bytes stream.t42 -o ansi -


SQUASHING
---------

`teletext squash` groups all copies of each subpage and, for every byte
position, keeps the most common value. Options:

* `-d/--min-duplicates N` - subpages seen fewer than N times are dropped
  (default 3). Lower this for short recordings.
* `-t/--threshold N` - only group copies that differ by less than N bytes,
  so that subpages which share a number but have different content (for
  example rolling news) are kept apart. The default of -1 groups all copies
  with the same page and subpage number.
* `-i/--ignore-empty` - when a subpage has duplicate rows, ignore the
  emptiest copies instead of the earliest.


INTERACTIVE VIEWER
------------------

`teletext interactive` emulates a teletext TV decoder in the terminal, which
must be at least 41x25 characters. It reads a looping packet stream, so
it is normally fed by `teletext service`:

    teletext service squashed.t42 | teletext interactive -i 100

Keys:

* `0`-`9`, `a`-`f` - enter a page number
* `.` - hold
* `r` - reveal concealed text
* `h`, `j`, `k`, `l` - follow the red, green, yellow and cyan fastext links
* `q` - quit

### National character sets

Teletext uses national variants of the G0 character set which replace a few
characters such as `#`, `[` or `{` with local letters (for example Ä, Ö, Ü and
ß in German). Use `--localcodepage` to choose the set:

* `auto` (default) - select the set from the national option bits in each page
  header. Bits for German select `ger`, bits for Swedish/Finnish select `swe`,
  anything else uses the default (English) set.
* `default`, `ger`, `swe`, `cyr` - always use this set, whatever the header
  says. This is useful because many broadcasters left the national option bits
  at zero, so `auto` falls back to English on those recordings.

For example, to view a German service whose headers do not carry the right
option bits:

    teletext service squashed.t42 | teletext interactive --localcodepage ger

`teletext html` takes the same `--localcodepage` option, without `auto`. If
it is not given, the default set is used.


SERVICES AND INSERTERS
----------------------

`teletext service` turns a stream into an endless broadcast carousel, with
`-r` to replace headers with a live clock and `-t` to set the header title.
`teletext servicedir DIR` does the same from a directory of t42 files laid out
as `DIR/<page>/<subpage>.t42` (e.g. `100/0000.t42`), reloading files as they
change. The output can be piped into `interactive`, raspi-teletext, or a USB
serial inserter via `teletext serial`.

The `examples/` directory contains scripts showing how to generate pages and
services from Python.


CELP AUDIO
----------

Some services, such as the Audetel audio description trials, sent CELP coded
speech in packets 4/30 and 4/31. `teletext celp` (needs the `audio` extra)
decodes this audio. Check for CELP packets, then play or save the audio of
data channel 4:

    teletext filter -m 4 -r 30 -r 31 -o text - stream.t42
    teletext celp play -d 4 stream.t42
    teletext celp play -d 4 -o audio.wav stream.t42

The decoder is experimental. See [CELP.md](CELP.md) for all options and the
packet format.


TESTS
-----

    python3 -m unittest discover -s tests -t .

The spellcheck tests need the `spellcheck` extra; GPU tests are skipped when
CUDA or OpenCL is not available.
