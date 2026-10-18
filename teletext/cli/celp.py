import click

from teletext.cli.clihelpers import packetreader
from teletext.celp import CELPDecoder


@click.group()
def celp():
    """Tools for analysing CELP audio packets.

    Some services carried compressed speech (CELP, code-excited linear
    prediction) in teletext independent data lines, for example the Audetel
    audio description trials. These commands decode that audio from a t42
    packet stream. The decoder is experimental and incomplete, so do not
    expect clean speech.

    CELP packets are rows 30 and 31 of magazine 4. Select them with the
    -d/--dcn option of each command:

    \b
      -d 4    packet 4/30, programme-related audio (Audetel)
      -d 12   packet 4/31, programme-independent audio

    To check whether a stream contains CELP packets, and which data channel
    they use:

    \b
      teletext filter -m 4 -r 30 -r 31 -o text - stream.t42

    Examples:

    \b
      teletext celp play -d 4 stream.t42
      teletext celp play -d 4 -o audio.wav stream.t42
      teletext celp plot -d 4 stream.t42

    Run "teletext celp COMMAND --help" for the options of each command.
    See CELP.md for a description of the packet format.
    """
    pass


@celp.command()
@click.option('-f', '--frame', type=click.IntRange(0, 1), default=None,
              help='Each packet holds two 20 ms audio frames. Decode only frame 0 or frame 1. Default: both, in order, '
                   'or only frame 0 (channel 1) in two channel mode.')
@click.option('-o', '--output', type=click.File('wb'),
              help='Write audio to this WAV file (8 kHz, 16 bit, mono) instead of playing it.')
@click.option('-l', '--lsf-lut', type=click.Choice(CELPDecoder.lsf_vector_quantizers.keys()), default='suddle-ong9', show_default=True,
              help='Line spectral frequency (LSF) quantizer table.')
@click.option('-g', '--gain-lut', type=click.Choice(CELPDecoder.vec_gain_quantizers.keys()), default='audetel', show_default=True,
              help='Excitation vector gain quantizer table.')
@click.option('-x', '--excitation', type=click.Choice(CELPDecoder.excitation_codebooks.keys()), default='pame', show_default=True,
              help='Stand-in for the unknown excitation codebook. noise: random noise, ignores the vector index. '
                   'gaussian: fixed random vector per index. clipped: overlapping centre-clipped random codebook. '
                   'pame: pitch-adaptive pulse trains followed by the clipped codebook (index mapping guessed).')
@packetreader(filtered='data', mag_hist=None, row_hist=None, err_hist=None, pass_progress=True)
def play(progress, frame, output, lsf_lut, gain_lut, excitation, packets):
    """Play or save audio decoded from CELP packets.

    Decodes the CELP frames in the packets of the selected data channel
    (-d/--dcn, usually 4) and plays them on the default sound output, or
    writes them to a WAV file with -o.

    The progress bar shows decoding error rates: L is the percentage of
    frames with invalid LSF parameters, VG the percentage of vector gains
    with errors detected by their Hamming check bits. High values point to
    a bad recording, a wrong data channel or a wrong quantizer table.

    Packets that contain text (test and talkback messages) are played as
    silence. In two channel mode only channel 1 is played unless -f is given.

    Warning: Will make a horrible noise.

    \b
    Examples:
      teletext celp play -d 4 stream.t42
      teletext celp play -d 4 -f 0 -o frame0.wav stream.t42
      teletext celp play -d 4 -l suddle -g unknown stream.t42
      teletext celp play -d 4 -x gaussian -o gaussian.wav stream.t42
    """
    dec = CELPDecoder(lsf_lut=lsf_lut, vec_gain_lut=gain_lut, excitation=excitation)
    if hasattr(progress, 'postfix'):
        progress.postfix.append(dec.stats())
    if output is not None:
        dec.convert(output, packets, frame=frame)
    else:
        dec.play(packets, frame=frame)


@celp.command()
@packetreader(filtered='data')
def plot(packets):
    """Plot statistics of CELP packets. Experimental code.

    Reads all packets of the selected data channel (-d/--dcn), then opens a
    matplotlib window with one column per CELP frame (0 and 1). The top row
    shows how often each of the 152 bits is set, coloured by parameter. The
    other rows show the pitch gain, vector gain, pitch lag and vector index
    of each subframe over time, and a histogram of the vector gains grouped
    by their error correction bits.

    All packets are held in memory, so use --limit on long recordings.

    \b
    Example:
      teletext celp plot -d 4 --limit 100000 stream.t42
    """
    CELPDecoder.plot(packets)
