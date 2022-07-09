import click

from teletext.cli.clihelpers import packetreader
from teletext.celp import CELPDecoder

@click.group()
def celp():
    """Tools for analysing CELP audio packets."""
    pass


@celp.command()
@click.option('-f', '--frame', type=int, default=None, help='Frame selection.')
@click.option('-o', '--output', type=click.File('wb'), help='Write audio to WAV file.')
@click.option('-l', '--lsf-lut', type=click.Choice(CELPDecoder.lsf_vector_quantizers.keys()), default='suddle', help='LSF vector look-up table.')
@click.option('-g', '--gain-lut', type=click.Choice(CELPDecoder.vec_gain_quantizers.keys()), default='audetel', help='LSF vector look-up table.')
@packetreader(filtered='data', mag_hist=None, row_hist=None, err_hist=None, pass_progress=True)
def play(progress, frame, output, lsf_lut, gain_lut, packets):
    """Play data from CELP packets. Warning: Will make a horrible noise."""
    dec = CELPDecoder(lsf_lut=lsf_lut, vec_gain_lut=gain_lut)
    progress.postfix.append(dec.stats())
    if output is not None:
        dec.convert(output, packets, frame=frame)
    else:
        dec.play(packets, frame=frame)


@celp.command()
@packetreader(filtered='data')
def plot(packets):
    """Plot data from CELP packets. Experimental code."""
    CELPDecoder.plot(packets)
