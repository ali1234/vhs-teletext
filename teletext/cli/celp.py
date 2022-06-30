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
@packetreader(filtered='data')
def play(frame, output, packets):
    """Play data from CELP packets. Warning: Will make a horrible noise."""
    if output is not None:
        CELPDecoder().convert(output, packets, frame=frame)
    else:
        CELPDecoder().play(packets, frame=frame)


@celp.command()
@packetreader(filtered='data')
def plot(packets):
    """Plot data from CELP packets. Experimental code."""
    CELPDecoder.plot(packets)
