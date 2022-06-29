import click

from teletext.cli.clihelpers import packetreader
from teletext.celp import CELPDecoder

@click.group()
def celp():
    """Tools for analysing CELP audio packets."""
    pass


@celp.command()
@packetreader(filtered='data')
def play(packets):
    """Play data from CELP packets. Warning: Will make a horrible noise."""
    CELPDecoder().play(packets)


@celp.command()
@packetreader(filtered='data')
def plot(packets):
    """Plot data from CELP packets. Experimental code."""
    CELPDecoder.plot(packets)
