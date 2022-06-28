import click

from teletext.cli.clihelpers import packetreader

@click.group()
def celp():
    """Tools for analysing CELP audio packets."""
    pass


@celp.command()
@packetreader(filtered='data')
def plot(packets):
    """Plot data from CELP packets. Experimental code."""
    from teletext.celp import plot as _plot
    _plot(packets)


@celp.command()
@packetreader(filtered='data')
def play(packets):
    """Play data from CELP packets. Warning: Will make a horrible noise."""
    from teletext.celp import _play
    _play(packets)
