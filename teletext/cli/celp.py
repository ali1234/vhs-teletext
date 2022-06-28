import click
import numpy as np
from tqdm import tqdm

from teletext.cli.clihelpers import command, carduser, chunkreader

@click.group()
def celp():
    """Tools for analysing CELP audio packets."""
    pass


@celp.command()
@click.argument('data', type=click.File('rb'))
def plot(data):
    """Plot data from CELP packets. Experimental code."""

    from teletext.celp import celp_plot

    celp_plot(data)


@celp.command()
@click.argument('data', type=click.File('rb'))
def play(data):
    """Play data from CELP packets. Warning: Will make a horrible noise."""

    from teletext.celp import celp_play

    celp_play(data)
