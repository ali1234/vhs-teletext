import sys
import multiprocessing

import click

import teletext.vbi.deconvolve


# TODO: parser.add_argument('-H', '--headers', help='Synonym for --ansi --numbered --rows 0.', action='store_true')

@click.command()
@click.argument('inputfile', type=click.Path(exists=True))
@click.option('-a/-t', '--ansi/--t42', default=sys.stdout.isatty(), help='Force output type.')
@click.option('-n', '--numbered', is_flag=True, help='When output is ansi, number the lines according to position in input file.')
@click.option('-c', '--config', default='bt8x8_pal', help='Capture card configuration. Default: bt8x8_pal.')
@click.option('-T', '--threads', type=int, default=1, help='Number of threads. Default: 1.')
@click.option('-C', '--force-cpu', help='Disable CUDA even if it is available.')
@click.option('-e', '--extra_roll', type=int, default=4, help='')
@click.option('-S', '--squash', type=int, default=1, help='Merge N consecutive rows to reduce output.')
@click.option('--start', type=int, default=0, help='Start at the Nth line of the input file.')
@click.option('--stop', type=int, default=-1, help='Stop before the Nth line of the input file.')
@click.option('--count', type=int, default=-1, help='Stop after processing N lines from the input file.')
@click.option('--mags', '-m', type=int, multiple=True, default=range(9))
@click.option('--rows', '-r', type=int, multiple=True, default=range(32))
def deconvolve(*args, **kwargs):
    """Deconvolve raw VBI samples into a teletext bitstream."""
    teletext.vbi.deconvolve.deconvolve(*args, **kwargs)






