import importlib
import sys

import click
from tqdm import tqdm

from teletext.file import FileChunker
from teletext.t42.packet import Packet



@click.command()
@click.argument('input', type=click.File('rb'))
@click.option('--start', type=int, default=0, help='Start at the Nth line of the input file.')
@click.option('--stop', type=int, default=None, help='Stop before the Nth line of the input file.')
@click.option('--step', type=int, default=1, help='Process every Nth line from the input file.')
@click.option('--limit', type=int, default=None, help='Stop after processing N lines from the input file.')
@click.option('--mags', '-m', type=int, multiple=True, default=range(9))
@click.option('--rows', '-r', type=int, multiple=True, default=range(32))
def pipe(input, start, stop, step, limit, mags, rows):

    chunks = FileChunker(input, 42, start, stop, step, limit)
    pbar = tqdm(chunks, unit=' Lines')
    packets = (Packet(data, number) for number, data in chunks)

    for p in packets:
        pbar.write(p.to_ansi(colour=True))


if __name__ == '__main__':
    pipe()


import teletext.vbi.deconvolve


# TODO: parser.add_argument('-H', '--headers', help='Synonym for --ansi --numbered --rows 0.', action='store_true')

@click.command()
@click.argument('input', type=click.File('rb'))
@click.option('--start', type=int, default=0, help='Start at the Nth line of the input file.')
@click.option('--stop', type=int, default=None, help='Stop before the Nth line of the input file.')
@click.option('--step', type=int, default=1, help='Process every Nth line from the input file.')
@click.option('--limit', type=int, default=None, help='Stop after processing N lines from the input file.')
@click.option('--mags', '-m', type=int, multiple=True, default=range(9))
@click.option('--rows', '-r', type=int, multiple=True, default=range(32))
#@click.option('-a/-t', '--ansi/--t42', default=sys.stdout.isatty(), help='Force output type.')
#@click.option('-n', '--numbered', is_flag=True, help='When output is ansi, number the lines according to position in input file.')
@click.option('-c', '--config', default='bt8x8_pal', help='Capture card configuration. Default: bt8x8_pal.')
#@click.option('-T', '--threads', type=int, default=1, help='Number of threads. Default: 1.')
@click.option('-C', '--force-cpu', is_flag=True, help='Disable CUDA even if it is available.')
@click.option('-e', '--extra_roll', type=int, default=4, help='')
#@click.option('-S', '--squash', type=int, default=1, help='Merge N consecutive rows to reduce output.')
def deconvolve(input, start, stop, step, limit, mags, rows, config, force_cpu, extra_roll, ):
    """Deconvolve raw VBI samples into Teletext packets."""

    from teletext.vbi.line import Line

    try:
        config = importlib.import_module('config_' + config)
    except ImportError:
        sys.stderr.write('No configuration file for ' + config + '.\n')

    Line.set_config(config)

    if force_cpu:
        Line.disable_cuda()

    global _extra_roll
    _extra_roll = extra_roll

    chunks = FileChunker(input, config.line_length, start, stop, step, limit)
    pbar = tqdm(chunks, unit=' Lines')

    lines = (Line(chunk, number) for number, chunk in pbar)
    packets = (l.deconvolve(extra_roll, mags, rows) for l in lines)

    for p in packets:
        if p is not None:
            pbar.write(p.to_ansi(colour=True))



