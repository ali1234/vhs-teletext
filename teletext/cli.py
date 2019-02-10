import importlib
import sys

import click
from tqdm import tqdm

from .file import FileChunker
from .packet import Packet
from .terminal import termify
from . import pipeline

def to_file(packets, f, attr):
    if attr == 'auto':
        attr = 'ansi' if f.isatty() else 'bytes'
    for p in packets:
        with tqdm.external_write_mode():
            tqdm.write(getattr(p, attr), file=f, end=b'')
        yield p


def baseparams(f):
    for d in [
        click.argument('input', type=click.File('rb'), default='-'),
        click.option('--start', type=int, default=0, help='Start at the Nth line of the input file.'),
        click.option('--stop', type=int, default=None, help='Stop before the Nth line of the input file.'),
        click.option('--step', type=int, default=1, help='Process every Nth line from the input file.'),
        click.option('--limit', type=int, default=None, help='Stop after processing N lines from the input file.'),
        click.option('--mags', '-m', type=int, multiple=True, default=range(9), help='Limit output to specific magazines.'),
        click.option('--rows', '-r', type=int, multiple=True, default=range(32), help='Limit output to specific rows.'),
        click.option('--pages', '-p', type=str, multiple=True, help='Limit output to specific pages.'),
        click.option('--paginate', '-P', is_flag=True, help='Sort rows into contiguous pages.'),
        click.option(
            '-o', '--output', type=(click.Choice(['auto', 'text', 'ansi', 'bar', 'bytes']), click.File('wb')),
            multiple=True, default=[('auto', '-')]
        ),
    ]:
        f = d(f)
    return f


def termopts(f):
    def t(windowed, less, *args, **kwargs):
        termify(windowed, less)
        f(*args, **kwargs)

    for d in [
        click.option('--windowed', '-W', is_flag=True, help='Connect stdout to a new terminal window.'),
        click.option('--less', '-L', is_flag=True, help='Page the output through less.'),
    ]:
        t = d(t)
    return t


@click.command()
@baseparams
@termopts
def pipe(input, start, stop, step, limit, mags, rows, pages, paginate, output):

    """Demultiplex and display t42 packet streams."""

    if pages is None or len(pages) == 0:
        pages = range(0x900)
    else:
        pages = {int(x, 16) for x in pages}
        paginate = True

    chunks = FileChunker(input, 42, start, stop, step, limit)
    bar = tqdm(chunks, unit=' Lines', dynamic_ncols=True)
    packets = (Packet(data, number) for number, data in bar)
    packets = (p for p in packets if p.mrag.magazine in mags and p.mrag.row in rows)
    if paginate:
        packets = pipeline.paginate(packets, pages)

    for attr, f in output:
        packets = to_file(packets, f, attr)

    for p in packets:
        pass


@click.command()
@baseparams
@click.option('-c', '--config', default='bt8x8_pal', help='Capture card configuration. Default: bt8x8_pal.')
@click.option('-C', '--force-cpu', is_flag=True, help='Disable CUDA even if it is available.')
@click.option('-e', '--extra_roll', type=int, default=4, help='')
def deconvolve(input, start, stop, step, limit, mags, rows, output, config, force_cpu, extra_roll, ):

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
    bar = tqdm(chunks, unit=' Lines', dynamic_ncols=True)
    lines = (Line(chunk, number) for number, chunk in bar)
    packets = (l.deconvolve(extra_roll, mags, rows) for l in lines)
    packets = (p for p in packets if p is not None)

    for attr, f in output:
        packets = to_file(packets, f, attr)

    for p in packets:
        pass
