import importlib
import sys
from functools import wraps

import click
from tqdm import tqdm

from .file import FileChunker
from .packet import Packet
from .terminal import termify
from . import pipeline


def to_file(packets, f, attr):
    if attr == 'auto':
        attr = 'ansi' if f.isatty() else 'bytes'
    if f.isatty():
        for p in packets:
            with tqdm.external_write_mode():
                f.write(getattr(p, attr))
            yield p
    else:
        for p in packets:
            f.write(getattr(p, attr))
            yield p

def ioparams(f):
    for d in [
        click.argument('input', type=click.File('rb'), default='-'),
        click.option(
            '-o', '--output', type=(click.Choice(['auto', 'text', 'ansi', 'bar', 'bytes']), click.File('wb')),
            multiple=True, default=[('auto', '-')]
        ),
    ]:
        f = d(f)
    return f


def filterparams(f):
    for d in [
        click.option('--start', type=int, default=0, help='Start at the Nth line of the input file.'),
        click.option('--stop', type=int, default=None, help='Stop before the Nth line of the input file.'),
        click.option('--step', type=int, default=1, help='Process every Nth line from the input file.'),
        click.option('--limit', type=int, default=None, help='Stop after processing N lines from the input file.'),
        click.option('--mags', '-m', type=int, multiple=True, default=range(9), help='Limit output to specific magazines.'),
        click.option('--rows', '-r', type=int, multiple=True, default=range(32), help='Limit output to specific rows.'),
    ]:
        f = d(f)
    return f


def termparams(f):
    @wraps(f)
    def t(windowed, less, **kwargs):
        termify(windowed, less)
        f(**kwargs)

    for d in [
        click.option('--windowed', '-W', is_flag=True, help='Connect stdout to a new terminal window.'),
        click.option('--less', '-L', is_flag=True, help='Page the output through less.'),
    ]:
        t = d(t)
    return t


@click.group()
def teletext():
    """Teletext stream processing toolkit."""
    pass


@teletext.command()
@ioparams
@filterparams
@click.option('--pages', '-p', type=str, multiple=True, help='Limit output to specific pages.')
@click.option('--paginate', '-P', is_flag=True, help='Sort rows into contiguous pages.')
@termparams
def filter(input, output, start, stop, step, limit, mags, rows, pages, paginate):

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


@teletext.command()
@ioparams
def squash(input, output):

    """Reduce errors in t42 stream by using frequency analysis."""

    chunks = FileChunker(input, 42)
    packets = (Packet(data, number) for number, data in chunks)
    packets = pipeline.subpage_squash(packets)

    for attr, f in output:
        packets = to_file(packets, f, attr)

    for p in packets:
        pass


@teletext.command()
@ioparams
@click.option('--language', '-l', default='en_GB', help='Language. Default: en_GB')
def spellcheck(input, output, language):

    """Spell check a t42 stream."""

    from .spellcheck import SpellChecker

    chunks = FileChunker(input, 42)
    packets = (Packet(data, number) for number, data in chunks)
    s = SpellChecker(language)
    packets = s.spellcheck_iter(packets)

    for attr, f in output:
        packets = to_file(packets, f, attr)

    for p in packets:
        pass


@teletext.command()
@ioparams
def service(input, output):

    """Build a service carousel from a t42 stream."""

    from teletext.service import Service

    svc = Service()

    chunks = FileChunker(input, 42)
    packets = (Packet(data, number) for number, data in chunks)
    subpages = pipeline.paginate(packets, yield_func=pipeline.subpages)

    for s in subpages:
        svc.magazines[s.mrag.magazine].pages[s.header.page].subpages[s.header.subpage] = s

    for attr, f in output:
        packets = to_file(svc, f, attr)

    for p in packets:
        pass


@teletext.command()
@click.argument('input', type=click.File('rb'), default='-')
def interactive(input):

    """Interactive teletext emulator."""

    from . import interactive
    interactive.main(input)


@teletext.command()
@click.argument('input', type=click.File('rb'), default='-')
@click.option('--editor', '-e', required=True, help='Teletext editor URL.')
def urls(input, editor):

    """Paginate a t42 stream and print edit.tf URLs."""

    chunks = FileChunker(input, 42)
    packets = (Packet(data, number) for number, data in chunks)
    subpages = pipeline.paginate(packets, yield_func=pipeline.subpages)

    for s in subpages:
        print(f'{editor}{s.url}')


@teletext.group()
def vbi():
    """Commands dealing with raw VBI sampling."""
    pass


@vbi.command()
@click.option('--device', '-d', type=click.File('rb'), default='/dev/vbi0', help='Capture device.')
def record(output, device):

    """Record VBI samples from a capture device."""

    import struct
    import sys

    chunks = FileChunker(device, 2048*32)
    bar = tqdm(chunks, unit=' Frames')

    prev_seq = None
    dropped = 0

    try:
        for n, chunk in bar:
            output.write(chunk)
            seq, = struct.unpack('<I', chunk[-4:])
            if prev_seq is not None and seq != (prev_seq + 1):
               dropped += 1
               sys.stderr.write('Frame drop? %d\n' % dropped)
            prev_seq = seq

    except KeyboardInterrupt:
        pass


@vbi.command()
@click.argument('input', type=click.File('rb'), default='-')
@click.option('-c', '--config', default='bt8x8_pal', help='Capture card configuration. Default: bt8x8_pal.')
def view(input, config):

    """Display raw VBI samples with OpenGL."""

    from teletext.vbi.viewer import VBIViewer
    from teletext.vbi.line import Line

    try:
        config = importlib.import_module('config_' + config)
    except ImportError:
        sys.stderr.write('No configuration file for ' + config + '.\n')

    Line.set_config(config)
    Line.disable_cuda()

    chunks = FileChunker(input, config.line_length)
    bar = tqdm(chunks, unit=' Lines', dynamic_ncols=True)
    lines = (Line(chunk, number) for number, chunk in bar)

    VBIViewer(lines, config)


@vbi.command()
@ioparams
@filterparams
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

    chunks = FileChunker(input, config.line_length, start, stop, step, limit)
    bar = tqdm(chunks, unit=' Lines', dynamic_ncols=True)
    lines = (Line(chunk, number) for number, chunk in bar)
    packets = (l.deconvolve(extra_roll, mags, rows) for l in lines)
    packets = (p for p in packets if p is not None)

    for attr, f in output:
        packets = to_file(packets, f, attr)

    for p in packets:
        pass
