import importlib
import sys
from functools import wraps

import click
from tqdm import tqdm

from .file import FileChunker
from .packet import Packet
from .stats import StatsList, MagHistogram, RowHistogram, Rejects
from .subpage import Subpage
from .terminal import termify

from . import pipeline

from .vbi.config import Config


def to_file(packets, f, attr):
    if attr == 'auto':
        attr = 'debug' if f.isatty() else 'bytes'
    if f.isatty():
        for p in packets:
            with tqdm.external_write_mode():
                f.write(getattr(p, attr))
            yield p
    else:
        for p in packets:
            f.write(getattr(p, attr))
            yield p


def filterparams(f):
    for d in [
        click.option('--start', type=int, default=0, help='Start at the Nth line of the input file.'),
        click.option('--stop', type=int, default=None, help='Stop before the Nth line of the input file.'),
        click.option('--step', type=int, default=1, help='Process every Nth line from the input file.'),
        click.option('--limit', type=int, default=None, help='Stop after processing N lines from the input file.'),
        click.option('-m', '--mags', type=int, multiple=True, default=range(9), help='Limit output to specific magazines.'),
        click.option('-r', '--rows', type=int, multiple=True, default=range(32), help='Limit output to specific rows.'),
    ][::-1]:
        f = d(f)
    return f


def progressparams(progress=None, mag_hist=None, row_hist=None):

    def p(f):
        for d in [
            click.option('--progress/--no-progress', default=progress, help='Display progress bar.'),
            click.option('--mag-hist/--no-mag-hist', default=mag_hist, help='Display magazine histogram.'),
            click.option('--row-hist/--no-row-hist', default=row_hist, help='Display row histogram.'),
        ][::-1]:
            f = d(f)
        return f
    return p


def carduser(extended=False):
    def c(f):
        if extended:
            for d in [
                click.option('--sample-rate', type=float, default=None, help='Override capture card sample rate (Hz).'),
                click.option('--line-trim', type=int, default=None, help='Override capture card line trim.'),
                click.option('--line-start-range', type=(int, int), default=(None, None), help='Override capture card line start offset.'),
            ][::-1]:
                f = d(f)

        @click.option('-c', '--card', type=click.Choice(list(Config.cards.keys())), default='bt8x8', help='Capture device type. Default: bt8x8.')
        @click.option('--line-length', type=int, default=None, help='Override capture card samples per line.')
        @wraps(f)
        def wrapper(card, line_length=None, sample_rate=None, line_trim=None, line_start_range=None, *args, **kwargs):
            if line_start_range == (None, None):
                line_start_range = None
            config = Config(card=card, line_length=line_length, sample_rate=sample_rate, line_trim=line_trim, line_start_range=line_start_range)
            return f(config=config, *args,**kwargs)
        return wrapper
    return c


def termparams(f):
    @wraps(f)
    def t(windowed, less, **kwargs):
        termify(windowed, less)
        f(**kwargs)

    for d in [
        click.option('-W', '--windowed', is_flag=True, help='Connect stdout to a new terminal window.'),
        click.option('-L', '--less', is_flag=True, help='Page the output through less.'),
    ][::-1]:
        t = d(t)
    return t


def packetreader(f):
    @click.argument('input', type=click.File('rb'), default='-')
    @click.option('--wst', is_flag=True, default=False, help='Input is 43 bytes per packet (WST capture card format.)')
    @filterparams
    @progressparams()
    @wraps(f)
    def wrapper(input, wst, start, stop, step, limit, mags, rows, progress, mag_hist, row_hist, *args, **kwargs):

        if input.isatty():
            raise click.UsageError('No input file and stdin is a tty - exiting.', )

        if wst:
            chunks = FileChunker(input, 43, start, stop, step, limit)
            chunks = (c[:42] for c in chunks)
        else:
            chunks = FileChunker(input, 42, start, stop, step, limit)

        if progress:
            chunks = tqdm(chunks, unit='Pkts', dynamic_ncols=True)
            if any((mag_hist, row_hist)):
                chunks.postfix = StatsList()

        packets = (Packet(data, number) for number, data in chunks)
        packets = (p for p in packets if p.mrag.magazine in mags and p.mrag.row in rows)

        if progress and mag_hist:
            packets = MagHistogram(packets)
            chunks.postfix.append(packets)
        if progress and row_hist:
            packets = RowHistogram(packets)
            chunks.postfix.append(packets)

        return f(packets=packets, *args, **kwargs)

    return wrapper


def packetwriter(f):
    @click.option(
        '-o', '--output', type=(click.Choice(['auto', 'text', 'ansi', 'debug', 'bar', 'bytes']), click.File('wb')),
        multiple=True, default=[('auto', '-')]
    )
    @wraps(f)
    def wrapper(output, *args, **kwargs):

        packets = f(*args, **kwargs)

        for attr, o in output:
            packets = to_file(packets, o, attr)

        for p in packets:
            pass

    return wrapper


@click.group()
def teletext():
    """Teletext stream processing toolkit."""
    pass


@teletext.command()
@click.option('-p', '--pages', type=str, multiple=True, help='Limit output to specific pages.')
@click.option('-s', '--subpages', type=str, multiple=True, help='Limit output to specific subpages.')
@click.option('-P', '--paginate', is_flag=True, help='Sort rows into contiguous pages.')
@packetwriter
@packetreader
def filter(packets, pages, subpages, paginate):

    """Demultiplex and display t42 packet streams."""

    if pages is None or len(pages) == 0:
        pages = range(0x900)
    else:
        pages = {int(x, 16) for x in pages}
        paginate = True

    if subpages is None or len(subpages) == 0:
        subpages = range(0x3f7f)
    else:
        subpages = {int(x, 16) for x in subpages}
        paginate = True

    if paginate:
        for pl in pipeline.paginate(packets, pages=pages, subpages=subpages):
            yield from pl
    else:
        yield from packets


@teletext.command()
@click.option('-d', '--min-duplicates', type=int, default=3, help='Only squash and output subpages with at least N duplicates.')
@click.option('-p', '--pages', type=str, multiple=True, help='Limit output to specific pages.')
@click.option('-s', '--subpages', type=str, multiple=True, help='Limit output to specific subpages.')
@packetwriter
@packetreader
def squash(packets, min_duplicates, pages, subpages):

    """Reduce errors in t42 stream by using frequency analysis."""

    if pages is None or len(pages) == 0:
        pages = range(0x900)
    else:
        pages = {int(x, 16) for x in pages}

    if subpages is None or len(subpages) == 0:
        subpages = range(0x3f7f)
    else:
        subpages = {int(x, 16) for x in subpages}

    for sp in pipeline.subpage_squash(
            pipeline.paginate(packets, pages=pages, subpages=subpages),
            min_duplicates=min_duplicates
    ):
        yield from sp.packets


@teletext.command()
@click.option('-l', '--language', default='en_GB', help='Language. Default: en_GB')
@packetwriter
@packetreader
def spellcheck(packets, language):

    """Spell check a t42 stream."""

    from .spellcheck import SpellChecker
    return SpellChecker(language).spellcheck_iter(packets)


@teletext.command()
@packetwriter
@packetreader
def service(packets):

    """Build a service carousel from a t42 stream."""

    from teletext.service import Service
    return Service.from_packets(packets)


@teletext.command()
@click.argument('input', type=click.File('rb'), default='-')
def interactive(input):

    """Interactive teletext emulator."""

    from . import interactive
    interactive.main(input)


@teletext.command()
@click.option('-e', '--editor', required=True, help='Teletext editor URL.')
@packetreader
def urls(packets, editor):

    """Paginate a t42 stream and print edit.tf URLs."""

    subpages = (Subpage.from_packets(pl) for pl in pipeline.paginate(packets))

    for s in subpages:
        print(f'{editor}{s.url}')


@teletext.command()
@click.argument('outdir', type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True))
@click.option('-t', '--template', type=click.File('r'), default=None, help='HTML template.')
@packetreader
def html(packets, outdir, template, progress, mag_hist, row_hist):

    """Generate HTML files from the input stream."""

    from teletext.service import Service

    if template is not None:
        template = template.read()

    svc = Service.from_packets(packets)
    svc.to_html(outdir, template)


@teletext.command()
@click.option('-d', '--device', type=click.File('rb'), default='/dev/vbi0', help='Capture device.')
@carduser()
def record(output, device, config):

    """Record VBI samples from a capture device."""

    import struct
    import sys

    chunks = FileChunker(device, config.line_length*32)
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


@teletext.command()
@click.argument('input', type=click.File('rb'), default='-')
@carduser(extended=True)
def vbiview(input, config):

    """Display raw VBI samples with OpenGL."""

    if input.isatty():
        raise click.UsageError('No input file and stdin is a tty - exiting.')

    from teletext.vbi.viewer import VBIViewer
    from teletext.vbi.line import Line

    Line.set_config(config)
    Line.disable_cuda()

    chunks = FileChunker(input, config.line_length)
    bar = tqdm(chunks, unit=' Lines', dynamic_ncols=True)
    lines = (Line(chunk, number) for number, chunk in bar)

    VBIViewer(lines, config)


@teletext.command()
@click.argument('input', type=click.File('rb'), default='-')
@click.option('-C', '--force-cpu', is_flag=True, help='Disable CUDA even if it is available.')
@click.option('-e', '--extra_roll', type=int, default=4, help='')
@carduser(extended=True)
@packetwriter
@filterparams
@progressparams(progress=True, mag_hist=True)
@click.option('--rejects/--no-rejects', default=True, help='Display percentage of lines rejected.')
def deconvolve(input, start, stop, step, limit, mags, rows, config, force_cpu, extra_roll, progress, mag_hist, row_hist, rejects):

    """Deconvolve raw VBI samples into Teletext packets."""

    if input.isatty():
        raise click.UsageError('No input file and stdin is a tty - exiting.')

    from teletext.vbi.line import Line

    Line.set_config(config)

    if force_cpu:
        Line.disable_cuda()

    chunks = FileChunker(input, config.line_length, start, stop, step, limit)

    if progress:
        chunks = tqdm(chunks, unit=' Lines', dynamic_ncols=True)
        if any((mag_hist, row_hist, rejects)):
            chunks.postfix = StatsList()

    lines = (Line(chunk, number) for number, chunk in chunks)
    if progress and rejects:
        lines = Rejects(lines)
        chunks.postfix.append(lines)

    packets = (l.deconvolve(extra_roll, mags, rows) for l in lines)
    packets = (p for p in packets if p is not None)

    if progress and mag_hist:
        packets = MagHistogram(packets)
        chunks.postfix.append(packets)
    if progress and row_hist:
        packets = RowHistogram(packets)
        chunks.postfix.append(packets)

    return packets
