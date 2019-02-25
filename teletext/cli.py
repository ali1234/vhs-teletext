import multiprocessing
import os
import stat
import sys
from functools import wraps

import click
from tqdm import tqdm

from .file import FileChunker
from .mp import itermap
from .packet import Packet
from .stats import StatsList, MagHistogram, RowHistogram, Rejects, ErrorHistogram
from .subpage import Subpage
from .terminal import termify

from . import pipeline

from .vbi.config import Config


def filterparams(f):
    for d in [
        click.option('-m', '--mags', type=int, multiple=True, default=range(9), help='Limit output to specific magazines.'),
        click.option('-r', '--rows', type=int, multiple=True, default=range(32), help='Limit output to specific rows.'),
    ][::-1]:
        f = d(f)
    return f


def progressparams(progress=None, mag_hist=None, row_hist=None, err_hist=None):

    def p(f):
        for d in [
            click.option('--progress/--no-progress', default=progress, help='Display progress bar.'),
            click.option('--mag-hist/--no-mag-hist', default=mag_hist, help='Display magazine histogram.'),
            click.option('--row-hist/--no-row-hist', default=row_hist, help='Display row histogram.'),
            click.option('--err-hist/--no-err-hist', default=err_hist, help='Display error distribution.'),
        ][::-1]:
            f = d(f)
        return f
    return p


def carduser(extended=False):
    def c(f):
        if extended:
            for d in [
                click.option('--sample-rate', type=float, default=None, help='Override capture card sample rate (Hz).'),
                click.option('--line-start-range', type=(int, int), default=(None, None), help='Override capture card line start offset.'),
            ][::-1]:
                f = d(f)

        @click.option('-c', '--card', type=click.Choice(list(Config.cards.keys())), default='bt8x8', help='Capture device type. Default: bt8x8.')
        @click.option('--line-length', type=int, default=None, help='Override capture card samples per line.')
        @wraps(f)
        def wrapper(card, line_length=None, sample_rate=None, line_start_range=None, *args, **kwargs):
            if line_start_range == (None, None):
                line_start_range = None
            config = Config(card=card, line_length=line_length, sample_rate=sample_rate, line_start_range=line_start_range)
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


def chunkreader(f):
    @click.argument('input', type=click.File('rb'), default='-')
    @click.option('--start', type=int, default=0, help='Start at the Nth line of the input file.')
    @click.option('--stop', type=int, default=None, help='Stop before the Nth line of the input file.')
    @click.option('--step', type=int, default=1, help='Process every Nth line from the input file.')
    @click.option('--limit', type=int, default=None, help='Stop after processing N lines from the input file.')
    @wraps(f)
    def wrapper(input, start, stop, step, limit, *args, **kwargs):

        if input.isatty():
            raise click.UsageError('No input file and stdin is a tty - exiting.', )

        if 'progress' in kwargs and kwargs['progress'] is None:
            if stat.S_ISFIFO(os.fstat(input.fileno()).st_mode):
                kwargs['progress'] = False

        chunker = lambda size: FileChunker(input, size, start, stop, step, limit)

        return f(chunker=chunker, *args, **kwargs)
    return wrapper


def packetreader(f):
    @chunkreader
    @click.option('--wst', is_flag=True, default=False, help='Input is 43 bytes per packet (WST capture card format.)')
    @filterparams
    @progressparams()
    @wraps(f)
    def wrapper(chunker, wst, mags, rows, progress, mag_hist, row_hist, err_hist, *args, **kwargs):

        if wst:
            chunks = chunker(43)
            chunks = (c[:42] for c in chunks)
        else:
            chunks = chunker(42)

        if progress is None:
            progress = True

        if progress:
            chunks = tqdm(chunks, unit='P', dynamic_ncols=True)
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
        if progress and err_hist:
            packets = ErrorHistogram(packets)
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

        if 'progress' in kwargs and kwargs['progress'] is None:
            for attr, o in output:
                if o.isatty():
                    kwargs['progress'] = False

        packets = f(*args, **kwargs)

        for attr, o in output:
            packets = pipeline.to_file(packets, o, attr)

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
@click.argument('a', type=click.File('rb'))
@click.argument('b', type=click.File('rb'))
@filterparams
def diff(a, b, mags, rows):
    """Show side by side difference of two t42 streams."""
    for chunka, chunkb in zip(FileChunker(a, 42), FileChunker(b, 42)):
        pa = Packet(chunka[1], chunka[0])
        pb = Packet(chunkb[1], chunkb[0])
        if (pa.mrag.row in rows and pa.mrag.magazine in mags) or (pb.mrag.row in rows and pa.mrag.magazine in mags):
            if any(pa[:] != pb[:]):
                print(pa.to_ansi(), pb.to_ansi())


@teletext.command()
@packetwriter
@packetreader
def finders(packets):

    """Apply finders to fix up common packets."""

    for p in packets:
        if p.type == 'header':
            p.header.apply_finders()
        yield p


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
@click.option('-t', '--threads', type=int, default=multiprocessing.cpu_count(), help='Number of threads.')
@packetwriter
@packetreader
def spellcheck(packets, language, threads):

    """Spell check a t42 stream."""

    try:
        from .spellcheck import spellcheck_packets
    except ModuleNotFoundError as e:
        if e.name == 'enchant':
            raise click.UsageError(f'{e.msg}. PyEnchant is not installed. Spelling checker is not available.')
        else:
            raise e
    else:
        return itermap(spellcheck_packets, packets, threads, language=language)


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
@click.argument('outdir', type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True), required=True)
@click.option('-t', '--template', type=click.File('r'), default=None, help='HTML template.')
@packetreader
def html(packets, outdir, template):

    """Generate HTML files from the input stream."""

    from teletext.service import Service

    if template is not None:
        template = template.read()

    svc = Service.from_packets(packets)
    svc.to_html(outdir, template)


@teletext.command()
@click.argument('output', type=click.File('wb'), default='-')
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
@click.option('-p', '--pause', is_flag=True, help='Start the viewer paused.')
@carduser(extended=True)
@chunkreader
def vbiview(chunker, config, pause):

    """Display raw VBI samples with OpenGL."""

    try:
        from teletext.vbi.viewer import VBIViewer
    except ModuleNotFoundError as e:
        if e.name.startswith('OpenGL'):
            raise click.UsageError(f'{e.msg}. PyOpenGL is not installed. VBI viewer is not available.')
        else:
            raise e
    else:
        from teletext.vbi.line import Line

        Line.configure(config, force_cpu=True)

        chunks = chunker(config.line_length)
        lines = (Line(chunk, number) for number, chunk in chunks)

        VBIViewer(lines, config, pause=pause)


@teletext.command()
@click.option('-M', '--mode', type=click.Choice(['deconvolve', 'slice']), default='deconvolve', help='Deconvolution mode.')
@click.option('-C', '--force-cpu', is_flag=True, help='Disable CUDA even if it is available.')
@click.option('-t', '--threads', type=int, default=multiprocessing.cpu_count(), help='Number of threads.')
@carduser(extended=True)
@packetwriter
@chunkreader
@filterparams
@progressparams(progress=True, mag_hist=True)
@click.option('--rejects/--no-rejects', default=True, help='Display percentage of lines rejected.')
def deconvolve(chunker, mags, rows, config, mode, force_cpu, threads, progress, mag_hist, row_hist, err_hist, rejects):

    """Deconvolve raw VBI samples into Teletext packets."""

    from teletext.vbi.line import process_lines

    if force_cpu:
        sys.stderr.write('CUDA disabled by user request.\n')

    chunks = chunker(config.line_length)

    if progress:
        chunks = tqdm(chunks, unit='L', dynamic_ncols=True)
        if any((mag_hist, row_hist, rejects)):
            chunks.postfix = StatsList()

    packets = itermap(process_lines, chunks, threads, mode=mode, config=config, force_cpu=force_cpu, mags=mags, rows=rows)

    if progress and rejects:
        packets = Rejects(packets)
        chunks.postfix.append(packets)

    packets = (p for p in packets if isinstance(p, Packet))

    if progress and mag_hist:
        packets = MagHistogram(packets)
        chunks.postfix.append(packets)
    if progress and row_hist:
        packets = RowHistogram(packets)
        chunks.postfix.append(packets)
    if progress and err_hist:
        packets = ErrorHistogram(packets)
        chunks.postfix.append(packets)

    return packets


@teletext.group()
def training():
    """Training and calibration tools."""
    pass


@training.command()
@click.argument('output', type=click.File('wb'), default='-')
def generate(output):
    """Generate training samples for raspi-teletext."""
    from teletext.vbi.training import generate_lines
    generate_lines(output)


@training.command()
@click.argument('outdir', type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True), required=True)
@click.option('-t', '--threads', type=int, default=multiprocessing.cpu_count(), help='Number of threads.')
@carduser()
@chunkreader
@click.option('--progress/--no-progress', default=True, help='Display progress bar.')
@click.option('--rejects/--no-rejects', default=True, help='Display percentage of lines rejected.')
def split(chunker, outdir, config, threads, progress, rejects):
    """Split training recording into intermediate bins."""
    from teletext.vbi.training import process_training, split
    chunks = chunker(config.line_length)
    if progress:
        chunks = tqdm(chunks, unit='L', dynamic_ncols=True)

    results = itermap(process_training, chunks, threads, config=config)

    if progress and rejects:
        results = Rejects(results)
        chunks.postfix = StatsList()
        chunks.postfix.append(results)

    results = (r for r in results if isinstance(r, tuple))

    files = [open(os.path.join(outdir, f'training.{n:02x}.dat'), 'wb') for n in range(256)]

    split(results, files)


@training.command()
@click.argument('indir', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True)
@click.argument('output', type=click.File('wb'), default='-')
def squash(output, indir):
    """Squash the intermediate bins into a single file."""
    from teletext.vbi.training import squash
    squash(output, indir)


@training.command()
@chunkreader
def showbin(chunker):
    """Visually display an intermediate training bin."""
    import numpy as np

    bars = ' ▁▂▃▄▅▆▇█'
    bits = ' █'

    chunks = chunker(27)

    for n, chunk in chunks:
        arr = np.frombuffer(chunk, dtype=np.uint8)
        bi = ''.join(bits[n] for n in np.unpackbits(arr[:3][::-1])[::-1])
        by = ''.join(bars[n] for n in arr[3:]>>5)
        print(f'[{bi}] [{by}]')


@training.command()
@click.argument('input', type=click.File('rb'), required=True)
@click.argument('output', type=click.File('wb'), required=True)
@click.option('-m', '--mode', type=click.Choice(['full', 'parity', 'hamming']), default='full')
@click.option('-b', '--bits', type=(int, int), default=(3, 21))
def build(input, output, mode, bits):
    """Build pattern tables."""
    from teletext.coding import parity_encode, hamming8_enc
    from teletext.vbi.pattern import build_pattern

    if mode == 'parity':
        pattern_set = set(parity_encode(range(0x80)))
    elif mode == 'hamming':
        pattern_set = set(hamming8_enc)
    else:
        pattern_set = range(256)

    chunks = FileChunker(input, 27)
    chunks = tqdm(chunks, unit='P', dynamic_ncols=True)

    build_pattern(chunks, output, *bits, pattern_set)
