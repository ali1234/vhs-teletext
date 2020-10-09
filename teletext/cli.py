import itertools
import multiprocessing
import os
import pathlib
import platform

import sys
from collections import defaultdict

import click
from tqdm import tqdm

from .clihelpers import packetreader, packetwriter, paginated, progressparams, filterparams, carduser, chunkreader, \
    command, profileopts
from .file import FileChunker
from .mp import itermap
from .packet import Packet, np
from .stats import StatsList, MagHistogram, RowHistogram, Rejects, ErrorHistogram
from .subpage import Subpage

from . import pipeline

if os.name == 'nt' and platform.release() == '10' and platform.version() >= '10.0.14393':
    # Fix ANSI color in Windows 10 version 10.0.14393 (Windows Anniversary Update)
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)


@click.group()
@profileopts
@click.option('-u', '--unicode', is_flag=True, help='Use experimental Unicode 13.0 Terminal graphics.')
def teletext(unicode):
    """Teletext stream processing toolkit."""
    if unicode:
        from . import printer
        printer._unicode13 = True


@command(teletext)
@packetwriter
@paginated()
@click.option('--pagecount', 'n', type=int, default=0, help='Stop after n pages. 0 = no limit. Implies -P.')
@packetreader
def filter(packets, pages, subpages, paginate, n):

    """Demultiplex and display t42 packet streams."""

    if n:
        paginate = True

    if paginate:
        for pn, pl in enumerate(pipeline.paginate(packets, pages=pages, subpages=subpages), start=1):
            yield from pl
            if pn == n:
                return
    else:
        yield from packets


@command(teletext, name='list')
@click.option('-s', '--subpages', is_flag=True, help='Also list subpages.')
@paginated(always=True, filtered=False)
@packetreader
@progressparams(progress=True, mag_hist=True)
def _list(packets, subpages):

    """List pages present in a t42 stream."""

    import textwrap

    seen = set()
    try:
        for pl in pipeline.paginate(packets):
            s = Subpage.from_packets(pl)
            identifier = f'{s.mrag.magazine}{s.header.page:02x}'
            if subpages:
                identifier += f':{s.header.subpage:04x}'
            seen.add(identifier)
    except KeyboardInterrupt:
        print('\n')
    finally:
        print('\n'.join(textwrap.wrap(' '.join(sorted(seen)))))


@command(teletext)
@click.argument('pattern')
@paginated(always=True)
@packetreader
def split(packets, pattern, pages, subpages):

    """Split a t42 stream in to multiple files."""

    counts = defaultdict(int)

    for pl in pipeline.paginate(packets, pages=pages, subpages=subpages):
        subpage = Subpage.from_packets(pl)
        m = subpage.mrag.magazine
        p = subpage.header.page
        s = subpage.header.subpage
        c = counts[(m,p,s)]
        counts[(m,p,s)] += 1
        f = pathlib.Path(pattern.format(m=m, p=f'{p:02x}', s=f'{s:04x}', c=f'{c:04d}'))
        f.parent.mkdir(parents=True, exist_ok=True)
        with f.open('ab') as ff:
            ff.write(b''.join(p.bytes for p in pl))


@command(teletext)
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


@command(teletext)
@packetwriter
@packetreader
def finders(packets):

    """Apply finders to fix up common packets."""

    for p in packets:
        if p.type == 'header':
            p.header.apply_finders()
        yield p


@command(teletext)
@click.option('-d', '--min-duplicates', type=int, default=3, help='Only squash and output subpages with at least N duplicates.')
@click.option('-i', '--ignore-empty', is_flag=True, default=False, help='Ignore the emptiest duplicate packets instead of the earliest.')
@packetwriter
@paginated(always=True)
@packetreader
def squash(packets, min_duplicates, pages, subpages, ignore_empty):

    """Reduce errors in t42 stream by using frequency analysis."""

    for sp in pipeline.subpage_squash(
            pipeline.paginate(packets, pages=pages, subpages=subpages),
            min_duplicates=min_duplicates, ignore_empty=ignore_empty
    ):
        yield from sp.packets


@command(teletext)
@click.option('-l', '--language', default='en_GB', help='Language. Default: en_GB')
@click.option('-b', '--both', is_flag=True, help='Show packet before and after corrections.')
@click.option('-t', '--threads', type=int, default=multiprocessing.cpu_count(), help='Number of threads.')
@packetwriter
@packetreader
def spellcheck(packets, language, both, threads):

    """Spell check a t42 stream."""

    try:
        from .spellcheck import spellcheck_packets
    except ModuleNotFoundError as e:
        if e.name == 'enchant':
            raise click.UsageError(f'{e.msg}. PyEnchant is not installed. Spelling checker is not available.')
        else:
            raise e
    else:
        if both:
            packets, orig_packets = itertools.tee(packets, 2)
            packets = itermap(spellcheck_packets, packets, threads, language=language)
            try:
                while True:
                    yield next(orig_packets)
                    yield next(packets)
            except StopIteration:
                pass
        else:
            yield from itermap(spellcheck_packets, packets, threads, language=language)


@command(teletext)
@packetwriter
@paginated(always=True, filtered=False)
@packetreader
def service(packets):

    """Build a service carousel from a t42 stream."""

    from teletext.service import Service
    return Service.from_packets(packets)


@command(teletext)
@click.argument('input', type=click.File('rb'), default='-')
def interactive(input):

    """Interactive teletext emulator."""

    from . import interactive
    interactive.main(input)


@command(teletext)
@click.option('-e', '--editor', type=str, default='https://zxnet.co.uk/teletext/editor/#',
              show_default=True, help='Teletext editor URL.')
@paginated(always=True)
@packetreader
def urls(packets, editor, pages, subpages):

    """Paginate a t42 stream and print edit.tf URLs."""

    subpages = (Subpage.from_packets(pl) for pl in pipeline.paginate(packets, pages=pages, subpages=subpages))

    for s in subpages:
        print(f'{editor}{s.url}')


@command(teletext)
@click.argument('outdir', type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True), required=True)
@click.option('-t', '--template', type=click.File('r'), default=None, help='HTML template.')
@paginated(always=True, filtered=False)
@packetreader
def html(packets, outdir, template):

    """Generate HTML files from the input stream."""

    from teletext.service import Service

    if template is not None:
        template = template.read()

    svc = Service.from_packets(packets)
    svc.to_html(outdir, template)


@command(teletext)
@click.argument('output', type=click.File('wb'), default='-')
@click.option('-d', '--device', type=click.File('rb'), default='/dev/vbi0', help='Capture device.')
@carduser()
def record(output, device, config):

    """Record VBI samples from a capture device."""

    import struct
    import sys

    if output.name.startswith('/dev/vbi'):
        raise click.UsageError(f'Refusing to write output to VBI device. Did you mean -d?')

    chunks = FileChunker(device, config.line_length*config.field_lines*2)
    bar = tqdm(chunks, unit=' Frames')

    prev_seq = None
    dropped = 0

    try:
        for n, chunk in bar:
            output.write(chunk)
            if config.card == 'bt8x8':
                seq, = struct.unpack('<I', chunk[-4:])
                if prev_seq is not None and seq != (prev_seq + 1):
                   dropped += 1
                   sys.stderr.write('Frame drop? %d\n' % dropped)
                prev_seq = seq

    except KeyboardInterrupt:
        pass


@command(teletext)
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

        chunks = chunker(config.line_length * np.dtype(config.dtype).itemsize, config.field_lines, config.field_range)

        lines = (Line(chunk, number) for number, chunk in chunks)

        VBIViewer(lines, config, pause=pause)


@command(teletext)
@click.option('-M', '--mode', type=click.Choice(['deconvolve', 'slice']), default='deconvolve', help='Deconvolution mode.')
@click.option('-f', '--tape-format', type=click.Choice(['vhs', 'betamax']), default='vhs', help='Source VCR format.')
@click.option('-C', '--force-cpu', is_flag=True, help='Disable CUDA even if it is available.')
@click.option('-t', '--threads', type=int, default=multiprocessing.cpu_count(), help='Number of threads.')
@carduser(extended=True)
@packetwriter
@chunkreader
@filterparams
@progressparams(progress=True, mag_hist=True)
@click.option('--rejects/--no-rejects', default=True, help='Display percentage of lines rejected.')
def deconvolve(chunker, mags, rows, config, mode, force_cpu, threads, progress, mag_hist, row_hist, err_hist, rejects, tape_format):

    """Deconvolve raw VBI samples into Teletext packets."""

    from teletext.vbi.line import process_lines

    if force_cpu:
        sys.stderr.write('CUDA disabled by user request.\n')

    chunks = chunker(config.line_length * np.dtype(config.dtype).itemsize, config.field_lines, config.field_range)

    if progress:
        chunks = tqdm(chunks, unit='L', dynamic_ncols=True)
        if any((mag_hist, row_hist, rejects)):
            chunks.postfix = StatsList()

    packets = itermap(process_lines, chunks, threads, mode=mode, config=config, force_cpu=force_cpu, mags=mags, rows=rows, tape_format=tape_format)

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


@command(training)
@click.argument('output', type=click.File('wb'), default='-')
def generate(output):
    """Generate training samples for raspi-teletext."""
    from teletext.vbi.training import PatternGenerator
    PatternGenerator().to_file(output)


@command(training)
@click.argument('outdir', type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True), required=True)
@click.option('-t', '--threads', type=int, default=multiprocessing.cpu_count(), help='Number of threads.')
@carduser()
@chunkreader
@click.option('--progress/--no-progress', default=True, help='Display progress bar.')
@click.option('--rejects/--no-rejects', default=True, help='Display percentage of lines rejected.')
def split(chunker, outdir, config, threads, progress, rejects):
    """Split training recording into intermediate bins."""
    from teletext.vbi.training import process_training, split

    chunks = chunker(config.line_length * np.dtype(config.dtype).itemsize, config.field_lines, config.field_range)

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


@command(training, name='squash')
@click.argument('indir', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True)
@click.argument('output', type=click.File('wb'), default='-')
def training_squash(output, indir):
    """Squash the intermediate bins into a single file."""
    from teletext.vbi.training import squash
    squash(output, indir)


@command(training)
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


@command(training)
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


@command(training)
def similarities():
    from teletext.vbi.pattern import Pattern

    pattern = Pattern(os.path.dirname(__file__) + '/vbi/data-' + tape_format + '/parity.dat')

    print(pattern.similarities())


@command(training)
@click.option('-t', '--threads', type=int, default=multiprocessing.cpu_count(), help='Number of threads.')
@carduser()
@chunkreader
@click.option('--progress/--no-progress', default=True, help='Display progress bar.')
@click.option('--rejects/--no-rejects', default=True, help='Display percentage of lines rejected.')
def crifc(chunker, config, threads, progress, rejects):
    """Split training recording into intermediate bins."""
    from teletext.vbi.training import process_crifc

    chunks = chunker(config.line_length * np.dtype(config.dtype).itemsize, config.field_lines, config.field_range)

    if progress:
        chunks = tqdm(chunks, unit='L', dynamic_ncols=True)

    process_crifc(chunks, config=config)
