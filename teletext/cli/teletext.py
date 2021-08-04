import itertools
import multiprocessing
import os
import pathlib
import platform

import sys
from collections import defaultdict

import click
from tqdm import tqdm

from teletext.charset import g0
from teletext.cli.clihelpers import packetreader, packetwriter, paginated, progressparams, filterparams, carduser, chunkreader, \
    command, profileopts
from teletext.file import FileChunker
from teletext.mp import itermap
from teletext.packet import Packet, np
from teletext.stats import StatsList, MagHistogram, RowHistogram, Rejects, ErrorHistogram
from teletext.subpage import Subpage
from teletext import pipeline
from teletext.cli.training import training
from teletext.cli.vbi import vbi

if os.name == 'nt' and platform.release() == '10' and platform.version() >= '10.0.14393':
    # Fix ANSI color in Windows 10 version 10.0.14393 (Windows Anniversary Update)
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)


@click.group()
@profileopts
@click.option('-u', '--unicode', is_flag=True, help='Use experimental Unicode 13.0 Terminal graphics.')
@click.version_option()
def teletext(unicode):
    """Teletext stream processing toolkit."""
    if unicode:
        from . import printer
        printer._unicode13 = True


teletext.add_command(training)
teletext.add_command(vbi)


@command(teletext)
@packetwriter
@paginated()
@click.option('--pagecount', 'n', type=int, default=0, help='Stop after n pages. 0 = no limit. Implies -P.')
@click.option('-k', '--keep-empty', is_flag=True, help='Keep empty packets in the output.')
@packetreader()
def filter(packets, pages, subpages, paginate, n, keep_empty):

    """Demultiplex and display t42 packet streams."""

    if n:
        paginate = True

    if not keep_empty:
        packets = (p for p in packets if not p.is_padding())

    if paginate:
        for pn, pl in enumerate(pipeline.paginate(packets, pages=pages, subpages=subpages), start=1):
            yield from pl
            if pn == n:
                return
    else:
        yield from packets


@command(teletext)
@packetwriter
@paginated()
@click.argument('regex', type=str)
@click.option('-v', is_flag=True, help='Invert matches.')
@click.option('-i', is_flag=True, help='Ignore case.')
@click.option('--pagecount', 'n', type=int, default=0, help='Stop after n pages. 0 = no limit. Implies -P.')
@click.option('-k', '--keep-empty', is_flag=True, help='Keep empty packets in the output.')
@packetreader()
def grep(packets, pages, subpages, paginate, regex, v, i, n, keep_empty):

    """Filter packets with a regular expression."""

    import re

    pattern = re.compile(regex.encode('ascii'), re.IGNORECASE if i else 0)

    if n:
        paginate = True

    if not keep_empty:
        packets = (p for p in packets if not p.is_padding())

    if paginate:
        for pn, pl in enumerate(pipeline.paginate(packets, pages=pages, subpages=subpages), start=1):
            for p in packets:
                if bool(v) != bool(re.search(pattern, p.to_bytes_no_parity())):
                    yield from pl
                    if pn == n:
                        return
    else:
        for p in packets:
            if bool(v) != bool(re.search(pattern, p.to_bytes_no_parity())):
                yield p


@command(teletext, name='list')
@click.option('-s', '--subpages', is_flag=True, help='Also list subpages.')
@paginated(always=True, filtered=False)
@packetreader()
@progressparams(progress=True, mag_hist=True)
def _list(packets, subpages):

    """List pages present in a t42 stream."""

    import textwrap

    packets = (p for p in packets if not p.is_padding())

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
@packetreader()
def split(packets, pattern, pages, subpages):

    """Split a t42 stream in to multiple files."""

    packets = (p for p in packets if not p.is_padding())
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
@filterparams()
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
@packetreader()
def finders(packets):

    """Apply finders to fix up common packets."""

    for p in packets:
        if p.type == 'header':
            p.header.apply_finders()
        yield p


@command(teletext)
@packetreader(filtered=False)
@click.option('-l', '--lines', type=int, default=32, help='Number of recorded lines per frame.')
@click.option('-f', '--frames', type=int, default=250, help='Number of frames to squash.')
def scan(packets, lines, frames):

    """Filter a t42 stream down to headers and bsdp, with squashing."""

    from teletext.pipeline import packet_squash, bsdp_squash_format1, bsdp_squash_format2
    bars = '_:|I'

    while True:
        actives = np.zeros((lines,), dtype=np.uint32)
        headers = [[], [], [], [], [], [], [], [], []]
        service = [[], []]
        start = None
        try:
            for i in range(frames):
                for n, p in enumerate(itertools.islice(packets, lines)):
                    if start is None:
                        start = p._number
                    if not p.is_padding():
                        if p.type == 'header':
                            p.header.apply_finders()
                        actives[n] += 1
                        if p.mrag.row == 0:
                            headers[p.mrag.magazine].append(p)
                        elif p.mrag.row == 30 and p.mrag.magazine == 8:
                            if p.broadcast.dc in [0, 1]:
                                service[0].append(p)
                            elif p.broadcast.dc in [2, 3]:
                                service[1].append(p)

        except StopIteration:
            pass
        if start is None:
            return
        active_group = 1*(actives>0) + 1*(actives>(frames/2)) + 1*(actives==frames)
        print(f'{start:8d}', '['+''.join(bars[a] for a in active_group)+']', end=' ')
        for h in headers:
            if h:
                print(packet_squash(h).header.displayable.to_ansi(), end=' ')
                break
        for s in service:
            if s:
                print(packet_squash(s).broadcast.displayable.to_ansi(), end=' ')
                break
        if service[0]:
            print(bsdp_squash_format1(service[0]), end=' ')
        if service[1]:
            print(bsdp_squash_format2(service[1]), end=' ')
        print()


@command(teletext)
@packetreader(filtered=False)
def celp(packets):
    """Dump CELP packets from t42 stream. We don't know how to decode them."""
    for p in packets:
        if p.mrag.magazine == 4 and p.mrag.row in [30, 31]:
            control = p._array[2]
            service = p._array[3]
            frame0 = p._array[4:23]
            frame1 = p._array[23:42]
            print("Service:" if p.mrag.row == 30 else "Control:", control)
            print("Fade:" if p.mrag.row == 30 else "Service:", service)
            print(frame0.tobytes().hex())
            print(frame1.tobytes().hex())

@command(teletext)
@click.option('-d', '--min-duplicates', type=int, default=3, help='Only squash and output subpages with at least N duplicates.')
@click.option('-i', '--ignore-empty', is_flag=True, default=False, help='Ignore the emptiest duplicate packets instead of the earliest.')
@packetwriter
@paginated(always=True)
@packetreader()
def squash(packets, min_duplicates, pages, subpages, ignore_empty):

    """Reduce errors in t42 stream by using frequency analysis."""

    packets = (p for p in packets if not p.is_padding())
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
@packetreader()
def spellcheck(packets, language, both, threads):

    """Spell check a t42 stream."""

    try:
        from teletext.spellcheck import spellcheck_packets
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
@packetreader()
def service(packets):

    """Build a service carousel from a t42 stream."""

    from teletext.service import Service
    return Service.from_packets(p for p in packets if  not p.is_padding())


@command(teletext)
@click.argument('input', type=click.File('rb'), default='-')
def interactive(input):

    """Interactive teletext emulator."""

    from teletext import interactive
    interactive.main(input)


@command(teletext)
@click.option('-e', '--editor', type=str, default='https://zxnet.co.uk/teletext/editor/#',
              show_default=True, help='Teletext editor URL.')
@paginated(always=True)
@packetreader()
def urls(packets, editor, pages, subpages):

    """Paginate a t42 stream and print edit.tf URLs."""

    packets = (p for p in packets if  not p.is_padding())
    subpages = (Subpage.from_packets(pl) for pl in pipeline.paginate(packets, pages=pages, subpages=subpages))

    for s in subpages:
        print(f'{editor}{s.url}')


@command(teletext)
@click.argument('outdir', type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True), required=True)
@click.option('-t', '--template', type=click.File('r'), default=None, help='HTML template.')
@click.option('--localcodepage', type=click.Choice(g0.keys()), default=None, help='Select codepage for Local Code of Practice')
@paginated(always=True, filtered=False)
@packetreader()
def html(packets, outdir, template, localcodepage):

    """Generate HTML files from the input stream."""

    from teletext.service import Service

    if template is not None:
        template = template.read()

    svc = Service.from_packets(p for p in packets if not p.is_padding())
    svc.to_html(outdir, template, localcodepage)


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
@click.option('-f', '--tape-format', type=click.Choice(['vhs', 'betamax', 'grundig_2x4']), default='vhs', help='Source VCR format.')
@click.option('-n', '--n-lines', type=int, default=None, help='Number of lines to display. Overrides card config.')
@carduser(extended=True)
@chunkreader
def vbiview(chunker, config, pause, tape_format, n_lines):

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

        Line.configure(config, force_cpu=True, tape_format=tape_format)

        if n_lines is not None:
            chunks = chunker(config.line_length * np.dtype(config.dtype).itemsize, n_lines, range(n_lines))
        else:
            chunks = chunker(config.line_length * np.dtype(config.dtype).itemsize, config.field_lines, config.field_range)

        lines = (Line(chunk, number) for number, chunk in chunks)

        VBIViewer(lines, config, pause=pause, nlines=n_lines)


@command(teletext)
@click.option('-M', '--mode', type=click.Choice(['deconvolve', 'slice']), default='deconvolve', help='Deconvolution mode.')
@click.option('-f', '--tape-format', type=click.Choice(['vhs', 'betamax', 'grundig_2x4']), default='vhs', help='Source VCR format.')
@click.option('-C', '--force-cpu', is_flag=True, help='Disable CUDA even if it is available.')
@click.option('-t', '--threads', type=int, default=multiprocessing.cpu_count(), help='Number of threads.')
@click.option('-k', '--keep-empty', is_flag=True, help='Insert empty packets in the output when line could not be deconvolved.')
@carduser(extended=True)
@packetwriter
@chunkreader
@filterparams()
@paginated()
@progressparams(progress=True, mag_hist=True)
@click.option('--rejects/--no-rejects', default=True, help='Display percentage of lines rejected.')
def deconvolve(chunker, mags, rows, pages, subpages, paginate, config, mode, force_cpu, threads, keep_empty, progress, mag_hist, row_hist, err_hist, rejects, tape_format):

    """Deconvolve raw VBI samples into Teletext packets."""

    if keep_empty and paginate:
        raise click.UsageError("Can't keep empty packets when paginating.")

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

    if keep_empty:
        packets = (p if isinstance(p, Packet) else Packet() for p in packets)
    else:
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

    if paginate:
        for p in pipeline.paginate(packets, pages=pages, subpages=subpages):
            yield from p
    else:
        yield from packets


