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
from teletext.cli.clihelpers import packetreader, packetwriter, paginated, \
    progressparams, filterparams, carduser, chunkreader
from teletext.file import FileChunker
from teletext.mp import itermap
from teletext.packet import Packet, np
from teletext.stats import StatsList, MagHistogram, RowHistogram, Rejects, ErrorHistogram
from teletext.subpage import Subpage
from teletext import pipeline
from teletext.cli.training import training
from teletext.cli.vbi import vbi
from teletext.cli.celp import celp

if os.name == 'nt' and platform.release() == '10' and platform.version() >= '10.0.14393':
    # Fix ANSI color in Windows 10 version 10.0.14393 (Windows Anniversary Update)
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)


@click.group(invoke_without_command=True, no_args_is_help=True)
@click.option('-u', '--unicode', is_flag=True, help='Use experimental Unicode 13.0 Terminal graphics.')
@click.version_option()
@click.help_option()
@click.option('--help-all', is_flag=True, help='Show help for all subcommands.')
@click.pass_context
def teletext(ctx, unicode, help_all):
    """Teletext stream processing toolkit."""
    if help_all:
        print(teletext.get_help(ctx))

        def help_recurse(group, ctx):
            for scmd in group.list_commands(ctx):
                cmd = group.get_command(ctx, scmd)
                nctx = click.Context(cmd, ctx, scmd)
                if isinstance(cmd, click.Group):
                    help_recurse(cmd, nctx)
                else:
                    click.echo()
                    click.echo(cmd.get_help(nctx))

        help_recurse(teletext, ctx)

    if unicode:
        from teletext import parser
        parser._unicode13 = True


teletext.add_command(training)
teletext.add_command(vbi)
teletext.add_command(celp)


@teletext.command()
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


@teletext.command()
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
            for p in pl:
                if bool(v) != bool(re.search(pattern, p.to_bytes_no_parity())):
                    yield from pl
                    if pn == n:
                        return
    else:
        for p in packets:
            if bool(v) != bool(re.search(pattern, p.to_bytes_no_parity())):
                yield p


@teletext.command(name='list')
@click.option('-c', '--count', is_flag=True, help='Show counts of each entry.')
@click.option('-s', '--subpages', is_flag=True, help='Also list subpages.')
@paginated(always=True, filtered=False)
@packetreader()
@progressparams(progress=True, mag_hist=True)
def _list(packets, count, subpages):

    """List pages present in a t42 stream."""

    import textwrap

    packets = (p for p in packets if not p.is_padding())

    seen = {}
    try:
        for pl in pipeline.paginate(packets):
            s = Subpage.from_packets(pl)
            identifier = f'{s.mrag.magazine}{s.header.page:02x}'
            if subpages:
                identifier += f':{s.header.subpage:04x}'
            if identifier in seen:
                seen[identifier]+=1
            else:
                seen[identifier]=1
    except KeyboardInterrupt:
        print('\n')
    finally:
        if count:
            maxdigits = len(str(max(seen.values())))
            formatstr="{page}/{count:0" + str(maxdigits) +"}"
        else:
            formatstr="{page}"
        seen = list(map(lambda e: formatstr.format(page = e[0], count = e[1]), seen.items()))
        print('\n'.join(textwrap.wrap(' '.join(sorted(seen)))))


@teletext.command()
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


@teletext.command()
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


@teletext.command()
@packetwriter
@packetreader()
def finders(packets):

    """Apply finders to fix up common packets."""

    for p in packets:
        if p.type == 'header':
            p.header.apply_finders()
        yield p


@teletext.command()
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


@teletext.command()
@click.option('-d', '--min-duplicates', type=int, default=3, help='Only squash and output subpages with at least N duplicates.')
@click.option('-t', '--threshold', type=int, default=-1, help='Max difference for squashing.')
@click.option('-i', '--ignore-empty', is_flag=True, default=False, help='Ignore the emptiest duplicate packets instead of the earliest.')
@packetwriter
@paginated(always=True)
@packetreader()
def squash(packets, min_duplicates, threshold, pages, subpages, ignore_empty):

    """Reduce errors in t42 stream by using frequency analysis."""

    packets = (p for p in packets if not p.is_padding())
    for sp in pipeline.subpage_squash(
            pipeline.paginate(packets, pages=pages, subpages=subpages),
            min_duplicates=min_duplicates, ignore_empty=ignore_empty,
            threshold=threshold
    ):
        yield from sp.packets


@teletext.command()
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


@teletext.command()
@packetwriter
@paginated(always=True, filtered=False)
@packetreader()
def service(packets):

    """Build a service carousel from a t42 stream."""

    from teletext.service import Service
    return Service.from_packets(p for p in packets if  not p.is_padding())


@teletext.command()
@packetwriter
@click.argument('directory', type=click.Path(exists=True, readable=True, file_okay=False, dir_okay=True))
def servicedir(directory):
    """Build a service from a directory of t42 files."""

    from teletext.servicedir import ServiceDir
    with ServiceDir(directory) as s:
        yield from s


@teletext.command()
@click.argument('input', type=click.File('rb'), default='-')
@click.option('-p', '--page', 'initial_page', type=str, default='100', help='Initial page.')
def interactive(input, initial_page):

    """Interactive teletext emulator."""

    from teletext import interactive
    interactive.main(input, int(initial_page, 16))


@teletext.command()
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

@teletext.command()
@click.argument('outdir', type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True), required=True)
@click.option('-f', '--font', type=click.File('rb'), help='PCF font for rendering.')
@paginated(always=True)
@packetreader()
def images(packets, outdir, font, pages, subpages):

    """Generate images for the input stream."""

    try:
        from teletext.image import subpage_to_image, load_glyphs
    except ModuleNotFoundError as e:
        if e.name == 'PIL':
            raise click.UsageError(
                f'{e.msg}. PIL is not installed. Image generation is not available.')
        else:
            raise e

    from teletext.service import Service

    glyphs = load_glyphs(font)

    packets = (p for p in packets if  not p.is_padding())
    svc = Service.from_packets(p for p in packets if not p.is_padding())

    subpages = tqdm(list(svc.all_subpages), unit="subpage")
    for s in subpages:
        image = subpage_to_image(s, glyphs)
        filename = f'P{s.mrag.magazine}{s.header.page:02x}-{s.header.subpage:04x}.png'
        subpages.set_description(filename, refresh=False)
        image.save(pathlib.Path(outdir) / f'P{s.mrag.magazine}{s.header.page:02x}-{s.header.subpage:04x}.png')
        if image._missing_glyphs:
            missing = ', '.join(f'{repr(c)} {hex(ord(c))}' for c in image._missing_glyphs)
            print(f'{filename} missing characters: {missing}')


@teletext.command()
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


@teletext.command()
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


@teletext.command()
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


@teletext.command()
@click.option('-M', '--mode', type=click.Choice(['deconvolve', 'slice']), default='deconvolve', help='Deconvolution mode.')
@click.option('-8', '--eight-bit', is_flag=True, help='Treat rows 1-25 as 8-bit data without parity check.')
@click.option('-f', '--tape-format', type=click.Choice(['vhs', 'betamax', 'grundig_2x4']), default='vhs', help='Source VCR format.')
@click.option('-C', '--force-cpu', is_flag=True, help='Disable GPU even if it is available.')
@click.option('-O', '--prefer-opencl', is_flag=True, default=False, help='Use OpenCL even if CUDA is available.')
@click.option('-t', '--threads', type=int, default=multiprocessing.cpu_count(), help='Number of threads.')
@click.option('-k', '--keep-empty', is_flag=True, help='Insert empty packets in the output when line could not be deconvolved.')
@carduser(extended=True)
@packetwriter
@chunkreader
@filterparams()
@paginated()
@progressparams(progress=True, mag_hist=True)
@click.option('--rejects/--no-rejects', default=True, help='Display percentage of lines rejected.')
def deconvolve(chunker, mags, rows, pages, subpages, paginate, config, mode, eight_bit, force_cpu, prefer_opencl, threads, keep_empty, progress, mag_hist, row_hist, err_hist, rejects, tape_format):

    """Deconvolve raw VBI samples into Teletext packets."""

    if keep_empty and paginate:
        raise click.UsageError("Can't keep empty packets when paginating.")

    from teletext.vbi.line import process_lines

    if force_cpu:
        sys.stderr.write('GPU disabled by user request.\n')

    chunks = chunker(config.line_length * np.dtype(config.dtype).itemsize, config.field_lines, config.field_range)

    if progress:
        chunks = tqdm(chunks, unit='L', dynamic_ncols=True)
        if any((mag_hist, row_hist, rejects)):
            chunks.postfix = StatsList()

    packets = itermap(process_lines, chunks, threads,
                      mode=mode, config=config,
                      force_cpu=force_cpu, prefer_opencl=prefer_opencl,
                      mags=mags, rows=rows,
                      tape_format=tape_format,
                      eight_bit=eight_bit)

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


