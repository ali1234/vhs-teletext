import cProfile
import os
import stat
from functools import wraps

import click
from tqdm import tqdm

from teletext import pipeline
from teletext.packet import Packet
from teletext.stats import StatsList, MagHistogram, RowHistogram, ErrorHistogram
from teletext.file import FileChunker
from teletext.vbi.config import Config

try:
    import plop.collector as plop
except ImportError:
    plop = None


def dcnparams(f):
    return click.option('-d', '--dcn', 'dcn', type=int, required=True, help='Data channel to read from.')(f)


def filterparams(enabled=True):
    def fp(f):
        if enabled:
            for d in [
                click.option('-m', '--mag', 'mags', type=int, multiple=True, default=range(9), help='Limit output to specific magazines. Can be specified multiple times.'),
                click.option('-r', '--row', 'rows', type=int, multiple=True, default=range(32), help='Limit output to specific rows. Can be specified multiple times.'),
            ][::-1]:
                f = d(f)
        return f
    return fp

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


def carduser(extended=True):
    def c(f):
        if extended:
            for d in [
                click.option('--sample-rate', type=float, default=None, help='Override capture card sample rate (Hz).'),
                click.option('--sample-rate-adjust', type=float, default=0, help='Adjustment to default capture card sample rate (Hz).'),
                click.option('--extra-roll', type=int, default=0, help='Shift line by N samples after locking to the packet.'),
                click.option('--line-start-range', type=(int, int), default=(None, None), help='Override capture card line start offset.'),
            ][::-1]:
                f = d(f)

        @click.option('-c', '--card', type=click.Choice(list(Config.cards.keys())), default='bt8x8', help='Capture device type. Default: bt8x8.')
        @click.option('--line-length', type=int, default=None, help='Override capture card samples per line.')
        @wraps(f)
        def wrapper(card, line_length=None, sample_rate=None, sample_rate_adjust=0, line_start_range=None, extra_roll=0, *args, **kwargs):
            if line_start_range == (None, None):
                line_start_range = None
            config = Config(card=card, line_length=line_length, sample_rate=sample_rate, sample_rate_adjust=sample_rate_adjust, line_start_range=line_start_range, extra_roll=extra_roll)
            return f(config=config, *args,**kwargs)
        return wrapper
    return c


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
            if hasattr(input, 'fileno') and stat.S_ISFIFO(os.fstat(input.fileno()).st_mode):
                kwargs['progress'] = False

        chunker = lambda size, flines=16, frange=range(0, 16): FileChunker(input, size, start, stop, step, limit, flines, frange)

        return f(chunker=chunker, *args, **kwargs)
    return wrapper

def packetreader(filtered=True):
    if filtered == 'data':
        filterdec = dcnparams
    else:
        filterdec = filterparams(filtered)

    def pr(f):
        @chunkreader
        @click.option('--wst', is_flag=True, default=False, help='Input is 43 bytes per packet (WST capture card format.)')
        @click.option('--ts', type=int, default=None, help='Input is MPEG transport stream. (Specify PID to extract.)')
        @filterdec
        @progressparams()
        @wraps(f)
        def wrapper(chunker, wst, ts, progress, mag_hist, row_hist, err_hist, *args, **kwargs):

            if wst and (ts is not None):
                raise click.UsageError('--wst and --ts can not be specified at the same time.')

            if wst:
                chunks = chunker(43)
                chunks = ((c[0],c[1][:42]) for c in chunks if c[1][0] != 0)
            elif ts is not None:
                from teletext.ts import pidextract
                chunks = chunker(188)
                chunks = pidextract(chunks, ts)
            else:
                chunks = chunker(42)

            if progress is None:
                progress = True

            if progress:
                chunks = tqdm(chunks, unit='P', dynamic_ncols=True)
                if any((mag_hist, row_hist)):
                    chunks.postfix = StatsList()

            packets = (Packet(data, number) for number, data in chunks)
            if 'mags' in kwargs and 'rows' in kwargs:
                mags = kwargs.pop('mags')
                rows = kwargs.pop('rows')
                packets = (p for p in packets if p.mrag.magazine in mags and p.mrag.row in rows)

            elif 'dcn' in kwargs:
                dcn = kwargs.pop('dcn')
                mags = (dcn & 0x7,)
                rows = (30 + (dcn>>3),)
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

    return pr


def paginated(always=False, filtered=True):
    def _paginated(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            paginate = always or kwargs['paginate']

            if filtered:
                pages = kwargs['pages']
                if pages is None or len(pages) == 0:
                    pages = range(0x900)
                else:
                    pages = {int(x, 16) for x in pages}
                    paginate = True
                kwargs['pages'] = pages

                subpages = kwargs['subpages']
                if subpages is None or len(subpages) == 0:
                    subpages = range(0x3f80)
                else:
                    subpages = {int(x, 16) for x in subpages}
                    paginate = True
                kwargs['subpages'] = subpages

            if paginate and 0 not in kwargs['rows']:
                raise click.BadArgumentUsage("Can't paginate when row 0 is filtered.")

            if not always:
                kwargs['paginate'] = paginate

            return f(*args, **kwargs)

        if filtered:
            wrapper = click.option('-s', '--subpage', 'subpages', type=str, multiple=True,
                      help='Limit output to specific subpage. Can be specified multiple times.')(wrapper)
            wrapper = click.option('-p', '--page', 'pages', type=str, multiple=True,
                      help='Limit output to specific page. Can be specified multiple times.')(wrapper)
        if not always:
            wrapper = click.option('-P', '--paginate', is_flag=True, help='Sort rows into contiguous pages.')(wrapper)

        return wrapper
    return _paginated


def packetwriter(f):
    @click.option(
        '-o', '--output', type=(click.Choice(['auto', 'text', 'ansi', 'debug', 'bar', 'bytes', 'hex', 'vbi']), click.File('wb')),
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


def profileopts(f):
    if plop is not None:
        @click.option('--profile', type=str, default=None)
        @click.pass_context
        @wraps(f)
        def group(ctx, profile, *args, **kwargs):
            ctx.ensure_object(dict)
            ctx.obj['PROFILE'] = profile
            return f(*args, **kwargs)
        return group
    else:
        return f


def command(group, *args, **kwargs):
    def deco(f):
        @group.command(*args, **kwargs)
        @click.pass_context
        @wraps(f)
        def cmd(ctx, *_args, **_kwargs):
            if plop is not None and ctx.obj['PROFILE'] is not None:
                # disable tqdm monitor thread as it messes with the profiling
                tqdm.monitor_interval = 0
                p = plop.Collector()
                p.start()
                try:
                    return f(*_args, **_kwargs)
                finally:
                    p.stop()
                    plop.FlamegraphFormatter().store(p, ctx.obj['PROFILE'])
            else:
                return f(*_args, **_kwargs)
        return cmd
    return deco
