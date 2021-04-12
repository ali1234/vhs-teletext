import multiprocessing
import os

import click
from tqdm import tqdm

from teletext.cli.clihelpers import carduser, chunkreader, \
    command
from teletext.file import FileChunker
from teletext.mp import itermap
from teletext.packet import Packet, np
from teletext.stats import StatsList, Rejects


@click.group()
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
