import click
import pathlib
import numpy as np
from tqdm import tqdm

from teletext.cli.clihelpers import carduser, chunkreader

@click.group()
def vbi():
    """Tools for analysing raw VBI samples."""
    pass


@vbi.command()
@click.argument('output', type=click.Path(writable=True))
@click.option('-d', '--diff', is_flag=True, help='User first differential of samples.')
@click.option('-s', '--show', is_flag=True, help='Show image when complete.')
@carduser(extended=True)
@chunkreader
def histogram(output, diff, show, chunker, config):
    from PIL import Image
    import colorsys

    n_lines = len(list(config.field_range))*2
    line_length = config.line_length - (1 if diff else 0)
    result = np.zeros((n_lines, 256, line_length), dtype=np.uint32)
    sel = np.arange(line_length)
    chunks = chunker(config.line_length * np.dtype(config.dtype).itemsize, config.field_lines, config.field_range)
    chunks = tqdm(chunks, unit='L', dynamic_ncols=True)
    for n, d in chunks:
        l = np.frombuffer(d, dtype=config.dtype) >> ((np.dtype(config.dtype).itemsize - 1) * 8)
        if diff:
            l = np.diff(l) + 128
        result[n%n_lines, l, sel] += 1

    for i in range(n_lines):
        for j in range(line_length):
            result[i,:,j] = 255*result[i,:,j]/np.max(result[i,:,j])

    # flip vertically
    result = result[:,::-1,:].reshape(-1, line_length)

    palette = np.zeros((256, 3), dtype=np.uint8)
    palette[0] = [0, 0, 0]
    for c in range(1, 256):
        palette[c] = [n * 255 for n in colorsys.hsv_to_rgb(c/1025, 1, 1)]

    rgb = palette[result]
    rgb[0::256, :] += 100
    rgb[0::32, :] = np.maximum(rgb[0::32, :], 32)

    i = Image.fromarray(rgb)
    if show:
        i.show()
    i.convert('RGB').save(output)


@vbi.command()
@carduser(extended=True)
@chunkreader
def plot(chunker, config):
    from teletext.gui.vbiplot import vbiplot

    vbiplot(chunker, config)


@vbi.command()
@carduser()
@chunkreader
@click.argument('output', type=click.File('wb'))
@click.option('--progress/--no-progress', default=True, help='Display progress bar.')
def copy(chunker, config, progress, output):
    """Copy input to output"""
    chunks = chunker(config.line_length * np.dtype(config.dtype).itemsize, config.field_lines, config.field_range)
    if progress:
        chunks = tqdm(chunks, unit='L', dynamic_ncols=True)
    for n, c in chunks:
        output.write(c)


@vbi.command()
@carduser()
@chunkreader
@click.argument('output', type=click.Path(), required=True)
@click.option('--progress/--no-progress', default=True, help='Display progress bar.')
def linesplit(chunker, config, progress, output):
    """Split VBI file into one file per line"""
    chunks = chunker(config.line_length * np.dtype(config.dtype).itemsize, config.field_lines, config.field_range)
    if progress:
        chunks = tqdm(chunks, unit='L', dynamic_ncols=True)
    output = pathlib.Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = [(output / f'{n:02x}.vbi').open("wb") for n in range(config.frame_lines)]
    for number, chunk in chunks:
        files[number % config.frame_lines].write(chunk)


@vbi.command()
@carduser()
@chunkreader
@click.argument('output', type=click.Path(), required=True)
@click.option('--progress/--no-progress', default=True, help='Display progress bar.')
def cluster(chunker, config, progress, output):
    """Split VBI file into clusters of similar lines"""
    import teletext.vbi.clustering
    chunks = chunker(config.line_length * np.dtype(config.dtype).itemsize, config.field_lines, config.field_range)
    if progress:
        chunks = tqdm(chunks, unit='L', dynamic_ncols=True)
    output = pathlib.Path(output)
    output.mkdir(parents=True, exist_ok=True)
    teletext.vbi.clustering.batch_cluster(chunks, output)


@vbi.command()
@carduser()
@click.argument('map', type=click.File('rb'), required=True)
@click.argument('output', type=click.File('wb'), required=True)
def rendermap(config, map, output):
    """Render cluster map to image"""
    import teletext.vbi.clustering
    teletext.vbi.clustering.rendermap(config, map, output)
