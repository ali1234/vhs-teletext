import click
import numpy as np

from teletext.cli.clihelpers import command, carduser, chunkreader

@click.group()
def vbi():
    """Tools for analysing raw VBI samples."""
    pass



@command(vbi)
@click.argument('output', type=click.Path(writable=True))
@click.option('-d', '--diff', is_flag=True, help='User first differential of samples.')
@click.option('-s', '--show', is_flag=True, help='Show image when complete.')
@carduser(extended=True)
@chunkreader
def histogram(output, diff, show, chunker, config):
    from PIL import Image

    line_length = config.line_length - (1 if diff else 0)
    result = np.zeros((config.field_lines*2, 256, line_length), dtype=np.uint32)
    sel = np.arange(line_length)
    chunks = chunker(config.line_length * np.dtype(config.dtype).itemsize, config.field_lines, config.field_range)
    for n, d in chunks:
        l = np.frombuffer(d, dtype=config.dtype) >> ((np.dtype(config.dtype).itemsize - 1) * 8)
        if diff:
            l = np.diff(l) + 128
        result[n%(config.field_lines*2)][l, sel] += 1

    # flip vertically
    result = result[:,::-1,:].reshape(-1, line_length)

    ne = (result > 0) * 16
    norm = 1000*result/np.max(result)

    i = Image.fromarray(256 - (ne+norm))
    if show:
        i.show()
    i.convert('RGB').save(output)
