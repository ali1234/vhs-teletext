import math

import numpy as np
from teletext.parser import Parser

from PIL import Image
from PIL.PcfFontFile import *


class PcfFontFileUnicode(PcfFontFile):
    def unicode_glyphs(self):
        metrics = self._load_metrics()
        bitmaps = self._load_bitmaps(metrics)

        # map character code to bitmap index
        encoding = {}

        fp, format, i16, i32 = self._getformat(PCF_BDF_ENCODINGS)

        firstCol, lastCol = i16(fp.read(2)), i16(fp.read(2))
        firstRow, lastRow = i16(fp.read(2)), i16(fp.read(2))

        i16(fp.read(2))  # default

        nencoding = (lastCol - firstCol + 1) * (lastRow - firstRow + 1)

        for i in range(nencoding):
            encodingOffset = i16(fp.read(2))
            if encodingOffset != 0xFFFF:
                encoding[i + firstCol] = encodingOffset

        glyphs = {}

        for ch, ix in encoding.items():
            if ix is not None:
                x, y, l, r, w, a, d, f = metrics[ix]
                glyph = (w, 0), (l, d - y, x + l, d), (0, 0, x, y), bitmaps[ix]
                glyphs[ch] = glyph[3]

        return glyphs


def load_glyphs(fp):
    f = PcfFontFileUnicode(fp)
    return f.unicode_glyphs()


class PrinterImage(Parser):

    def __init__(self, tt, glyphs, colour=True, codepage=0):
        self.colour = colour
        self.column = 0
        self.glyphs = glyphs
        self.image = Image.new("P", (12*len(tt), 20))
        self.missing = set()
        super().__init__(tt)

    def emitcharacter(self, c):
        try:
            glyph = self.glyphs[ord(c)]
        except KeyError:
            self.missing.add(c)
        else:
            data = np.choose(glyph, (self._state['bg'], self._state['fg']))
            i = Image.fromarray(data.astype(np.uint8), "P")
            i = i.resize((
                i.width * (2 if self._state['dw'] else 1),
                i.height * (2 if self._state['dh'] else 1),
            ))
            self.image.paste(i, (self.column*12, 0))
        self.column += 1

    def parse(self):
        super().parse()
        return self.missing


def subpage_to_image(s, glyphs):
    img = Image.new("P", (12*40, 25*10))
    missing = set()
    img.putpalette([
        0, 0, 0,
        255, 0, 0,
        0, 255, 0,
        255, 255, 0,
        0, 0, 255,
        255, 0, 255,
        0, 255, 255,
        255, 255, 255,
    ])
    prnt = PrinterImage(s.header.displayable._array, glyphs)
    missing.update(prnt.parse())
    img.paste(prnt.image, (12*8, 0))

    for i in range(0, 24):
        # only draw the line if previous line does not contain double height code
        if i == 0 or np.all(s.displayable[i - 1, :] != 0x0d):
            prnt = PrinterImage(s.displayable[i, :], glyphs)
            missing.update(prnt.parse())
            img.paste(prnt.image, (0, (i+1)*10))

    img = img.resize((img.width, img.height*2))
    img = img.convert("RGB").resize((math.floor(img.width*1.2), img.height))
    result = Image.new("RGB", (720,576))
    result.paste(img, (
        (result.width - img.width) // 2,
        (result.height - img.height) // 2,
    ))
    img = result
    img._missing_glyphs = missing

    return img
