#!/usr/bin/env python

# * Copyright 2011 Alistair Buxton <a.j.buxton@gmail.com>
# *
# * License: This program is free software; you can redistribute it and/or
# * modify it under the terms of the GNU General Public License as published
# * by the Free Software Foundation; either version 3 of the License, or (at
# * your option) any later version. This program is distributed in the hope
# * that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# * GNU General Public License for more details.

# Tool to turn a teletext packet stream into something readable.
# If the first argument begins with 'h' the output will be in html.
# Otherwise it will be in rtf8.
# Usage:
# cat <data> | ./print.py

import sys
import numpy as np

from util import mrag, page, unhamm84, subcode

BLACK = '\033[30m'
RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
MAGENTA = '\033[35m'
CYAN = '\033[36m'
WHITE = '\033[37m'
RESET = '\033[0m'

BLACK = '\033[30m'
RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
MAGENTA = '\033[35m'
CYAN = '\033[36m'
WHITE = '\033[37m'
RESET = '\033[0m'

def setfg(c):
    return '\033[3%sm' % (str(c))

def setbg(c):
    return '\033[4%sm' % (str(c))

def ttchar(c, mosaic, solid):
    if mosaic and (c < '@' or c > '_'):
        return unichr(ord(c)+0xee00) if solid else unichr(ord(c)+0xede0)
    else:
        if c == '#':
            return unichr(ord('#'))
        elif c == '_':
            return unichr(ord('#'))
        else:
            return unichr(ord(c))


def ttext(data):
    fg = 7
    bg = 0
    mosaic = False
    solid = True
    output = u""
    output += WHITE
    for c in data:
        h = c&0xf0
        l = c&0x0f
        if h == 0x0:
            if l < 8:
                fg = l
                output += ' '+setfg(fg)
                mosaic = False
            else:
                output += ' '
        elif h == 0x10:
            if l < 8:
                fg = l
                output += ' '+setfg(fg)
                mosaic = True
                solid = True
            elif l == 0x09:
                solid = True
                output += ' '
            elif l == 0x10:
                solid = False
                output += ' '
            elif l == 0xc:
                bg = 0
                output += ' '+setbg(fg)
            elif l == 0x0d:
                bg = fg
                output += setbg(fg)+' '
            else:
                output += ' '
        else:
            output += ttchar(chr(c), mosaic, solid)
    output += RESET
    return output





html_colours = ['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white']

def setfgbg_html(f, b):
    return '</span><span style="color:%s; background:%s;">' % (html_colours[f], html_colours[b])

def ttext_html(data):
    fg = 7
    bg = 0
    mosaic = False
    solid = True
    output = u'<span style="color:white; background:black;">'
    for c in data:
        h = c&0xf0
        l = c&0x0f
        if h == 0x0:
            if l < 8:
                fg = l
                output += ' '+setfgbg_html(fg, bg)
                mosaic = False
            else:
                output += ' '
        elif h == 0x10:
            if l < 8:
                fg = l
                output += ' '+setfgbg_html(fg, bg)
                mosaic = True
                solid = True
            elif l == 0x09:
                solid = True
                output += ' '
            elif l == 0x10:
                solid = False
                output += ' '
            elif l == 0xc:
                bg = 0
                output += ' '+setfgbg_html(fg, bg)
            elif l == 0x0d:
                bg = fg
                output += setfgbg_html(fg, bg)+' '
            else:
                output += ' '
        else:
            output += ttchar(chr(c), mosaic, solid)
    output += '</span>'
    return output


def printit(bits, html=True):
    data = []
    parity = 0
    spaces = 0
    for i in range(len(bits)):
        x = ord(bits[i])
        c = x&0x7f
        data.append(c)

    if html:
        return ttext_html(data)
    else:
        return ttext(data)

header = """<html><head><meta http-equiv="Content-Type" content="text/html; charset=utf-8"><style>
@font-face {font-family: teletext2; src: url('teletext2.ttf');}
pre {font-family:teletext2;font-size:20px;line-height:20px;color:red;background:black;}
</style></head><body><pre>
"""

footer = """</pre></body></html>"""

def do_print(tt, html=False):
    ((m, r),e) = mrag(np.fromstring(tt[:2], dtype=np.uint8))
    sys.stdout.write("%1d %2d" % (m, r))
    if r == 0:
        (p,e) = page(np.fromstring(tt[2:4], dtype=np.uint8))
        sys.stdout.write("   P%1d%02x " % (m,p))
        sys.stdout.write(printit(tt[10:], html).encode('utf8'))
    elif r== 30: # broadcast service data
        # designation code
        (d,e) = unhamm84(ord(tt[2]))
        # initial page
        (p,e) = page(np.fromstring(tt[3:5], dtype=np.uint8))
        ((s,m),e) = subcode(np.fromstring(tt[5:9], dtype=np.uint8))
        sys.stdout.write(" %1d I%1d%02x:%04x " % (d, m, p, s))
        if d&2:
            sys.stdout.write("(PDC) ")
        else:
            sys.stdout.write("(NET) ")
        sys.stdout.write(printit(tt[22:], html).encode('utf8'))
    else:
        sys.stdout.write(printit(tt[2:], html).encode('utf8'))

    sys.stdout.write('\n')
    sys.stdout.flush()

if __name__=='__main__':
        
        if len(sys.argv) > 1 and sys.argv[1][0] == 'h':
            html = True
        else:
            html = False

        if html:
            print header

        while(True):
            tt = sys.stdin.read(42)
            if len(tt) < 42:
                exit(0)
            else:
                ((m,r),e) = mrag(np.fromstring(tt[:2], dtype=np.uint8))
                do_print(tt, html)
            sys.stdout.flush()

        if html:
            print footer
