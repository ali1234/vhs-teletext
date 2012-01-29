#!/usr/bin/env python

import re
from util import mrag, page, subcode_bcd, unhamm84
import numpy as np

class Printer(object):

    def __init__(self, tt, codepage=0):
        tt = tt&0x7f
        self.tt = tt&0x7f
        self.fg = 7
        self.bg = 0
        self.mosaic = False
        self.solid = True
        self.double = False
        self.flash = False
        self.conceal = False
        self.boxed = False
        self.fasttext = False
        self.flinkopen = False
        # ignored for now
        self.codepage = codepage

        # anchor for header links so we can bookmark a subpage
        self.anchor = ""

    def set_fasttext(self, data, mag):
        self.fasttext = True
        self.links = []
        for n in range(4):
            nn = n*6
            (p,e) = page(data[nn+3:nn+5])
            ((s,m),e) = subcode_bcd(data[nn+5:nn+9])
            m = (mag^m)&0x7
            if m == 0:
                m = 8
            self.links.append("%1d%02x" % (m,p))

    def ttchar(self, c):
        if self.mosaic and (c < ord('@') or c > ord('_')):
            return unichr(c+0xee00) if self.solid else unichr(c+0xede0)
        else:
            if c == ord('#'):
                return unichr(0xa3) # pound sign
            elif c == ord('_'):
                return unichr(ord('#'))
            elif c == ord('`'):
                return unichr(0x2014) # em dash
            elif c == ord('~'):
                return unichr(0xf7) # division symbol
            elif c == 0x7f:
                return unichr(0xe65f) # rectangle
            else:
                return unichr(c)

    def htmlspanstyle(self, fg=None, bg=None):
        return '<span class="f%d b%d%s%s%s%s">' % ((fg or self.fg), (bg or self.bg), 
                      (" dh" if self.double else ""), (" fl" if self.flash else ""),
                      (" cn" if self.conceal else ""), (" bx" if self.boxed else " nx"))

    def setstyle(self, html, fg=None, bg=None):
        if html:
            link = ''
            linkclose = ''
            if self.fasttext:
                if self.flinkopen:
                    linkclose = '</a>'
                    self.flinkopen = False
                if self.fg in [1,2,3,6]:
                    link = '<a href="%s.html">' % self.links[[1,2,3,6].index(self.fg)]
                    self.flinkopen = True
                
            return linkclose+'</span>'+self.htmlspanstyle()+link

        else:
            return '\033[3%dm\033[4%dm' % ((fg or self.fg), (bg or self.bg))

    def linkify(self, html):
        e = '([^0-9])([0-9]{3})([^0-9]|$)'
        def repl(match):
            return '%s<a href="%s.html%s">%s</a>%s' % (match.group(1), match.group(2), self.anchor, match.group(2), match.group(3))
        p = re.compile(e)
        return p.sub(repl, html)

    def transform(self, c, html=False):
        h = c&0xf0
        l = c&0x0f
        if h == 0x0:
            if l < 8:
                self.fg = l
                self.conceal = False
                ret = ' '+self.setstyle(html)
                self.mosaic = False
            elif l == 0x8: # flashing
                self.flash = True
                ret = ' '+self.setstyle(html)
            elif l == 0x9: # steady
                self.flash = False
                ret = ' '+self.setstyle(html)
            elif l == 0xa: # flashing
                self.boxed = True
                ret = ' '+self.setstyle(html)
            elif l == 0xb: # steady
                self.boxed = False
                ret = ' '+self.setstyle(html)
            elif l == 0xc: # single height
                self.double = False
                ret = ' '+self.setstyle(html)
            elif l == 0xd: # double height
                self.double = True
                ret = ' '+self.setstyle(html)
            else:
                ret = ' '
                #print hex(int(c&0xff))
        elif h == 0x10:
            if l < 8:
                self.fg = l
                self.conceal = False
                ret = ' '+self.setstyle(html)
                self.mosaic = True
                self.solid = True
            elif l == 0x8: # conceal
                self.conceal = True
                ret = ' '+self.setstyle(html)
            elif l == 0x9:
                self.solid = True
                ret = ' '
            elif l == 0xa:
                self.solid = False
                ret = ' '
            elif l == 0xc:
                self.bg = 0
                ret = self.setstyle(html)+' '
            elif l == 0xd:
                self.bg = self.fg
                ret = self.setstyle(html)+' '
            else:
                ret = ' '
                #print hex(int(c&0xff))
        else:
            ret = self.ttchar(c)

        return ret

    def string_html(self):
        head = self.htmlspanstyle(fg=7, bg=0)
        body = "".join([self.transform(x, html=True) for x in self.tt])
        foot = '</span>'
        if self.fasttext:
            if self.flinkopen:
                foot += '</a>'
        else:
            body = self.linkify(body)
        return head+body.encode('utf8')+foot+'\n'

    def string_ansi(self):
        head = self.setstyle(html=False, fg=7, bg=0)
        body = "".join([self.transform(x, html=False) for x in self.tt])
        return head+body.encode('utf8')+'\033[0m'








def do_print(tt):
    if type(tt) == type(''):
        tt = np.fromstring(tt, dtype=np.uint8)
    ret = ""
    ((m, r),e) = mrag(tt[:2])
    ret += "%1d %2d" % (m, r)
    if r == 0:
        (p,e) = page(tt[2:4])
        ((s,c),e) = subcode_bcd(tt[4:10])
        ret += "   P%1d%02x " % (m,p)
        ret += Printer(tt[10:]).string_ansi()
        ret += " %04x %x" % (s,c)
    elif r == 30: # broadcast service data
        # designation code
        (d,e) = unhamm84(tt[2])
        # initial page
        (p,e) = page(tt[3:5])
        ((s,m),e) = subcode_bcd(np.fromstring(tt[5:9], dtype=np.uint8))
        ret += " %1d I%1d%02x:%04x " % (d, m, p, s)
        if d&2:
            ret += "(PDC) "
        else:
            ret += "(NET) "
        ret += Printer(tt[22:]).string_ansi()
    elif r == 27: # broadcast service data
        # designation code
        (d,e) = unhamm84(tt[2])
        ret += " %1d " % (d)
        for n in range(6):
            nn = n*6
            (p,e) = page(tt[nn+3:nn+5])
            ((s,m),e) = subcode_bcd(tt[nn+5:nn+9])
            ret += " %1d%02x:%04x " % (m,p,s)
    else:
        ret += Printer(tt[2:]).string_ansi()

    return ret



if __name__=='__main__':
    import sys
    import numpy as np

    while(True):
        tt = sys.stdin.read(42)
        if len(tt) < 42:
            exit(0)
        else:
            tt = np.fromstring(tt, dtype=np.uint8)
            ((m,r),e) = mrag(tt[:2])
            #if r == 0 or r == 30: # to only print headers etc
            print do_print(tt)


