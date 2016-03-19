import re
import numpy as np

class PrinterANSI(object):

    def __init__(self, tt, colour=True, codepage=0):
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

        self.colour = colour


    def set_fasttext(self, data, mag):
        self.fasttext = True
        self.links = []
        for n in range(4):
            nn = n*6
            (p,e) = page_decode(data[nn+3:nn+5])
            ((s,m),e) = subcode_bcd_decode(data[nn+5:nn+9])
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


    def setstyle(self, fg=None, bg=None):
        if self.colour:
            return '\033[3%dm\033[4%dm' % ((fg or self.fg), (bg or self.bg))
        else:
            return ''


    def transform(self, c):
        h = c&0xf0
        l = c&0x0f
        if h == 0x0:
            if l < 8:
                self.fg = l
                self.conceal = False
                ret = ' '+self.setstyle()
                self.mosaic = False
            elif l == 0x8: # flashing
                self.flash = True
                ret = ' '+self.setstyle()
            elif l == 0x9: # steady
                self.flash = False
                ret = ' '+self.setstyle()
            elif l == 0xa: # flashing
                self.boxed = True
                ret = ' '+self.setstyle()
            elif l == 0xb: # steady
                self.boxed = False
                ret = ' '+self.setstyle()
            elif l == 0xc: # single height
                self.double = False
                ret = ' '+self.setstyle()
            elif l == 0xd: # double height
                self.double = True
                ret = ' '+self.setstyle()
            else:
                ret = ' '
                #print hex(int(c&0xff))
        elif h == 0x10:
            if l < 8:
                self.fg = l
                self.conceal = False
                ret = ' '+self.setstyle()
                self.mosaic = True
                self.solid = True
            elif l == 0x8: # conceal
                self.conceal = True
                ret = ' '+self.setstyle()
            elif l == 0x9:
                self.solid = True
                ret = ' '
            elif l == 0xa:
                self.solid = False
                ret = ' '
            elif l == 0xc:
                self.bg = 0
                ret = self.setstyle()+' '
            elif l == 0xd:
                self.bg = self.fg
                ret = self.setstyle()+' '
            else:
                ret = ' '
                #print hex(int(c&0xff))
        else:
            ret = self.ttchar(c)

        return ret


    def __str__(self):
        head = self.setstyle(fg=7, bg=0)
        body = "".join([self.transform(x) for x in self.tt])
        return head+body.encode('utf8')+('\033[0m' if self.colour else '')



class PrinterHTML(object):

    def __init__(self, tt, codepage=0):
        PrinterANSI.__init__(self, tt, codepage)


    def htmlspanstyle(self, fg=None, bg=None):
        return '<span class="f%d b%d%s%s%s%s">' % ((fg or self.fg), (bg or self.bg), 
                      (" dh" if self.double else ""), (" fl" if self.flash else ""),
                      (" cn" if self.conceal else ""), (" bx" if self.boxed else " nx"))


    def setstyle(self, html, fg=None, bg=None):
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


    def linkify(self, html):
        e = '([^0-9])([0-9]{3})([^0-9]|$)'
        def repl(match):
            return '%s<a href="%s.html%s">%s</a>%s' % (match.group(1), match.group(2), self.anchor, match.group(2), match.group(3))
        p = re.compile(e)
        return p.sub(repl, html)


    def __str__(self):
        head = self.htmlspanstyle(fg=7, bg=0)
        body = "".join([self.transform(x) for x in self.tt])
        foot = '</span>'
        if self.fasttext:
            if self.flinkopen:
                foot += '</a>'
        else:
            body = self.linkify(body)
        return head+body.encode('utf8')+foot+'\n'




