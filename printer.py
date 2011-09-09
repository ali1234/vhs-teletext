
import re

class Printer(object):

    def __init__(self, tt, codepage=0):
        tt = tt&0x7f
        self.tt = tt&0x7f
        self.fg = 7
        self.bg = 0
        self.mosaic = False
        self.solid = True
        self.fasttext = False
        # ignored for now
        self.codepage = codepage

    def set_fasttext(self, data):
        self.fasttext = True

    def ttchar(self, c):
        if self.mosaic and (c < ord('@') or c > ord('_')):
            return unichr(c+0xee00) if self.solid else unichr(ord(c)+0xede0)
        else:
            if c == ord('#'):
                return unichr(ord('#'))
            elif c == ord('_'):
                return unichr(ord('#'))
            else:
                return unichr(c)

    def htmlspanstyle(self, fg=None, bg=None):
        return '<span class="f%d b%d">' % ((fg or self.fg), (bg or self.bg))

    def setstyle(self, html, fg=None, bg=None):
        if html:
            return '</span>'+self.htmlspanstyle()
        else:
            return '\033[3%dm\033[4%dm' % ((fg or self.fg), (bg or self.bg))

    def linkify(self, html):
        e = '([^0-9])([0-9]{3})([^0-9]|$)'
        def repl(match):
            return '%s<a href="%s.html">%s</a>%s' % (match.group(1), match.group(2), match.group(2), match.group(3))
        p = re.compile(e)
        return p.sub(repl, html)

    def transform(self, c, html=False):
        h = c&0xf0
        l = c&0x0f
        if h == 0x0:
            if l < 8:
                self.fg = l
                ret = ' '+self.setstyle(html)
                self.mosaic = False
            else:
                ret = ' '
        elif h == 0x10:
            if l < 8:
                self.fg = l
                ret = ' '+self.setstyle(html)
                self.mosaic = True
                self.solid = True
            elif l == 0x09:
                self.solid = True
                ret = ' '
            elif l == 0x10:
                self.solid = False
                ret = ' '
            elif l == 0xc:
                self.bg = 0
                ret = ' '+self.setstyle(html)
            elif l == 0x0d:
                self.bg = self.fg
                ret = self.setstyle(html)+' '
            else:
                ret = ' '
        else:
            ret = self.ttchar(c)

        return ret

    def string_html(self):
        head = self.htmlspanstyle(fg=7, bg=0)
        body = "".join([self.transform(x, html=True) for x in self.tt])
        foot = '</span>'
        if self.fasttext:
            pass
        else:
            body = self.linkify(body)
        return head+body.encode('utf8')+foot+'\n'

    def string_ansi(self):
        head = self.setstyle(html=False, fg=7, bg=0)
        body = "".join([self.transform(x, html=False) for x in self.tt])
        return head+"".body.encode('utf8')+'\033[0m'

