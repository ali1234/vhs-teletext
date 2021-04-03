import re

from .parser import Parser


class PrinterANSI(Parser):

    def __init__(self, tt, colour=True, codepage=0):
        self.colour = colour
        super().__init__(tt)

    def fgChanged(self):
        if self.colour:
            self._results.append('\033[3{fg}m'.format(**self._state))

    def bgChanged(self):
        if self.colour:
            self._results.append('\033[4{bg}m'.format(**self._state))

    def emitcharacter(self, c):
        self._results.append(c)

    def parse(self):
        self._results = []
        if self.colour:
            self._results.append('\033[37m\033[40m')
        super().parse()
        if self.colour:
            self._results.append('\033[0m')

    def __str__(self):
        return ''.join(self._results)


class PrinterHTML(Parser):

    def __init__(self, tt, fastext=None, pages_set=range(0x100), localcodepage=None, codepage=0):
        self.flinkopen = False
        self.fastext = fastext
        self.pages_set = pages_set

        # anchor for header links so we can bookmark a subpage
        self.anchor = ""

        super().__init__(tt, localcodepage, codepage)

    def ttchar(self, c):
        # Use the unicode characters produced by the base parser
        # but escape < and > so as not to break the HTML.
        c = Parser.ttchar(self, c)
        if c == ord('<'):
            return '&lt;'
        elif c == ord('>'):
            return '&gt;'
        else:
            return c

    def stateChanged(self):
        link = ''
        linkclose = ''
        if self.fastext:
            if self.flinkopen:
                linkclose = '</a>'
                self.flinkopen = False
            fg = self._state['fg']
            if fg in [1,2,3,6] and self.fastext[[1,2,3,6].index(fg)] in self.pages_set:
                link = '<a href="%s.html">' % self.fastext[[1,2,3,6].index(fg)]
                self.flinkopen = True

        self._results.extend([
            linkclose, '</span>',
            '<span class="f{fg} b{bg}'.format(**self._state),
            (' dh' if self._state['dh'] else ''),
            (' fl' if self._state['flash'] else ''),
            (' cn' if self._state['conceal'] else ''),
            (' bx' if self._state['boxed'] else ' nx'),
            '">', link
        ])

    def emitcharacter(self, c):
        self._results.append(c)

    def linkify(self, html):
        e = '([^0-9])([0-9]{3})([^0-9]|$)'
        def repl(match):
            if match.group(2) in self.pages_set:
                return '%s<a href="%s.html%s">%s</a>%s' % (match.group(1), match.group(2), self.anchor, match.group(2), match.group(3))
            else:
                return '%s%s%s' % (match.group(1), match.group(2), match.group(3))
        p = re.compile(e)
        return p.sub(repl, html)

    def parse(self):
        self._results = ['<span class="row"><span class="f7 b0 nx">']
        super().parse()
        self._results.append('</span></span>')
        if self.flinkopen:
            self._results.append('</a>')
        self._string = ''.join(self._results)
        if self.fastext is None:
            self._string = self.linkify(self._string)

    def __str__(self):
        return self._string
