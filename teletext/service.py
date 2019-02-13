import datetime
import os
import textwrap

from collections import defaultdict

from tqdm import tqdm

from .packet import Packet
from . import pipeline


class Page(object):
    def __init__(self):
        self.subpages = {}
        self._iter = self._gen()

    def _gen(self):
        while True:
            if len(self.subpages) > 0:
                yield from sorted(self.subpages.items())
            yield 0x3f7f, None

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iter)


class Magazine(object):
    def __init__(self, title='Unnamed  '):
        self.title = title
        self.pages = defaultdict(Page)
        self._iter = self._gen()

    def _gen(self):
        while True:
            for pageno, page in sorted(self.pages.items()):
                spno, subpage = next(page)
                if subpage is None:
                    p = Packet()
                    p.mrag.row = 0
                    p.header.page = 0xff
                    p.header.subpage = spno
                    yield p
                else:
                    subpage.header.page = pageno
                    subpage.header.subpage = spno
                    yield from subpage.packets

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iter)


class Service(object):
    def __init__(self, replace_headers=False):
        self.magazines = defaultdict(Magazine)
        self.priorities = [1,1,1,1,1,1,1,1]
        self.replace_headers = replace_headers
        self._iter = self._gen()

    def header(self, title, mag, page):
        t = datetime.datetime.now()
        return '%9s%1d%02x' % (title, mag, page) + t.strftime(" %a %d %b\x03%H:%M/%S")

    def _gen(self):
        while True:
            for n,m in sorted(self.magazines.items()):
                for count in range(self.priorities[n&0x7]):
                    packet = next(m)
                    packet.mrag.magazine = n
                    if self.replace_headers and packet.type == 'header':
                        packet.header.displayable.place_string(self.header(m.title, n, packet.header.page))
                    yield packet

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iter)

    def packets(self, n):
        for i in range(n):
            yield next(self)

    @property
    def pages_set(self):
        return set(f'{m}{p:02x}' for m, mag in self.magazines.items() for p, _ in mag.pages.items())

    @classmethod
    def from_packets(cls, packets):
        svc = cls(replace_headers=False)
        subpages = pipeline.paginate(packets, yield_func=pipeline.subpages)

        for s in subpages:
            svc.magazines[s.mrag.magazine].pages[s.header.page].subpages[s.header.subpage] = s

        return svc

    def to_html(self, outdir, template=None):

        pages_set = self.pages_set

        if template is None:
            template = textwrap.dedent("""\
                <html>
                    <head>
                        <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
                        <title>Page {page}</title>
                        <link rel="stylesheet" type="text/css" href="teletext.css" title="Default Style"/>
                        <link rel="alternative stylesheet" type="text/css" href="teletext-noscanlines.css" title="No Scanlines"/>
                        <script type="text/javascript" src="cssswitch.js"></script>
                    </head>
                    <body onload="set_style_from_cookie()">
                    {body}
                    </body>
                </html>
            """)

        for magazineno, magazine in tqdm(self.magazines.items(), desc='Magazines', unit='M'):
            for pageno, page in tqdm(magazine.pages.items(), desc='Pages', unit='P'):
                pagestr = f'{magazineno}{pageno:02x}'
                outfile = open(os.path.join(outdir, f'{pagestr}.html'), 'w')
                body = '\n'.join(
                    subpage.to_html(pages_set) for subpage in page.subpages.values()
                )
                outfile.write(template.format(page=pagestr, body=body))

