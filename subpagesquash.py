#!/usr/bin/env python

import sys, os
import numpy as np

from util import subcode_bcd, mrag, page, bitwise_mode
from printer import Printer, do_print
from page import Page
from fragment import fragments

class Squasher(object):
    def __init__(self, filename):
        data = file(filename, 'rb')
        self.pages = []
        self.page_count = 0
        print filename,
        done = False
        while not done:
            p = data.read(42*26)
            if len(p) < (42*26):
                done = True
            else:
                p = Page(np.fromstring(p, dtype=np.uint8))
                for flist in fragments:
                    tmp = [(f.test(p),n,f) for n,f in enumerate(flist)]
                    ans = max(tmp)
                    if ans[0] > 0:
                        ans[2].fix(p)
                self.pages.append(p)
                self.page_count += 1
        print "%5d" % self.page_count,

        self.subcodes = self.guess_subcodes()
        self.subcode_count = len(self.subcodes)

        self.m = self.pages[0].m
        self.p = self.pages[0].p

        for i in range(3):
         unique_pages = self.hamming()
         squashed_pages = []
         for pl in unique_pages:
             page = Page(self.squash(pl))
             squashed_pages += [page]*len(pl)
         self.pages = squashed_pages

        unique_pages = self.hamming()
        squashed_pages = []
        for pl in unique_pages:
          if len(pl) > 1 or len(unique_pages) == 1:
            squashed_pages += [Page(self.squash(pl))]


        # sort it
        sorttmp = [(p.s, p) for p in squashed_pages]
        sorttmp.sort()
        squashed_pages = [p[1] for p in sorttmp]
        self.squashed_pages = squashed_pages

        print "%3d" % self.subcode_count, "%3d" % len(squashed_pages), "%3d" % len(unique_pages)

    def guess_subcodes(self):
        subpages = [x.ds for x in self.pages if x.ds < 0x100]
        us = set(subpages)
        sc = [(s,subpages.count(s)) for s in us]
        sc.sort()

        if len(sc) < 2:
          return sc
        else:
          if sc[0][0] == 0 and sc[0][1] > (len(subpages)*0.8):
            good = [0]
          else:
            good = []
            bad = []
            for n in range(len(sc)):
                if sc[n][0] == n+1:
                    good.append(sc[n][0])
                else:
                    bad.append(sc[n][0])

        return good

    def hamming(self):
        unique_pages = []
        unique_pages.append([self.pages[0]])

        for p in self.pages[1:]:
            matched = False
            for op in unique_pages:
                if p.hamming(op[0]):
                    op.append(p)
                    matched = True
                    break
            if not matched:
                unique_pages.append([p])

        sorttmp = [(len(u),u) for u in unique_pages]
        sorttmp.sort(reverse=True)
        unique_pages = [x[1] for x in sorttmp]

        #if len(unique_pages) > self.subcode_count:
        #    unique_pages = unique_pages[:self.subcode_count]
        self.print_this = (len(unique_pages) != self.subcode_count)

        return unique_pages

    def squash(self, pages):
        return bitwise_mode([x.array for x in pages])

    def to_str(self):
        return "".join([p.to_str() for p in self.squashed_pages])

    def to_html(self):
        header = """<html><head><meta http-equiv="Content-Type" content="text/html; charset=utf-8">
<title>Page %d%02x</title><link rel="stylesheet" type="text/css" href="teletext.css" /></head>
<body><pre>""" % (self.m, self.p)
        body = "".join([p.to_html("#%d" % n) for n,p in enumerate(self.squashed_pages)])
        footer = "</body>"

        return header+body+footer

def main_work_subdirs(gl):
    for root, dirs, files in os.walk(gl['pwd']):
        dirs.sort()
        if root == gl['pwd']:
            for d2i in dirs:
                print(d2i)

if __name__=='__main__':
    indir = sys.argv[1]
    outdir = sys.argv[2]

    outpath = os.path.join('.', outdir)
    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    for root, dirs, files in os.walk(indir):
        dirs.sort()
        files.sort()
        for f in files:
            s = Squasher(os.path.join('.', root, f))
            m = s.m
            if m == 0:
                m = 8
            outfile = "%d%02x.html" % (m, s.p)
            of = file(os.path.join(outpath, outfile), 'wb')
            of.write(s.to_html())

