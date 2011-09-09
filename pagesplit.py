#!/usr/bin/env python

import sys, os
import numpy as np

from util import mrag, page
from finders import BBC1
from printer import do_print

def splitlist(l, s):
    t = [0]+[n for n,i in enumerate(l) if i[0] == s]+[len(l)]
    return [l[t[i]:t[i+1]] for i in range(len(t)-1)]

class PageWriter(object):
    def __init__(self, outdir):
        self.outdir = outdir
        self.count = 0

    def write_page(self, ps):
        if ps[0].me or ps[0].pe:
            pass
        else:
            m = str(ps[0].m)
            p = '%02x' % ps[0].p
            path = os.path.join('.', self.outdir, m)
            if not os.path.isdir(path):
                os.makedirs(path)
            f = os.path.join(path, p)
            of = file(f, 'ab')
            for p in ps:
                of.write(p.tt)
            of.close()
            if self.count % 10 == 0:
                print "%08d" % self.count, f, '- ',
                do_print(np.fromstring(ps[0].tt, dtype=np.uint8))
            self.count += 1

    def write_page_2(self, ps):
        if ps[0].me or ps[0].pe:
            pass
        else:
            m = str(ps[0].m)
            p = '%02x' % ps[0].p
            path = os.path.join('.', self.outdir, m, p)
            if not os.path.isdir(path):
                os.makedirs(path)
            n = 0
            while(os.path.isfile(os.path.join(path, "%08d" % n))):
                n += 1
            f = os.path.join(path, "%08d" % n)
            of = file(f, 'ab')
            for p in ps:
                of.write(p.tt)
            of.close()
            print f, '- ',
            do_print(ps[0].tt)

class PacketHolder(object):
    def __init__(self, tt):
        (self.m,self.r),e = mrag(np.fromstring(tt[:2], dtype=np.uint8))
        if BBC1.find(tt):
            self.r = 0
            BBC1.check_page_info()
            self.me = BBC1.me
            self.pe = BBC1.pe
            self.p = BBC1.p
                
        elif self.r == 0:
            self.r = -1
            self.m = -1
        self.tt = tt
        self.good = None

class NullHandler(object):
    def add_packet(self, p):
        # immediately reject it
        p.good = False

class MagHandler(object):
    packet_sequence = [0, 27, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
                       10, 11, 12, 13, 14, 15, 16, 17, 
                       18, 19, 20, 21, 22, 23, 24]
    psl = len(packet_sequence)

    def __init__(self, m, pagewriter):
        self.m = m
        self.packets = []
        self.seen_header = False
        self.pagewriter = pagewriter

    def good_page(self):
        self.pagewriter.write_page(self.packets)
        self.packets = []

    def bad_page(self):
        for p in self.packets:
            p.good = False
        self.packets = []

    def split_page_inner(self):
        pass

    # unused, work in progress
    def try_split_page(self):
        if self.packets[0].r != 0:
            # start of page missing
            if self.seen_header: # this should not happen
                sys.stderr.write("algorithm is broken lol\n")
                sys.stderr.flush()
                self.bad_page()
            elif len(self.packets) < MagHandler.psl:
                self.good_page()
            else: # this should not happen
                sys.stderr.write("algorithm is really broken lol\n")
                sys.stderr.flush()
                self.bad_page()
        else:
            if len(self.packets) < MagHandler.psl:
                return
            else:
                self.split_page_inner()

    def check_page(self):
        if len(self.packets) == MagHandler.psl:
            rows = [p.r for p in self.packets]
            c = [a1 == b1 for a1,b1 in zip(rows,MagHandler.packet_sequence)].count(True)
            if c >= (MagHandler.psl - 0): # allow up to n corrupted row numbers
                self.good_page()
                return
                
        self.bad_page()

    def add_packet(self, p):
        if p.r == 0:
            self.check_page()
            self.seen_header = True
        self.packets.append(p)



if __name__=='__main__':

    w = PageWriter(sys.argv[1])

    mags = [NullHandler(), MagHandler(1,w), NullHandler(), MagHandler(3,w),
            MagHandler(4,w), MagHandler(5,w), MagHandler(6,w), NullHandler()]

    packet_list = []

    while(True):
        tt = sys.stdin.read(42)
        if len(tt) < 42:
            exit(0)

        p = PacketHolder(tt)
        if p.r in MagHandler.packet_sequence:
            mags[p.m].add_packet(p)
        else:
           p.good = False
        #packet_list.append(p)
        #while packet_list[0].good is not None:
        #    pp = packet_list.pop(0)
        #    if not pp.good:
        #        sys.stdout.write(pp.tt)
        #sys.stdout.flush()
