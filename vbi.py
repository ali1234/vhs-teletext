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

# This is the main data analyser.

import sys
import os
import numpy as np
from scipy.ndimage import gaussian_filter1d as gauss
from scipy.optimize import fminbound

import pylab

from util import paritybytes, setbyte, normalise, hammbytes, allbytes, mrag, notzero

import util

from guess import Guess

import time

import finders
import math

np.seterr(invalid= 'raise')

class Vbi(object):
    '''This class represents a line of raw vbi data and all our attempts to
        decode it.'''

    possible_bytes = [hammbytes]*2 + [paritybytes]*40

    def __init__(self, vbi, bitwidth=5.112, gauss_sd=4.0, gauss_sd_offset=4.0,
                 offset_low = 75.0, offset_high = 119.0,
                 thresh_low = 1.1, thresh_high = 2.36,
                 allow_unmatched = False):

        # data arrays

        # vbi is the raw line as an array of 2048 floats
        self.vbi = vbi

        # blurring amounts
        self.gauss_sd = gauss_sd
        self.gauss_sd_offset = gauss_sd_offset

        # Offset range to check for signal drift, in samples.
        # The algorithm will check with sub sample accuracy.
        # The offset finder is very fast (it uses bisection)
        # so this range can be relatively large. But not too
        # large or you get false positives.
        self.offset_low = offset_low
        self.offset_high = offset_high

        # black level of the signal
        self.black = np.mean(self.vbi[:80])

        # Threshold multipliers. The black level of the signal 
        # is derived from the mean of the area before the VBI 
        # begins. This is multiplied by the following factors 
        # to give the low/high thresholds. Anything outside
        # this range is assumed to be a 0 or a 1. Tweaking these
        # can improve results, but often at a speed cost.
        self.thresh_low = self.black*thresh_low
        self.thresh_high = self.black*thresh_high

        # Allow vbi.py to emitt packet 0's that don't match
        # any finder? Set to false when you have finders
        # for all headers in the data.
        self.allow_unmatched = allow_unmatched

        # vbi packet bytewise
        self._mask0 = np.zeros(42, dtype=np.uint8)
        self._mask1 = np.zeros(42, dtype=np.uint8)

        self.g = Guess(bitwidth=bitwidth)


    def find_offset_and_scale(self):
        '''Tries to find the offset of the vbi data in the raw samples.'''

        # Split into chunks and ensure there is something "interesting" in each
        target = gauss(self.vbi, self.gauss_sd_offset)
        d = [np.std(target[x:x+128]) < 10.0 for x in range(64, 2048, 128)]
        if any(d):
            return False

        low = 64
        high = 256
        target = gauss(self.vbi[low:high], self.gauss_sd_offset)

        def _inner(offset):
            self.g.set_offset(offset)

            self.g.update_cri(low, high)
            guess_scaled = self.g.convolved[low:high]
            mask_scaled = self.g.mask[low:high]

            a = guess_scaled*mask_scaled
            b = np.clip(target*mask_scaled, self.black, 256)

            scale = a.std()/b.std()
            b -= self.black
            b *= scale
            a = np.clip(a, 0, 256*scale)

            return np.sum(np.square(b-a))

        offset = fminbound(_inner, self.offset_low, self.offset_high)

        # call it also to set self.offset and self.scale
        return (_inner(offset) < 10)

    def make_guess_mask(self):
        a = []

        for i in range(42*8):
            (low, high) = self.g.get_bit_pos(i)
            a.append(self.vbi[low:high])

        mins = np.array([min(x) for x in a])
        maxs = np.array([max(x) for x in a])
        avgs = np.array([np.array(x).mean() for x in a])

        for i in range(42):
            mini = mins[i*8:(i+1)*8]
            maxi = maxs[i*8:(i+1)*8]
            avgi = avgs[i*8:(i+1)*8]
            self._mask0[i] = 0xff
            for j in range(8):
                if mini[j] < self.thresh_low:
                    self._mask0[i] &= ~(1<<j)
                if maxi[j] > self.thresh_high:
                    self._mask1[i] |= (1<<j)

        tmp = self._mask1 & self._mask0
        self._mask0 |= self._mask1
        self._mask1 = tmp

    def make_possible_bytes(self, possible_bytes):
        def masked(b, n):
            m0 = util.m0s[self._mask0[n]]
            m1 = util.m1s[self._mask1[n]]
            m = m0 & m1 & b
            if m:
                return m
            else:
                mm0 = m0 & b
                mm1 = m1 & b
                if len(mm0) < len(mm1):
                    return mm0 or mm1 or b
                else:
                    return mm1 or mm0 or b

        self.possible_bytes = [masked(b,n) for n,b in enumerate(possible_bytes)]

    def _deconvolve_make_diff(self, (low, high)):
        a = normalise(self.g.convolved)
        diff_sq = np.square(a - self.target)
        return np.sum(diff_sq)
        # an interesting trick I discovered. 
        # bias the result towards the curent area of interest
        return np.sum(diff_sq[:low]) + 2.6*np.sum(diff_sq[low:high]) + np.sum(diff_sq[high:])

    def _deconvolve_pass(self, first=0, last=42):
        for n in range(first, last):
            nb = self.possible_bytes[n]

            changed = self.g.set_update_range(n+4, 1)

            if len(nb) == 100000:
                self.g.set_byte(n, nb[0])
            else:
                ans = []
                for b1 in nb:
                    self.g.set_byte(n, b1)
                    ans.append((self._deconvolve_make_diff(changed),b1))

                best = min(ans)
                self.g.set_byte(n, best[1])
        self.g.update_all()

    def _deconvolve(self):
        for it in range(10):
            self._deconvolve_pass()
            # if this iteration didn't produce a change in the answer
            # then the next one won't either, so stop.
            if (self.g.bytes == self._oldbytes).all():
                #print it
                break
            self._oldbytes[:] = self.g.bytes

    def _nzdeconvolve(self):
        for it in range(10):
            ans=[]
            changed = self.g.set_update_range(4, 2)
            for nb in notzero:
                self.g.set_two_bytes(0, nb[0], nb[1])
                ans.append((self._deconvolve_make_diff(changed),nb))
            best = min(ans)
            self.g.set_two_bytes(0, best[1][0], best[1][1])

            self._deconvolve_pass(first=2)
            # if this iteration didn't produce a change in the answer
            # then the next one won't either, so stop.
            if (self.g.bytes == self._oldbytes).all():
                #print it
                break
            self._oldbytes[:] = self.g.bytes

    def deconvolve(self):
        target = gauss(self.vbi, self.gauss_sd)
        self.target = normalise(target)

        self.make_guess_mask()
        self.make_possible_bytes(Vbi.possible_bytes)

        self._oldbytes = np.zeros(42, dtype=np.uint8)

        self._deconvolve()

        packet = "".join([chr(x) for x in self.g.bytes])

        F = finders.test(finders.all_headers, packet)
        if F:
                sys.stderr.write("matched by finder "+F.name+"\n");
                sys.stderr.flush()               
                self.make_possible_bytes(F.possible_bytes)
                self._deconvolve()
                F.find(self.g.bytes)
                packet = F.fixup()
                return packet

        # if the packet did not match any of the finders then it isn't 
        # a packet 0 (or 30). if the packet still claims to be a packet 0 it 
        # will mess up the page splitter. so redo the deconvolution but with 
        # packet 0 (and 30) header removed from possible bytes.

        # note: this doesn't work. i am not sure why. a packet in 63322
        # does not match the finders but still passes through this next check
        # with r=0. which should be impossible.
        ((m,r),e) = mrag(self.g.bytes[:2])
        if r == 0:
            sys.stderr.write("packet falsely claimed to be packet %d\n" % r);
            sys.stderr.flush()
            if not self.allow_unmatched:
                self._nzdeconvolve()
            packet = "".join([chr(x) for x in self.g.bytes])
        # if it's a link packet, it is completely hammed
        elif r == 27:
            self.make_possible_bytes([hammbytes]*42)
            self._deconvolve()
            packet = "".join([chr(x) for x in self.g.bytes])

        return packet
            
        


def process_file((inname, outname)):
  print inname, outname
  try:
    data = np.fromstring(file(inname).read(), dtype=np.uint8)
    outfile = file(outname, 'wb')
    for line in range(32):
        offset = line*2048
        vbiraw = data[offset:offset+2048]
        v = Vbi(vbiraw)
        tmp = v.find_offset_and_scale()
        if tmp:
            outfile.write(v.deconvolve())
        else:
            outfile.write("\xff"*42)
    outfile.close()
  except IOError:
    pass


def test_file(outfile):
    return os.path.isfile(outfile) and (os.path.getsize(outfile) == (42 * 32))

def list_files(inputpath, outputpath, first, count):
    for frame in range(first, first+count, 1):
        frame = "%08d" % frame
        if test_file(outputpath + '/' + frame + '.t42'):
            #print "skipping %s\n" % (outputpath + '/' + frame + '.t42')
            pass
        else:
            yield (inputpath + '/' + frame + '.vbi', 
                  outputpath + '/' + frame + '.t42')


if __name__ == '__main__':
    import multiprocessing
    from multiprocessing.pool import IMapIterator, Pool
    import cProfile
    import pstats
    def wrapper(func):
      def wrap(self, timeout=None):
        # Note: the timeout of 1 googol seconds introduces a rather subtle
        # bug for Python scripts intended to run many times the age of the universe.
        return func(self, timeout=timeout if timeout is not None else 1e100)
      return wrap
    IMapIterator.next = wrapper(IMapIterator.next)


    try:
        path = sys.argv[1]
    except:
        print "Usage:", sys.argv[0], "<path> [<first> <count>]\n"
        print "  path: directory with VBI files to process"
        print "  first: first file to process"
        print "  count: maximum number of files to process\n"
        exit(-1)

    try:
        first = int(sys.argv[2], 10)
        count = int(sys.argv[3], 10)
    except:
        first = 0
        count = 10000000

    if 1:
        p = Pool(multiprocessing.cpu_count())
        it = p.imap(process_file, list_files(path+'/vbi/', path+'/t42/', first, count), chunksize=1)
        for i in it:
            pass

    else: # single thread mode for debugging
        def doit():
            map(process_file, list_files(path+'/vbi/', path+'/t42/', first, count))
        cProfile.run('doit()', 'myprofile')
        p = pstats.Stats('myprofile')
        p.sort_stats('cumulative').print_stats(50)



