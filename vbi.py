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

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d as gauss
from scipy.optimize import fminbound

import pylab

from util import paritybytes, setbyte, normalise, hammbytes, allbytes, mrag, notzero

from finders import *

import time

class Vbi(object):
    '''This class represents a line of raw vbi data and all our attempts to
        decode it.'''

    possible_bytes = [hammbytes]*2 + [paritybytes]*40

    def __init__(self, vbi, finders):

        # data arrays

        # vbi is the raw line as an array of 2048 floats
        self.vbi = vbi

        # finders find packet 0 based on pattern matching
        self.finders = finders

        # black level of the signal
        self.black = np.mean(self.vbi[:80])

        # scale factor of the vbi, calculated later
        self.scale = 0

        # note: guess and mask are 376 "bits" long because an extra "byte" is 
        # needed at the beginning and the end for interpolation to work 
        # correctly. setbyte and bitstobytes take this into account.

        # guess is the best guess at what the vbi contains
        self.guess = np.zeros(47*8, dtype=np.float32)
        setbyte(self.guess, -1, 0x0)
        setbyte(self.guess, 0, 0x55)
        setbyte(self.guess, 1, 0x55)
        setbyte(self.guess, 2, 0x27)

        # mask is used to mark regions of interest in the interpolated guess
        self.mask = np.zeros(47*8, dtype=np.float32)
        setbyte(self.mask, -1, 0xff)
        setbyte(self.mask, 0, 0xff)
        setbyte(self.mask, 1, 0xff)
        setbyte(self.mask, 2, 0xff)

        # vbi packet bytewise
        self._mask0 = np.zeros(42, dtype=np.uint8)
        self._mask1 = np.zeros(42, dtype=np.uint8)

        # parameters for interpolation
        self._bitwidth = 5.112
        self._offset = 0

        # params for convolve
        self._gauss_sd = 5.5 # TWEAK: amount of gaussian blurring to apply
                             # this cuts out hi freq. noise in the samples
                             # but also reduces the amount of data

        # interpolation objects
        self._interp_x = np.zeros(376, dtype=np.float32)
        self._guess_x = np.zeros(2048, dtype=np.float32)
        self.set_bitwidth(self._bitwidth)
        self.set_offset(self._offset)

        self.guess_scaler = interp1d(self._interp_x, self.guess, 
                                     kind='linear', copy=False, 
                                     bounds_error=False, fill_value=0)
        # note that mask_scaler uses the same offsets as guess_scaler
        # and has fill value = 1 because we always are interested in the
        # samples outside the signal as they should never change
        self.mask_scaler = interp1d(self._interp_x, self.mask, 
                                     kind='linear', copy=False, 
                                     bounds_error=False, fill_value=1)

        self.count = 0
        self.it = 0

    def set_bitwidth(self, bitwidth):
        self._bitwidth = bitwidth
        self._interp_x[:] = (np.arange(0,47*8,1.0) * bitwidth) - bitwidth*8

    def set_offset(self, offset):
        self._offset = offset
        self._guess_x[:] = np.arange(0,2048,1.0) - offset

    def find_offset_and_scale(self):
        '''Tries to find the offset of the vbi data in the raw samples.'''

        # Only consider the general area of the CRI
        target = gauss(self.vbi[64:256], self._gauss_sd)

        def _inner(offset):
            self.set_offset(offset)
            guess_scaled = gauss(self.guess_scaler(self._guess_x[64:256]), self._gauss_sd)
            #mask_scaled = gauss(self.mask_scaler(self._guess_x[64:256]), self._gauss_sd)
            mask_scaled = self.mask_scaler(self._guess_x[64:256]) # not blurring this seems to be better

            a = guess_scaled*mask_scaled
            b = np.clip(target*mask_scaled, self.black, 256)

            self.scale = a.std()/b.std()
            b -= self.black
            b *= self.scale
            a = np.clip(a, 0, 256*self.scale)

            return np.sum(np.square(a-b))

        offset = fminbound(_inner, 96.0, 110.0)

        # call it also to set self.offset and self.scale
        return (_inner(offset) < 5.0)

    def make_guess_mask(self,o=0):
        a = []
        for i in range(42*8):
            a.append([])
        b = 4*8

        gx = self._guess_x + (self._bitwidth*0.5) + o

        for i in range(2048):
            while b < 368 and gx[i] > self._interp_x[b+1]:
                b += 1
            if self._interp_x[b] < gx[i] and b < 368:
                a[b-4*8].append(self.vbi[i])

        mins = np.array([min(x) for x in a])
        maxs = np.array([max(x) for x in a])
        avgs = np.array([np.array(x).mean() for x in a])

        for i in range(42):
            mini = mins[i*8:(i+1)*8]
            maxi = maxs[i*8:(i+1)*8]
            avgi = avgs[i*8:(i+1)*8]
            self._mask0[i] = 0xff
            for j in range(8):
                if mini[j] < self.black+15.0:
                    self._mask0[i] &= ~(1<<j)
                if maxi[j] > self.black*2.35:
                    self._mask1[i] |= (1<<j)

        tmp = self._mask1 & self._mask0
        self._mask0 |= self._mask1
        self._mask1 = tmp

    def make_possible_bytes(self, possible_bytes):
        def masked(b, n):
            m0 = self._mask0[n]
            m1 = self._mask1[n]
            m = [x for x in b if (x&m0)==x==(x|m1)]
            if m != []:
                return m
            else:
                mm0 = [x for x in b if (x&m0)==x]
                mm1 = [x for x in b if (x|m1)==x]
                if len(mm0) < len(mm1):
                    return mm0 or mm1 or b
                else:
                    return mm1 or mm0 or b

        self.possible_bytes = [masked(b,n) for n,b in enumerate(possible_bytes)]

    def _deconvolve_make_diff(self):
        self.count += 1
        a = gauss(self.guess_scaler(self._guess_x), self._gauss_sd)
        a = normalise(a)
        return np.sum(np.square(a-self.target))

    def _deconvolve_pass(self, first=0, last=42):
        for n in range(first, last):
            nb = self.possible_bytes[n]
            
            if len(nb) == 1:
                setbyte(self.guess, n+3, nb[0])
                self._bytes[n] = nb[0]
            else:
                ans = []
                for b1 in nb:
                    setbyte(self.guess, n+3, b1)
                    ans.append((self._deconvolve_make_diff(),b1))

                best = min(ans)
                setbyte(self.guess, n+3, best[1])
                self._bytes[n] = best[1]

    def _deconvolve(self):
        for it in range(10): # TWEAK: maximum number of iterations.
            self.it += 1
            self._deconvolve_pass()
            # if this iteration didn't produce a change in the answer
            # then the next one won't either, so stop.
            if (self._bytes == self._oldbytes).all():
                break
            self._oldbytes[:] = self._bytes

    def _nzdeconvolve(self):
        for it in range(10): # TWEAK: maximum number of iterations.
            self.it += 1
            ans=[]
            for nb in notzero:
                setbyte(self.guess, 3, nb[0])
                setbyte(self.guess, 4, nb[1])
                ans.append((self._deconvolve_make_diff(),nb))
            best = min(ans)
            setbyte(self.guess, 3, best[1][0])
            setbyte(self.guess, 4, best[1][1])
            self._bytes[0] = best[1][0]
            self._bytes[1] = best[1][0]

            self._deconvolve_pass(first=2)
            # if this iteration didn't produce a change in the answer
            # then the next one won't either, so stop.
            if (self._bytes == self._oldbytes).all():
                break
            self._oldbytes[:] = self._bytes

    def deconvolve(self):
        self.make_guess_mask()
        self.make_possible_bytes(Vbi.possible_bytes)

        target = gauss(self.vbi, self._gauss_sd)
        self.target = normalise(target)

        self._bytes = np.zeros(42, dtype=np.uint8) | 0x55
        self._oldbytes = np.zeros(42, dtype=np.uint8)

        self._deconvolve()

        packet = "".join([chr(x) for x in self._bytes])

        for F in self.finders:
            if F.find(packet):
                sys.stderr.write("matched by finder "+F.name+"\n");
                sys.stderr.flush()               
                self.make_possible_bytes(F.possible_bytes)
                self._deconvolve()
                packet = "".join([chr(x) for x in self._bytes])
                F.find(packet)
                packet = F.fixup()
                return packet

        # if the packet did not match any of the finders then it isn't 
        # a packet 0 (or 30). if the packet still claims to be a packet 0 it 
        # will mess up the page splitter. so redo the deconvolution but with 
        # packet 0 (and 30) header removed from possible bytes.

        # note: this doesn't work. i am not sure why. a packet in 63322
        # does not match the finders but still passes through this next check
        # with r=0. which should be impossible.
        ((m,r),e) = mrag(self._bytes[:2])
        if r == 0 or r == 30:
            sys.stderr.write("packet falsely claimed to be packet %d\n" % r);
            sys.stderr.flush()
            self._nzdeconvolve()
            packet = "".join([chr(x) for x in self._bytes])
        # if it's a link packet, it is completely hammed
        elif r == 27:
            self.make_possible_bytes([hammbytes]*42)
            self._deconvolve()
            packet = "".join([chr(x) for x in self._bytes])

        return packet
            
        


def process_file(filename):
  ans = []
  try:
    f = file(filename).read()
    for line in range(32):
        offset = line*2048
        vbiraw = np.array(np.fromstring(f[offset:offset+2048], dtype=np.uint8), dtype=np.float32)
        v = Vbi(vbiraw, [BBC1, TeletextLtd, FourTel, BBC1_BSD])
        c2 = c1 = c0 = time.time()
        tmp = v.find_offset_and_scale()
        c1 = time.time()
        if tmp:
            packet = v.deconvolve()
        else:
            packet = None
        c2 = time.time()
        timing = (c1-c0, c2-c1, v.it, v.count)

        ans.append((packet, timing))
        
  except IOError:
    pass

  return (filename, ans)  

def list_files(datapath):
    for frame in range(1000, 1001, 1):
        frame = "%08d" % frame
        yield datapath+'/'+frame+'.vbi'

#../0008//00063322.vbi
if __name__ == '__main__':

    def do_file(filename):
        sys.stderr.write(filename+'\n')
        ans = process_file(filename)[1]
        for p,t in ans:
            if p:
                sys.stdout.write(p)
        sys.stdout.flush()

    datapath = sys.argv[1]
    map(do_file, list_files(datapath))





