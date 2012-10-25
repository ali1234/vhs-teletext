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

import config

import time
import printer
from vbi import Vbi

import guessview

def process_file(filename):
  ans = []
  try:
    f = file(filename).read()
    for line in range(32):
        offset = line*2048
        vbiraw = np.array(np.fromstring(f[offset:offset+2048], dtype=np.uint8), dtype=np.float32)
        for tl in np.arange(1.05,1.2,0.05):
         for th in np.arange(2.1,2.4,0.05):
          for g in np.arange(1.1,5.2,1.0):
           for go in np.arange(1.1,5.2,1.0):
            #thresh_low=1.15 thresh_high=2.10 gauss_sd=1.10 gauss_sd_offset=1.10
            v = Vbi(vbiraw, thresh_low=tl, thresh_high=th, gauss_sd=g, gauss_sd_offset=go)
            tmp = v.find_offset_and_scale()
            if tmp:
                packet = v.deconvolve()
                guessview.draw(vbiraw, v.g.convolved*256)
                print printer.do_print(packet), "thresh_low=%1.2f thresh_high=%1.2f gauss_sd=%1.2f gauss_sd_offset=%1.2f" % (tl, th, g, go), line
            else:
                packet = None
                print "no teletext in line", line        
        
  except IOError:
    pass

  return (filename, ans)  

if __name__ == '__main__':

    guessview.main()

    #do_file('data/0022/vbi/00171104.vbi')
    process_file(sys.argv[1])
    #do_file('data/0022/vbi/00171106.vbi')
    #do_file('data/0022/vbi/00171107.vbi')
    #do_file('data/0022/vbi/00171108.vbi')







