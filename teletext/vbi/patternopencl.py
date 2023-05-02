# * Copyright 2023 Dr. David Alan Gilbert <dave@treblig.org>
# *   based on Alistair's patterncuda.py
# *
# * License: This program is free software; you can redistribute it and/or
# * modify it under the terms of the GNU General Public License as published
# * by the Free Software Foundation; either version 3 of the License, or (at
# * your option) any later version. This program is distributed in the hope
# * that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# * GNU General Public License for more details.

import numpy as np
import pyopencl as cl

from .pattern import Pattern

openclctx = cl.create_some_context(interactive=False)

class PatternOpenCL(Pattern):
    prg = cl.Program(openclctx, """
    __kernel void correlate(global float* restrict input, global float* restrict patterns, global float* restrict result,
                            int range_low, int range_high)
    {
      int x = get_global_id(0);
      int y = get_global_id(1);
      int iidx = x * 8;
      int ridx = (x * get_global_size(1)) + y;
      int pidx = y * 24;
      float d;

      result[ridx] = 0;

      for (int i=range_low;i<range_high;i++) {
        d = input[iidx+i] - patterns[pidx+i];
        result[ridx] += (d*d);
      }
    }

    // Each workitem takes one character x npatterns/minpar values
    // and finds the minimum, writing one value and index into the
    // temporaries
    // The temporaries are 40 characters wide
    // Done as a 2D parallel, X is character,
    // Y is npatterns/minpar chunk of correlate results
    __kernel void minerr1(global float* restrict input,
                         global float* restrict tmp_val, global int* restrict tmp_idx,
                         int npatterns, int minpar)
    {
      int ch = get_global_id(0);
      int width = get_global_size(0);
      int patblock = get_global_id(1);
      int patstep = npatterns / minpar;
      int patstart = patblock * patstep;
      int patend = patstart + patstep;

      int inindex = patstart + npatterns*ch;
      int bestidx = patstart;
      float bestval = input[inindex];
      float val;

      for (int p=patstart; p<patend; p++, inindex+=1) {
        val = input[inindex];
        if (val < bestval) {
          bestval = val;
          bestidx = p;
        }
      }

      int tidx = patblock*width + ch;
      tmp_idx[tidx] = bestidx;
      tmp_val[tidx] = bestval;
    }

    // Each workitem takes one character x minpar values and finds the
    // minimum of the temporary minima and writes the index
    // Done as a 1D parallel over the characters
    __kernel void minerr2(global float* restrict tmp_val, global int* restrict tmp_idx,
                          global int* restrict indexes,
                          int minpar)
    {
      int ch = get_global_id(0);
      int width = get_global_size(0);

      int iidx = ch;
      int bestidx = tmp_idx[iidx];
      float bestval = tmp_val[iidx];
      float val;

      iidx+=width;
      for (int i=1;i<minpar;i++,iidx+=width) {
        val = tmp_val[iidx];
        if (val < bestval) {
          bestidx = tmp_idx[iidx];
          bestval = val;
        }
      }
      indexes[ch] = bestidx;
    }
    """).build()

    def __init__(self, filename):
        Pattern.__init__(self, filename)

        self.queue = cl.CommandQueue(openclctx)

        mf = cl.mem_flags

        self.kernel_correlate = self.prg.correlate
        self.kernel_min1 = self.prg.minerr1
        self.kernel_min2 = self.prg.minerr2

        # patterns is already float32 (see Pattern __init__)
        self.patterns_gpu = cl.Buffer(openclctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.patterns)

        # the input of the correlate -
        # size copying CUDA code, something like 40chars, 8 bits each, float??
        self.input_match = cl.Buffer(openclctx, mf.READ_WRITE, 4*((40*8)+16))

        # output of the correlate
        self.result_match = cl.Buffer(openclctx, mf.HOST_NO_ACCESS, 4*40*self.n)

        # How much to split the min search by vertically
        self.minpar = 256

        # Temporaries used during parallel min (value and index)
        self.mintmp_val = cl.Buffer(openclctx, mf.HOST_NO_ACCESS, 4*40*self.minpar)
        self.mintmp_idx = cl.Buffer(openclctx, mf.HOST_NO_ACCESS, 4*40*self.minpar)

        # output of the min pass - an integer index to which pattern was best
        # for each character
        self.result_minidx = cl.Buffer(openclctx, mf.WRITE_ONLY, 4*40)
        # and a copy of that for np
        self.result_minidx_np = np.zeros(40, dtype=np.uint32)

    def match(self, inp):
        l = (len(inp)//8)-2
        x = l & -l # highest power of two which divides l, up to 8
        y = min(1024//x, self.n)

        # copy data in
        e_copy = cl.enqueue_copy(self.queue, self.input_match, inp.astype(np.float32), is_blocking = False)
        # call corellate
        # Output is one row per character, with one value per pattern
        self.kernel_correlate.set_args(self.input_match, self.patterns_gpu, self.result_match,
                                       np.int32(self.start), np.int32(self.end))

        e_corr = cl.enqueue_nd_range_kernel(self.queue, self.kernel_correlate,
                                            (l, self.n), None,
                                            wait_for = (e_copy,))

        # Run min pass 1
        # squashes the set of patterns down into minpar minima
        assert (self.n % self.minpar) == 0

        self.kernel_min1.set_args(self.result_match,
                                  self.mintmp_val, self.mintmp_idx,
                                  np.int32(self.n), np.int32(self.minpar))

        e_min1 = cl.enqueue_nd_range_kernel(self.queue, self.kernel_min1,
                                            (l,self.minpar), None,
                                            wait_for = (e_corr,))

        # Run min pass 2
        # squashes the temporaries down to a final minimum index for each char
        self.kernel_min2.set_args(self.mintmp_val, self.mintmp_idx,
                                  self.result_minidx,
                                  np.int32(self.minpar))

        e_min2 = cl.enqueue_nd_range_kernel(self.queue, self.kernel_min2,
                                            (l,), None,
                                            wait_for = (e_min1,))


        # and get the index values back from OpenCL
        e_out = cl.enqueue_copy(self.queue, self.result_minidx_np, self.result_minidx, wait_for = (e_min2,))
        e_out.wait()
        return self.bytes[self.result_minidx_np[:l],0]

