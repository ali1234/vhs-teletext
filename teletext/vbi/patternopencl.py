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
    __kernel void correlate(global float* input, global float* patterns, global float* result,
                            int range_low, int range_high)
    {
      int x = get_global_id(0);
      int y = get_global_id(1);
      int iidx = x * 8;
      int ridx = (x * get_global_size(1)) + y;
      int pidx = y * 24;

      result[ridx] = 0;

      for (int i=range_low;i<range_high;i++) {
        float d = input[iidx+i] - patterns[pidx+i];
        result[ridx] += (d*d);
      }
    }

    __kernel void minerr(global float* input, global int* indexes, int npatterns)
    {
      int ch = get_global_id(0);
      int start = npatterns * ch;

      int bestidx = 0;
      float bestval = input[start];

      for (int i=1;i<npatterns;i++) {
        float val = input[start+i];
        if (val < bestval) {
          bestval = val;
          bestidx = i;
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
        self.kernel_min = self.prg.minerr

        # patterns is already float32 (see Pattern __init__)
        self.patterns_gpu = cl.Buffer(openclctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.patterns)

        # the input of the correlate -
        # size copying CUDA code, something like 40chars, 8 bits each, float??
        self.input_match = cl.Buffer(openclctx, mf.READ_WRITE, 4*((40*8)+16))

        # output of the correlate
        self.result_match = cl.Buffer(openclctx, mf.READ_WRITE, 4*40*self.n)

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

        # Run min pass
        # Output is a set of indexes giving the best pattern per character
        self.kernel_min.set_args(self.result_match, self.result_minidx,
                                 np.int32(self.n))

        e_min = cl.enqueue_nd_range_kernel(self.queue, self.kernel_min,
                                           (l,), None,
                                           wait_for = (e_corr,))

        # and get the index values back from OpenCL
        cl.enqueue_copy(self.queue, self.result_minidx_np, self.result_minidx, wait_for = (e_min,))

        return self.bytes[self.result_minidx_np[:l],0]

