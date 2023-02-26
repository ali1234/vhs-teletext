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
    """).build()

    def __init__(self, filename):
        Pattern.__init__(self, filename)

        self.queue = cl.CommandQueue(openclctx)

        mf = cl.mem_flags
        # patterns is already float32 (see Pattern __init__)
        self.patterns_gpu = cl.Buffer(openclctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.patterns)

        # the input of the correlate -
        # size copying CUDA code, something like 40chars, 8 bits each, float??
        self.input_match = cl.Buffer(openclctx, mf.READ_WRITE, 4*((40*8)+16))

        # output of the correlate
        self.result_match = cl.Buffer(openclctx, mf.READ_WRITE, 4*40*self.n)

    def match(self, inp):
        l = (len(inp)//8)-2
        x = l & -l # highest power of two which divides l, up to 8
        y = min(1024//x, self.n)

        # copy data in
        cl.enqueue_copy(self.queue, self.input_match, inp.astype(np.float32))
        # call corellate
        self.prg.correlate(self.queue, (l, self.n), None,
                           self.input_match, self.patterns_gpu, self.result_match,
                           np.int32(self.start), np.int32(self.end))
        # TODO do a min pass to find smallest (start with numpy?)
        #raise Exception("Not yet implemented (match)")

