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

        raise Exception("Not yet implemented")

    def match(self, inp):
        raise Exception("Not yet implemented")

