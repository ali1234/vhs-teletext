# * Copyright 2016 Alistair Buxton <a.j.buxton@gmail.com>
# *
# * License: This program is free software; you can redistribute it and/or
# * modify it under the terms of the GNU General Public License as published
# * by the Free Software Foundation; either version 3 of the License, or (at
# * your option) any later version. This program is distributed in the hope
# * that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# * GNU General Public License for more details.

import numpy as np

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule

import skcuda
from skcuda.misc import argmin
skcuda.misc._global_cublas_allocator = cuda.mem_alloc

from .pattern import Pattern

class PatternCUDA(Pattern):

    mod = SourceModule("""
        __global__ void correlate(float *input, float *patterns, float *result, int range_low, int range_high)
        {
            int x = (threadIdx.x + (blockDim.x*blockIdx.x));
            int y = (threadIdx.y + (blockDim.y*blockIdx.y));
            int iidx = x * 8;
            int ridx = (x * blockDim.y * gridDim.y) + y;
            int pidx = y * 24;
        
            float d;
            result[ridx] = 0;
        
            for (int i=range_low;i<range_high;i++) {
                d = input[iidx+i] - patterns[pidx+i];
                result[ridx] += (d*d);
            }
        }
    """)

    correlate = mod.get_function("correlate")

    def __init__(self, filename):
        Pattern.__init__(self, filename)

        if self.n&1023 != 0:
            raise ValueError('Number of patterns must be a multiple of 1024.')

        self.patterns_gpu = cuda.mem_alloc(self.patterns.nbytes)
        cuda.memcpy_htod(self.patterns_gpu, self.patterns)

        self.input_gpu = cuda.mem_alloc(4*((40*8)+16))
        self.result_gpu = gpuarray.empty((40,self.n), dtype=np.float32, allocator=cuda.mem_alloc)


    def match(self, inp):
        l = (len(inp)//8)-2
        x = l & -l # highest power of two which divides l, up to 8
        y = min(1024//x, self.n)
        cuda.memcpy_htod(self.input_gpu, inp.astype(np.float32))
        PatternCUDA.correlate(self.input_gpu, self.patterns_gpu, self.result_gpu, np.int32(self.start), np.int32(self.end), block=(x, y, 1), grid=(l//x, self.n//y))
        result = argmin(self.result_gpu, axis=1).get()
        return self.bytes[result[:l],0]

