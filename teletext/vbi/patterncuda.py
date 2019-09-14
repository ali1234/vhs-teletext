# * Copyright 2016 Alistair Buxton <a.j.buxton@gmail.com>
# *
# * License: This program is free software; you can redistribute it and/or
# * modify it under the terms of the GNU General Public License as published
# * by the Free Software Foundation; either version 3 of the License, or (at
# * your option) any later version. This program is distributed in the hope
# * that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# * GNU General Public License for more details.
import atexit

import numpy as np

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from pycuda.driver import ctx_flags

cuda.init()
cudadevice = cuda.Device(0)
cudacontext = cudadevice.make_context(flags=ctx_flags.SCHED_YIELD)
atexit.register(cudacontext.pop)

import skcuda
from skcuda.misc import _get_minmax_kernel
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
    argmin = _get_minmax_kernel(np.float32, "min")[1]

    def __init__(self, filename):
        Pattern.__init__(self, filename)

        if self.n&1023 != 0:
            raise ValueError('Number of patterns must be a multiple of 1024.')

        self.patterns_gpu = cuda.mem_alloc(self.patterns.nbytes)
        cuda.memcpy_htod(self.patterns_gpu, self.patterns)

        self.input_match = cuda.mem_alloc(4*((40*8)+16))
        self.result_match = gpuarray.empty((40,self.n), dtype=np.float32, allocator=cuda.mem_alloc)

        self.result_min = gpuarray.empty((40,), dtype=np.float32, allocator=cuda.mem_alloc)
        self.result_argmin = gpuarray.empty((40,), dtype=np.uint32, allocator=cuda.mem_alloc)

    def match(self, inp):
        l = (len(inp)//8)-2
        x = l & -l # highest power of two which divides l, up to 8
        y = min(1024//x, self.n)
        cuda.memcpy_htod(self.input_match, inp.astype(np.float32))

        PatternCUDA.correlate(
            self.input_match, self.patterns_gpu, self.result_match,
            np.int32(self.start), np.int32(self.end),
            block=(x, y, 1), grid=(l//x, self.n//y)
        )

        PatternCUDA.argmin(
            self.result_match, self.result_min, self.result_argmin,
            np.uint32(self.n), np.uint32(l),
            block=(32, 1, 1), grid=(l, 1, 1), stream=None
        )

        result = self.result_argmin.get()
        return self.bytes[result[:l],0]

