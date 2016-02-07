# * Copyright 2016 Alistair Buxton <a.j.buxton@gmail.com>
# *
# * License: This program is free software; you can redistribute it and/or
# * modify it under the terms of the GNU General Public License as published
# * by the Free Software Foundation; either version 3 of the License, or (at
# * your option) any later version. This program is distributed in the hope
# * that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# * GNU General Public License for more details.

import numpy

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule

import skcuda
from skcuda.misc import argmin
skcuda.misc._global_cublas_allocator = cuda.mem_alloc

from pattern import Pattern

class PatternCUDA(Pattern):

    mod = SourceModule("""
  __global__ void correlate(float *input, float *patterns, float *result)
  {
    int x = (threadIdx.x + (blockDim.x*blockIdx.x));
    int y = (threadIdx.y + (blockDim.y*blockIdx.y));
    int iidx = x * 8;
    int ridx = (y * 40) + x;
    int pidx = y * 14;

    float d = input[iidx] - patterns[pidx];
    result[ridx] = d * d;
    iidx += 1;
    pidx += 1;
    d = input[iidx] - patterns[pidx];
    result[ridx] += d * d;
    iidx += 1;
    pidx += 1;
    d = input[iidx] - patterns[pidx];
    result[ridx] += d * d;
    iidx += 1;
    pidx += 1;
    d = input[iidx] - patterns[pidx];
    result[ridx] += d * d;
    iidx += 1;
    pidx += 1;
    d = input[iidx] - patterns[pidx];
    result[ridx] += d * d;
    iidx += 1;
    pidx += 1;
    d = input[iidx] - patterns[pidx];
    result[ridx] += d * d;
    iidx += 1;
    pidx += 1;
    d = input[iidx] - patterns[pidx];
    result[ridx] += d * d;
    iidx += 1;
    pidx += 1;
    d = input[iidx] - patterns[pidx];
    result[ridx] += d * d;
    iidx += 1;
    pidx += 1;
    d = input[iidx] - patterns[pidx];
    result[ridx] += d * d;
    iidx += 1;
    pidx += 1;
    d = input[iidx] - patterns[pidx];
    result[ridx] += d * d;
    iidx += 1;
    pidx += 1;
    d = input[iidx] - patterns[pidx];
    result[ridx] += d * d;
    iidx += 1;
    pidx += 1;
    d = input[iidx] - patterns[pidx];
    result[ridx] += d * d;
    iidx += 1;
    pidx += 1;
    d = input[iidx] - patterns[pidx];
    result[ridx] += d * d;
    iidx += 1;
    pidx += 1;
    d = input[iidx] - patterns[pidx];
    result[ridx] += d * d;
  }

    """)

    correlate = mod.get_function("correlate")

    def __init__(self, filename):
        Pattern.__init__(self, filename)

        self.patterns_gpu = cuda.mem_alloc(self.patterns.nbytes)
        cuda.memcpy_htod(self.patterns_gpu, self.patterns)

        self.input_gpu = cuda.mem_alloc(4*((40*8)+6))
        self.result_gpu = gpuarray.empty((self.n,40), dtype=numpy.float32, allocator=cuda.mem_alloc)
        self.result = numpy.zeros((40,)).astype(numpy.float32)


    def match(self, inp):
        cuda.memcpy_htod(self.input_gpu, inp.astype(numpy.float32))
        PatternCUDA.correlate(self.input_gpu, self.patterns_gpu, self.result_gpu, block=(10, 64, 1), grid=(4, 128))
        return argmin(self.result_gpu, axis=0).get()

