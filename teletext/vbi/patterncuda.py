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

from .pattern import Pattern

cuda.init()
cudadevice = cuda.Device(0)
cudacontext = cudadevice.make_context(flags=ctx_flags.SCHED_YIELD)
atexit.register(cudacontext.pop)


class PatternCUDA(Pattern):

    correlate = SourceModule("""
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
    """).get_function("correlate")

    # argmin from scikit-cuda/blob/master/skcuda/misc.py
    argmin = SourceModule("""
        /*
        Copyright (c) 2009-2019, Lev E. Givon. All rights reserved.

        Redistribution and use in source and binary forms, with or without modification, are 
        permitted provided that the following conditions are met:

            Redistributions of source code must retain the above copyright notice, this list of 
            conditions and the following disclaimer.
            Redistributions in binary form must reproduce the above copyright notice, this list 
            of conditions and the following disclaimer in the documentation and/or other materials 
            provided with the distribution.
            Neither the name of Lev E. Givon nor the names of any contributors may be used to 
            endorse or promote products derived from this software without specific prior 
            written permission.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
        EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
        OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
        SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
        SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT 
        OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) 
        HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR 
        TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
        EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
        */
 
        __global__ void minmax_row_kernel(float* mat, float* target,
                                          unsigned int* idx_target,
                                          unsigned int width,
                                          unsigned int height) {
            __shared__ float max_vals[32];
            __shared__ unsigned int max_idxs[32];
            float cur_max = 3.4028235e+38;
            unsigned int cur_idx = 0;
            float val = 0;
    
            for (unsigned int i = threadIdx.x; i < width; i += 32) {
                val = mat[blockIdx.x * width + i];
    
                if (val < cur_max) {
                    cur_max = val;
                    cur_idx = i;
                }
            }
            max_vals[threadIdx.x] = cur_max;
            max_idxs[threadIdx.x] = cur_idx;
            __syncthreads();
    
            if (threadIdx.x == 0) {
                cur_max = 3.4028235e+38;
                cur_idx = 0;
    
                for (unsigned int i = 0; i < 32; i++)
                    if (max_vals[i] < cur_max) {
                        cur_max = max_vals[i];
                        cur_idx = max_idxs[i];
                    }
    
                target[blockIdx.x] = cur_max;
                idx_target[blockIdx.x] = cur_idx;
            }
        }
    """).get_function("minmax_row_kernel")

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

