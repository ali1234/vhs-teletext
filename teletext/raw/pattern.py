import struct
import numpy

class Pattern(object):
    def __init__(self, filename):
        f = open(filename, 'rb')
        self.inlen,self.outlen,self.n = struct.unpack('>III', f.read(12))
        self.patterns = numpy.fromstring(f.read(self.inlen*self.n), dtype=numpy.uint8)
        self.patterns = self.patterns.reshape((self.n, self.inlen))
        self.patterns = self.patterns.astype(numpy.float32)
        self.bytes = numpy.fromstring(f.read(self.outlen*self.n), dtype=numpy.uint8)
        self.bytes = self.bytes.reshape((self.n, self.outlen))
        f.close()

    def match(self, inp):
        diffs = self.patterns - inp
        diffs = diffs * diffs
        return self.bytes[numpy.argmin(numpy.sum(diffs, axis=1))]







# Classes used to build pattern files from training data.
# Not used during normal decoding.

class PatternBuilder(object):

    def __init__(self, inwidth):
        self.patterns = []
        self.inwidth = inwidth

    def read_array(self, filename):
        data = open(filename, 'rb').read()
        a = numpy.fromstring(data, dtype=numpy.uint8)
        a = a.reshape((len(a)/self.inwidth,self.inwidth))
        return numpy.mean(a, axis=0).astype(numpy.uint8)

    def write_patterns(self, filename):
        f = open(filename, 'wb')
        header = struct.pack('>III', len(self.patterns[0][0]), len(self.patterns[0][1]), len(self.patterns))
        f.write(header)
        for (i,o) in self.patterns:
            i.tofile(f)
        for (i,o) in self.patterns:
            f.write(o)
        f.close()

    def add_file(self, filename):
        i = self.read_array(filename.strip())
        o = self.get_output_bytes(filename)
        print i, o
        self.patterns.append((i, o))

class PatternBuilderMrag(PatternBuilder):

    def __init__(self):
        PatternBuilder.__init__(self, 18)

    def get_output_bytes(self, filename):
        filename = filename.split('/')[-1]
        byte1 = int(filename[9:1:-1], 2)
        byte2 = int(filename[18:10:-1], 2)
        return numpy.array([byte1, byte2], dtype=numpy.uint8)

class PatternBuilderParity(PatternBuilder):

    def __init__(self):
        PatternBuilder.__init__(self, 14)

    def get_output_bytes(self, filename):
        filename = filename.split('/')[-1]
        byte1 = int(filename[14:6:-1], 2)
        return numpy.array([byte1], dtype=numpy.uint8)





if __name__ == '__main__':
    import sys

    M = PatternBuilderMrag()
    with open(sys.argv[1]) as mrag:
        for line in mrag:
            M.add_file(line.strip())
    M.write_patterns('mrag_patterns')

    P = PatternBuilderParity()
    with open(sys.argv[2]) as par:
        for line in par:
            P.add_file(line.strip())
    P.write_patterns('parity_patterns')

