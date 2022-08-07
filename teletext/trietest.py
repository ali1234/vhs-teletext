import lzma
from collections import defaultdict

import psutil
import humanize
import numpy as np
from tqdm import tqdm
import dahuffman


class Compressor:
    def __init__(self):
        self.packet_count = 0

    @property
    def original_size(self):
        return self.packet_count * 42

    @property
    def disk_size(self):
        return self.original_size

    @property
    def ratio(self):
        return self.original_size / (self.disk_size or 1)

    @property
    def percent(self):
        return 100 / (self.ratio or 1)

    @property
    def memory(self):
        return psutil.Process().memory_info().rss

    def __str__(self):
        return f', {humanize.naturalsize(self.memory, binary=True)}, {self.percent:.1f}%, {self.ratio:.1f}x'


class Box(Compressor):
    filters = [
        {"id": lzma.FILTER_DELTA, "dist": 5},
        {"id": lzma.FILTER_LZMA2, "preset": 7 | lzma.PRESET_EXTREME},
    ]
    def __init__(self):
        super().__init__()
        self.compressed_total = 0

    def insert(self, packet):
        self.packet_count += 1
        comp = lzma.compress(packet, format=lzma.FORMAT_RAW, filters=self.filters)
        self.compressed_total += len(comp)

    @property
    def disk_size(self):
        return self.compressed_total + (self.packet_count * 4)

    def dump(self):
        print('  Packet count:', humanize.intword(self.packet_count))
        print(' Original size:', humanize.naturalsize(self.original_size, binary=True))
        print('Est. disk size:', humanize.naturalsize(self.disk_size, binary=True))
        print(f'  Compression: {self.percent:.1f}% ({self.ratio:.1f}x)')


class Trie(Compressor):

    def __init__(self):
        super().__init__()
        self.child = {}
        self.node_count = 0
        self.succ_count = 0
        self.leaf_count = 0

    def insert(self, packet):
        current = self.child
        for l in packet[:-2]:
            if l not in current:
                current[l] = {}
                self.node_count += 1
            current = current[l]
        if packet[-2] not in current:
            current[packet[-2]] = {}
            self.node_count += 1
            self.succ_count += 1
        current = current[packet[-2]]
        if packet[-1] not in current:
            current[packet[-1]] = 0
            self.node_count += 1
            self.leaf_count += 1
        current[packet[-1]] += 1
        self.packet_count += 1

    @property
    def node_size(self):
        return self.node_count + (self.succ_count * 3)

    @property
    def ref_size(self):
        return self.packet_count * 4

    @property
    def disk_size(self):
        return self.node_size + self.ref_size

    @property
    def ram_size(self):
        # 4 bytes each for parent, sibling, children
        return self.node_count * (1 + 4)

    def dump(self):
        print('  Packet count:', humanize.intword(self.packet_count))
        print('    Node count:', humanize.intword(self.node_count))
        print('    Succ count:', humanize.intword(self.succ_count))
        print('    Leaf count:', humanize.intword(self.leaf_count))
        print(' Original size:', humanize.naturalsize(self.original_size, binary=True))
        print('Est. disk size:', humanize.naturalsize(self.disk_size, binary=True))
        print('         Nodes:', humanize.naturalsize(self.node_size, binary=True))
        print('          Refs:', humanize.naturalsize(self.ref_size, binary=True))
        print(' Flat RAM size:', humanize.naturalsize(self.leaf_count*42, binary=True))
        print(' Trie RAM size:', humanize.naturalsize(self.ram_size, binary=True))
        print(f'  Compression: {self.percent:.1f}% ({self.ratio:.1f}x)')


class Huff(Compressor):
    def __init__(self):
        super().__init__()
        self.compressed_size = 0
        self.freqs = defaultdict(int)
        self.packets = set()

    def insert(self, packet):
        self.packet_count += 1
        if packet not in self.packets:
            self.packets.add(packet)
            for b in packet:
                self.freqs[b] += 1

    @property
    def unique_size(self):
        return len(self.packets) * 42

    @property
    def disk_size(self):
        return self.compressed_size + (self.packet_count * 4) + 256

    @property
    def node_size(self):
        return self.compressed_size + 256

    @property
    def ref_size(self):
        return self.packet_count * 4

    def dump(self):
        codec = dahuffman.HuffmanCodec.from_frequencies(self.freqs)
        for p in tqdm(self.packets):
            self.compressed_size += len(codec.encode(p))

        print('  Packet count:', humanize.intword(self.packet_count))
        print('  Unique count:', humanize.intword(len(self.packets)))
        print(' Original size:', humanize.naturalsize(self.original_size, binary=True))
        print('   Unique size:', humanize.naturalsize(self.unique_size, binary=True))
        print('Est. disk size:', humanize.naturalsize(self.disk_size, binary=True))
        print('         Nodes:', humanize.naturalsize(self.node_size, binary=True))
        print('          Refs:', humanize.naturalsize(self.ref_size, binary=True))
        print(f'  Compression: {self.percent:.1f}% ({self.ratio:.1f}x)')



class OOM(Exception):
    pass


def main(progress, packets):
    t = Trie()
    limit = 16*1024*1024*1024

    progress.postfix.append(t)

    try:
        for n, p in enumerate(packets, start=1):
            mrag = bytes([((p.mrag.magazine & 0x7) << 5) | p.mrag.row])
            t.insert(mrag + p.bytes[2:])
            if (n & 0xffff) == 0:
                if t.memory > limit:
                    raise OOM
    except OOM:
        progress.close()
        print('Stopping at memory limit:', humanize.naturalsize(limit, binary=True))
    except KeyboardInterrupt:
        progress.close()
        print('Stopping due to keyboard interrupt.')
    finally:
        t.dump()




