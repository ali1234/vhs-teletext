# * Copyright 2016 Alistair Buxton <a.j.buxton@gmail.com>
# *
# * License: This program is free software; you can redistribute it and/or
# * modify it under the terms of the GNU General Public License as published
# * by the Free Software Foundation; either version 3 of the License, or (at
# * your option) any later version. This program is distributed in the hope
# * that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# * GNU General Public License for more details.

from functools import partial
import os


class RawLineReader(object):
    def __init__(self, filename, line_length, start=0, stop=-1):
        self.filename = filename
        self.line_length = line_length
        self.start = start
        self.stop = stop
        self.pos = start

    def __enter__(self):
        self.file = open(self.filename, 'rb')
        self.size = None
        try:
            self.file.seek(0, os.SEEK_END)
            self.size = self.file.tell()//self.line_length
            self.file.seek(self.start*self.line_length, os.SEEK_SET)
        except OSError:
            pass

        return self

    def __exit__(self, *args):
        self.file.close()

    def __len__(self):
        return self.size

    def __iter__(self):
        rawlines = iter(partial(self.file.read, self.line_length), b'')
        for n,rl in enumerate(rawlines):
            offset = n + self.start
            if len(rl) < self.line_length:
                return
            elif offset == self.stop:
                return
            else:
                yield (offset,rl)
