#!/usr/bin/env python
# find packet 0

import sys
import numpy as np

from util import *

# BBC packet 0 example:
# 012345678901234567890123456789012345678901
# xxyyyyyyyyCEEFAX 1 217 Wed 25 Dec 18:29/53
# match1   "CEEFAX 1                  :  /  "
# match2   "eeeeeeeeenhheullenneullennennenn"
# e = exact
# n = number
# u = upper case
# l = lower case

class Finder(object):
    def __init__(self, match1, match2, pagepos):
        self.match1 = np.fromstring(match1, dtype = np.uint8)
        self.match2 = np.fromstring(match2, dtype = np.uint8)
        self.passrank = (5+self.calculaterank(self.match1))*0.5
        self.pagepos = pagepos
        self.possible_bytes = [hammbytes]*2 + [allbytes]*8
        for n in range(32):
            c = self.match2[n]
            if c == ord('e'):
                self.possible_bytes.append([makeparity(self.match1[n])])
            elif c == ord('u'):
                self.possible_bytes.append(upperbytes)
            elif c == ord('l'):
                self.possible_bytes.append(lowerbytes)
            elif c == ord('n'):
                self.possible_bytes.append(numberbytes)
            elif c == ord('h'):
                self.possible_bytes.append(hexbytes)
            else:
                self.possible_bytes.append(paritybytes)

    def findexact(self, visual):
        return ((self.match2 == ord('e')) & 
                (visual == self.match1)).sum()

    def findupper(self, visual):
        return ((self.match2 == ord('u')) & 
                (visual >= ord('A')) &
                (visual <= ord('Z'))).sum()

    def findlower(self, visual):
        return ((self.match2 == ord('l')) & 
                (visual >= ord('a')) &
                (visual <= ord('z'))).sum()

    def findnumber(self, visual):
        return ((self.match2 == ord('n')) & 
                (visual >= ord('0')) &
                (visual <= ord('9'))).sum()

    def findhex(self, visual):
        return ((self.match2 == ord('h')) & 
                (((visual >= ord('0')) & (visual <= ord('9'))) |
                 ((visual >= ord('A')) & (visual <= ord('F'))) |
                 ((visual >= ord('a')) & (visual <= ord('f'))))).sum()

    def calculaterank(self, visual):
        rank = 0
        rank += self.findexact(visual)
        rank += self.findupper(visual)*0.1
        rank += self.findlower(visual)*0.1
        rank += self.findnumber(visual)*0.2
        rank += self.findhex(visual)*0.1
        return rank

    def find(self, packet):
        rank = 0
        self.packet = np.fromstring(packet, dtype=np.uint8)
        (self.m,self.r),e = mrag(self.packet[:2])
        if self.r == 0:
            rank += 5
        rank += self.calculaterank(self.packet[10:]&0x7f)
        return (rank > self.passrank)

    def fixup(self):
        self.packet[0:2] = makemrag(self.m, 0)
        for n in range(0, 32):
            if self.match2[n] == ord('e'):
                self.packet[n+10] = makeparity(self.match1[n])
        return "".join([chr(x) for x in self.packet])
       

    
BBC1 = Finder("CEEFAX 1 217 Wed 25 Dec\x0318:29/53",
              "eeeeeeeeennneullenneullennennenn", 9)

if __name__=='__main__':

    F = BBC1

    while(True):
        tt = sys.stdin.read(42)
        if len(tt) < 42:
            exit(0)
        if F.find(tt):
            sys.stdout.write(F.fixup())
            sys.stdout.flush()
