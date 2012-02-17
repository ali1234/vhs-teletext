#!/usr/bin/env python

import os
import pygame
from pygame.locals import *


def bin2srf(binstr):
    length = len(binstr) * 8
    grey = mono2greyLSB(binstr, 1024, 64, 0, 1)
    surface = pygame.image.fromstring(grey, (1024, 64), "P")
    surface.set_palette([(0,0,0), (255,255,255)])
    return surface

def main():

    vbi = os.open('/dev/vbi0', os.O_RDONLY)

    pygame.init()

    os.environ['SDL_VIDEO_WINDOW_POS'] = str(0) + "," + str(0)

    height = 32 # max number of lines accepted by tv
    repeat = 2 # repeat each packet n times in case of errors
               # you probably want 1 or 2

    width = 1024
    offset= 0
    end = width+offset

    screen = pygame.display.set_mode((width,216), DOUBLEBUF)
    pygame.display.set_caption('Teletext')

    running = 1
    palette = [(x,x,x) for x in range(256)]
    while running:
        data = os.read(vbi,65536)
        data = "".join([data[x] for x in range(0, len(data), 2)])
        buff = ""
        for i in range(32):
            if i == 16:
                buff += "\x00"*width*6
            start = width*i
            for n in range(2):
                buff += "\x00"*width
            for n in range(4):
                buff += data[start+offset:start+end]


        #print len(buff)
        surface = pygame.image.frombuffer(buff, (width, 33*6), "P")
        surface.set_palette(palette)
        screen.blit(surface, (0,2))
        pygame.display.flip()        
        




if __name__ == '__main__': main()
