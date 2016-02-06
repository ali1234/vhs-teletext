#!/usr/bin/env python

import os, random
from math import sin,cos,pi,floor
from OpenGL.GL import *
from OpenGL.GLU import *
import pygame, pygame.image, pygame.key
from pygame.locals import *

w = 1600
h = 128

def resize((width, height)):
    if height==0:
        height=1.0
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, 2048, 0, 256)
#    gluOrtho2D(-(width/(2*height)), width/(2*height), 0, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def init():
    glShadeModel(GL_SMOOTH)
    glClearColor(1.0, 1.0, 1.0, 0.0)
    glClearDepth(1.0)
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT )
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)
    glDisable(GL_DEPTH_TEST)    

def main():

    video_flags = OPENGL|DOUBLEBUF
    
    pygame.init()
    surface = pygame.display.set_mode((w,h), video_flags)
    pygame.key.set_repeat(100,30)

    resize((w,h))
    init()
    

def draw(data, guess):
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT )
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glColor4f(1.0,0.0,0.0,1.0)
    glBegin(GL_LINE_STRIP)
    for n in range(2048):
        glVertex2f(n, data[n])
    glEnd()

    glColor4f(0.0,0.5,0.0,1.0)
    glBegin(GL_LINE_STRIP)
    for n in range(2048):
        glVertex2f(n, guess[n])
    glEnd()


    pygame.display.flip()



# encode with:
# transcode --use_rgb -i output.raw -x raw=RGB,null -y xvid4 -o output.xvid.avi -k -z  -f 50
# (for xvid)

# video format looks like:
# pixel { byte r,g,b };  // 24 bit
# line  { pixel[720] }; // left to right
# frame { line[576] }; // bottom to top
# video { frame[n] }; // n frames

if __name__ == '__main__': main()
