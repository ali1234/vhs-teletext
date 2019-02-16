import time
from itertools import islice
import numpy as np

from OpenGL.GLUT import *
from OpenGL.GL import *


class VBIViewer(object):

    def __init__(self, lines, config, name = "VBI Viewer", width=800, height=256, nlines=32, tint=True, show_grid=True, show_slices=False):
        self.config = config
        self.show_grid = show_grid
        self.show_slices = show_slices
        self.tint = tint

        self.nlines = nlines

        self.lines_src = lines

        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)
        glutInitWindowSize(width,height)
        glutCreateWindow(name)

        glutDisplayFunc(self.display)
        glutReshapeFunc(self.reshape)

        glMatrixMode(GL_PROJECTION)
        glOrtho(0, config.line_length, 0, self.nlines, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, glGenTextures(1))
        glPixelStorei(GL_UNPACK_ALIGNMENT,1)

        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)

        glutMainLoop()

    def reshape(self, width, height):
        self.width = width
        self.height = height
        glViewport(0, 0, width, height)

    def draw_slice(self, slice, r, g, b, a=1.0):
        glColor4f(r, g, b, a)
        glBegin(GL_LINES)
        glVertex2f(slice.start, 0)
        glVertex2f(slice.start, self.nlines)
        glVertex2f(slice.stop, 0)
        glVertex2f(slice.stop, self.nlines)
        glEnd()

    def draw_h_grid(self, r, g, b, a=1.0):
        glColor4f(r, g, b, a)
        glBegin(GL_LINES)
        for x in range(self.nlines):
            glVertex2f(0, x)
            glVertex2f(2048, x)
        glEnd()

    def draw_bits(self, r, g, b, a=1.0):
        glColor4f(r, g, b, a)
        glBegin(GL_LINES)
        for x in self.config.bits[:-8:8]:
            glVertex2f(x, 0)
            glVertex2f(x, self.nlines)
        glEnd()

    def draw_lines(self):
        lines = list(islice(self.lines_src, 0, self.nlines))

        if len(lines) != self.nlines:
            exit(0)

        glEnable(GL_TEXTURE_2D)
        for n,l in enumerate(lines[::-1]):
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.config.line_length, 1, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, l.orig.tostring())
            if self.tint:
                if l.is_teletext:
                    glColor4f(0.5, 1.0, 0.7, 1.0)
                else:
                    glColor4f(1.0, 0.5, 0.5, 1.0)

            glBegin(GL_QUADS)

            glTexCoord2f(0, 1)
            glVertex2f(0, n)

            glTexCoord2f(0, 0)
            glVertex2f(0, (n+1))

            glTexCoord2f(1, 0)
            glVertex2f(self.config.line_length, (n+1))

            glTexCoord2f(1, 1)
            glVertex2f(self.config.line_length, n)

            glEnd()

        glDisable(GL_TEXTURE_2D)

    def display(self):

        self.draw_lines()

        if self.show_grid:

            if self.height / self.nlines > 3:
                self.draw_h_grid(0, 0, 0, 0.25)

            if self.width / 42 > 5:
                self.draw_bits(0, 0, 0, 0.25)

        if self.show_slices:
            self.draw_slice(self.config.line_start_slice, 1, 0, 0, 0.5)
            self.draw_slice(self.config.line_start_pre, 0, 1, 0, 0.5)
            self.draw_slice(self.config.line_start_post, 0, 0, 1, 0.5)

        glutSwapBuffers()
        glutPostRedisplay()
