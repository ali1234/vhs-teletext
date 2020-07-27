from OpenGL.GLU import *
from OpenGL.GL import *

from PyQt5.QtWidgets import QOpenGLWidget

import numpy as np

from teletext.vbi.config import Config


class LineWidget(object):
    def __init__(self, glw):
        self._glw = glw
        self._glw.paintGL = self.paintGL
        self._glw.initializeGL = self.initializeGL
        self._glw.resizeGL = self.resizeGL

        self.show_grid = True
        self.tint = True

        self.line_attr = 'resampled'

        self.config = Config()
        self.lines = []

    def setlines(self, lines):
        self.lines = lines

        self._glw.update()

    def initializeGL(self):
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, glGenTextures(1))
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)

    def resizeGL(self, width, height):
        self.width = width
        self.height = height
        glViewport(0, 0, width, height)

    def draw_slice(self, slice, r, g, b, a=1.0):
        glColor4f(r, g, b, a)
        glBegin(GL_LINES)
        glVertex2f(slice.start, 0)
        glVertex2f(slice.start, len(self.lines))
        glVertex2f(slice.stop, 0)
        glVertex2f(slice.stop, len(self.lines))
        glEnd()

    def draw_h_grid(self, r, g, b, a=1.0):
        glColor4f(r, g, b, a)
        glBegin(GL_LINES)
        for x in range(len(self.lines)):
            glVertex2f(0, x)
            glVertex2f(self.config.resample_size, x)
        glEnd()

    def draw_bits(self, r, g, b, a=1.0):
        glColor4f(r, g, b, a)
        glBegin(GL_LINES)
        for x in range(0, 368,8):
            glVertex2f((x*8)+90, 0)
            glVertex2f((x*8)+90, len(self.lines))
        glEnd()

    def draw_freq_bins(self, n, r, g, b, a=1.0):
        glColor4f(r, g, b, a)
        glBegin(GL_LINES)
        for x in self.config.fftbins:
            glVertex2f(self.config.resample_size*x/256, 0)
            glVertex2f(self.config.resample_size*x/256, len(self.lines))
        glEnd()

    def draw_lines(self):

        glEnable(GL_TEXTURE_2D)
        for n,l in enumerate(self.lines[::-1]):
            array = getattr(l, self.line_attr)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, array.size, 1, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, np.clip(array, 0, 255).astype(np.uint8).tostring())
            if self.tint:
                if l.is_teletext:
                    glColor4f(0.5, 1.0, 0.7, 1.0)
                else:
                    glColor4f(1.0, 0.5, 0.5, 1.0)
            else:
                glColor4f(1.0, 1.0, 1.0, 1.0)

            glBegin(GL_QUADS)

            glTexCoord2f(0, 1)
            glVertex2f(0, n)

            glTexCoord2f(0, 0)
            glVertex2f(0, (n+1))

            glTexCoord2f(1, 0)
            glVertex2f(self.config.resample_size, (n+1))

            glTexCoord2f(1, 1)
            glVertex2f(self.config.resample_size, n)

            glEnd()

        glDisable(GL_TEXTURE_2D)

    def paintGL(self):
        if len(self.lines):
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            glOrtho(0, self.config.resample_size, 0, len(self.lines), -1, 1)
            glMatrixMode(GL_MODELVIEW)

            self.draw_lines()

            if self.height / len(self.lines) > 3:
                self.draw_h_grid(0, 0, 0, 0.25)

            if self.show_grid:
                if self.line_attr == 'fft':
                    self.draw_freq_bins(256, 1, 1, 1, 0.5)
                elif self.line_attr == 'rolled' and self.width / 42 > 5:
                    self.draw_bits(1, 1, 1, 0.5)
                elif self.line_attr == 'resampled':
                    self.draw_slice(self.config.start_slice, 0, 1, 0, 0.5)
