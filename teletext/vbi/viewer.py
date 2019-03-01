import time

import numpy as np
from itertools import islice

from OpenGL.GLUT import *
from OpenGL.GL import *


class VBIViewer(object):

    def __init__(self, lines, config, name = "VBI Viewer", width=800, height=512, nlines=32, tint=True, show_grid=True, show_slices=False, pause=False):
        self.config = config
        self.show_grid = show_grid
        self.tint = tint
        self.pause = pause
        self.single_step = False
        self.name = name

        self.line_attr = 'original'

        self.nlines = nlines

        self.lines_src = lines
        self.lines = list(islice(self.lines_src, 0, self.nlines))

        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)
        glutInitWindowSize(width,height)
        glutCreateWindow(name)
        self.set_title()

        glutDisplayFunc(self.display)
        glutReshapeFunc(self.reshape)
        glutKeyboardFunc(self.keyboard)
        glutMouseFunc(self.mouse)

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

    def keyboard(self, key, x, y):
        if key == b'g':
            self.show_grid ^= True
        elif key == b'c':
            self.tint ^= True
        elif key == b'p':
            self.pause ^= True
        elif key == b'n':
            self.single_step = True
        elif key == b'r':
            self.dumpline(x, y, teletext=False)
        elif key == b't':
            self.dumpline(x, y, teletext=True)
        elif key == b'R':
            self.dumpall(teletext=False)
        elif key == b'T':
            self.dumpall(teletext=True)
        elif key == b'1':
            self.line_attr = 'original'
        elif key == b'2':
            self.line_attr = 'rolled'
        elif key == b'3':
            self.line_attr = 'fft'
        elif key == b'4':
            self.line_attr = 'chopped'
        elif key == b'q':
            exit(0)
        self.set_title()

    def mouse(self, button, state, x, y):
        if state == GLUT_DOWN:
            l = self.lines[self.nlines * y//self.height]
            if button == 3:
                l.roll += 1
            elif button == 4:
                l.roll -= 1
            if l.is_teletext:
                print(l.deconvolve().debug.decode('utf8')[:-1], 'er:', l.roll)
            else:
                print(l._reason)
            sys.stdout.flush()

    def dumpline(self, x, y, teletext):
        if teletext:
            print('Writing to teletext.vbi')
            fn = 'teletext.vbi'
        else:
            print('Writing to reject.vbi')
            fn = 'reject.vbi'
        l = self.lines[self.nlines * y // self.height]
        with open(fn, 'ab') as f:
            f.write(l._original_bytes)

    def dumpall(self, teletext):
        if teletext:
            print('Writing all to teletext.vbi')
            fn = 'teletext.vbi'
        else:
            print('Writing all to reject.vbi')
            fn = 'reject.vbi'
        with open(fn, 'ab') as f:
            for l in self.lines:
                f.write(l._original_bytes)

    def set_title(self):
        glutSetWindowTitle(f'{self.name} - {self.line_attr}{" (paused)" if self.pause else ""}')

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

    def draw_freq_bins(self, n, r, g, b, a=1.0):
        glColor4f(r, g, b, a)
        glBegin(GL_LINES)
        for x in self.config.fftbins:
            glVertex2f(self.config.line_length*x/256, 0)
            glVertex2f(self.config.line_length*x/256, self.nlines)
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
            glVertex2f(self.config.line_length, (n+1))

            glTexCoord2f(1, 1)
            glVertex2f(self.config.line_length, n)

            glEnd()

        glDisable(GL_TEXTURE_2D)

    def display(self):

        self.draw_lines()

        if self.height / self.nlines > 3:
            self.draw_h_grid(0, 0, 0, 0.25)

        if self.show_grid:
            if self.line_attr == 'fft':
                self.draw_freq_bins(256, 1, 1, 1, 0.5)
            elif self.line_attr == 'rolled' and self.width / 42 > 5:
                self.draw_bits(1, 1, 1, 0.5)
            elif self.line_attr == 'original':
                self.draw_slice(self.config.start_slice, 0, 1, 0, 0.5)

        glutSwapBuffers()
        glutPostRedisplay()

        if self.pause and not self.single_step:
            time.sleep(0.1)
        else:
            next_lines = list(islice(self.lines_src, 0, self.nlines))

            if len(next_lines) > 0:
                self.lines = next_lines
            self.single_step = False
