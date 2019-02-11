from itertools import islice
import numpy as np

from OpenGL.GLUT import *
from OpenGL.GL import *


class VBIViewer(object):

    def __init__(self, lines, config, name = "VBI Viewer", width=1024, height=512, nlines=32, pass_teletext=True, pass_rejects=False, show_grid=False):
        self.config = config
        self.show_grid = show_grid

        self.nlines = nlines

        self.lines_src = lines

        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)
        glutInitWindowSize(width,height)
        glutCreateWindow(name)

        glutDisplayFunc(self.display)

        glMatrixMode(GL_PROJECTION)
        glOrtho(0, config.line_length, 0, self.nlines, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, glGenTextures(1))
        glPixelStorei(GL_UNPACK_ALIGNMENT,1)

        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)


        glutMainLoop()

    def display(self):

        vbi = np.array([x.orig for x in islice(self.lines_src, 0, self.nlines)], dtype=np.uint8).tostring()
#        time.sleep(0.1)
        if not sys.stdout.isatty():
            sys.stdout.write(vbi)

        if len(vbi) != self.config.line_length*self.nlines:
#            return
            exit(0)

        glEnable(GL_TEXTURE_2D)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.config.line_length, self.nlines, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, vbi)


        glBegin(GL_QUADS)

        glTexCoord2f(0, 1)
        glVertex2f(   0, 0)

        glTexCoord2f(0, 0)
        glVertex2f(   0, self.nlines)

        glTexCoord2f(1, 0)
        glVertex2f(self.config.line_length, self.nlines)

        glTexCoord2f(1, 1)
        glVertex2f(self.config.line_length, 0)

        glEnd()

        if 1:

            glDisable(GL_TEXTURE_2D)

            glBegin(GL_LINES)

        if self.show_grid:
            glColor3f(0, 0, 0)

            if self.nlines < 65:
              for x in range(self.nlines):
                glVertex2f(0, x)
                glVertex2f(2048, x)

        if self.show_grid:
            glColor3f(1, 0, 0)

            for x in self.config.line_start_range:
                glVertex2f(x, 0)
                glVertex2f(x, self.nlines)

            glColor3f(0, 1, 0)
            for x in self.config.line_start_pre:
                glVertex2f(x, 0)
                glVertex2f(x, self.nlines)

            for x in self.config.line_start_post:
                glVertex2f(x, 0)
                glVertex2f(x, self.nlines)


            glVertex2f(self.config.line_trim, 0)
            glVertex2f(self.config.line_trim, self.nlines)

        if self.show_grid:
            glColor3f(0, 0, 1)
            for x in self.config.bits[::8]:
                glVertex2f(x, 0)
                glVertex2f(x, self.nlines)

        if 1:
            glEnd()

        glutSwapBuffers()
        glutPostRedisplay()
