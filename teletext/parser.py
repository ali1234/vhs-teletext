import re

from . import charset

_unicode13 = False


class Parser(object):

    "Abstract base class for parsers"

    def __init__(self, tt):
        self.tt = tt
        self._state = {}
        self.parse()

    def reset(self):
        self._state['fg'] = 7
        self._state['bg'] = 0
        self._state['dw'] = False
        self._state['dh'] = False
        self._state['mosaic'] = False
        self._state['solid'] = True
        self._state['flash'] = False
        self._state['conceal'] = False
        self._state['boxed'] = False
        self._state['rendered'] = True

        self._heldmosaic = ' '
        self._heldsolid = True
        self._held = False
        self._esc = False
        self._codepage = 0 # not implemented

    def setstate(self, **kwargs):
        any = False
        for state, value in kwargs.items():
            if value != self._state[state]:
                self._state[state] = value
                any = True
                if state in ['dw', 'dh']:
                    self._heldmosaic = ' '
                getattr(self, state+'Changed', lambda: None)()
        if any:
            getattr(self, 'stateChanged', lambda: None)()

    def ttchar(self, c):
        if self._state['mosaic'] and c not in range(0x41, 0x5B):
            if _unicode13:
                return charset.g1[c]
            else:
                return chr(c+0xee00) if self._state['solid'] else chr(c+0xede0)
        else:
            return charset.g0[c]

    def _emitcharacter(self, c):
        getattr(self, 'emitcharacter', lambda x: None)(c)
        if self._state['dw']:
            self._state['rendered'] = not self._state['rendered']
        else:
            self._state['rendered'] = True

    def emitcode(self):
        if self._held:
            tmp = self._state['solid']
            self._state['solid'] = self._heldsolid
            self._emitcharacter(self._heldmosaic)
            self._state['solid'] = tmp
        else:
            self._emitcharacter(' ')

    def setat(self, **kwargs):
        self.setstate(**kwargs)
        self.emitcode()

    def setafter(self, **kwargs):
        self.emitcode()
        self.setstate(**kwargs)

    def parsebyte(self, b):
        h, l = int(b&0xf0), int(b&0x0f)
        if h == 0x0:
            if l < 8:
                self.setafter(fg=l, mosaic=False, conceal=False)
                self._heldmosaic = ' '
            elif l == 0x8: # flashing
                self.setafter(flash=True)
            elif l == 0x9: # steady
                self.setat(flash=False)
            elif l == 0xa: # end box
                self.setafter(boxed=False)
            elif l == 0xb: # start box
                self.setafter(boxed=True)
            else: # sizes
                dh, dw = bool(l&1), bool(l&2)
                if dh or dw:
                    self.setafter(dh=dh, dw=dw)
                else:
                    self.setat(dh=dh, dw=dw)

        elif h == 0x10:
            if l < 8:
                self.setafter(fg=l, mosaic=True, conceal=False)
            elif l == 0x8: # conceal
                self.setat(conceal=True)
            elif l == 0x9: # contiguous mosaic
                self.setat(solid=True)
            elif l == 0xa: # separated mosaic
                self.setat(solid=False)
            elif l == 0xb: # esc/switch
                self.emitcode()
                self._esc = not self._esc
            elif l == 0xc: # black background
                self.setat(bg = 0)
            elif l == 0xd: # new background
                self.setat(bg = self._state['fg'])
            elif l == 0xe: # hold mosaic
                self._held = True
                self.emitcode()
            elif l == 0xf: # release mosaic
                self.emitcode()
                self._held = False
        else:
            c = self.ttchar(b)
            if self._state['mosaic'] and (b & 0x20):
                self._heldmosaic = c
                self._heldsolid = self._state['solid']
            self._emitcharacter(c)

    def parse(self):
        self.reset()
        for c in self.tt&0x7f:
            self.parsebyte(c)
