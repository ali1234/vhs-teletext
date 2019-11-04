import itertools
import unittest

from teletext.elements import *


class TestElement(unittest.TestCase):

    cls = Element
    shape = (2,)

    def setUp(self):
        self._array = np.zeros(self.shape, dtype=np.uint8)
        self.element = self.cls(self._array.shape, self._array)

    def test_getitem(self):
        for i, j in itertools.product(range(self.shape[0]), range(256)):
            self._array[i] = j
            self.assertEqual(self.element[i], j)
            self.assertEqual(self.element[i], self._array[i])

    def test_setitem(self):
        for i, j in itertools.product(range(self.shape[0]), range(256)):
            self.element[i] = j
            self.assertEqual(self.element[i], j)
            self.assertEqual(self.element[i], self._array[i])

    def test_repr(self):
        self.assertEqual(repr(self.element), f'{self.cls.__name__}({repr(self._array)})')

    def test_errors(self):
        self.assertRaises(NotImplementedError, lambda: self.element.errors)

    def test_bytes(self):
        self.assertEqual(self._array.tobytes(), self.element.bytes)


class TestElementParity(TestElement):

    cls = ElementParity
    shape = (2, )

    def test_errors(self):
        self.assertTrue(all(self.element.errors))
        self._array[:] = 0x20 # correct parity
        self.assertFalse(any(self.element.errors))


class TestElementHamming(TestElement):

    cls = ElementHamming
    shape = (2, )

    def test_errors(self):
        pass


class TestMrag(TestElementHamming):

    cls = Mrag
    shape = (2, )

    def setUp(self):
        self._array = np.zeros(self.shape, dtype=np.uint8)
        self.element = self.cls(self._array)

    def test_magazine(self):
        for i in range(1, 9):
            self._array[:] = 0
            self.element.magazine = i
            self.assertEqual(self.element.magazine, i)
            self.assertTrue(any(self._array))

    def test_row(self):
        for i in range(32):
            self._array[:] = 0
            self.element.row = i
            self.assertEqual(self.element.row, i)
            self.assertTrue(any(self._array))


class TestDisplayable(TestElementParity):

    cls = Displayable
    shape = (11, )

    def test_place_string(self):
        self.element.place_string('Hello World', x=0)
        self.assertFalse(any(self.element.errors))


class TestElementDesignationCode(TestElement):

    cls: DesignationCode
    shape = (2, )

    def test_set_dc(self):
        for i in range(16):
            self.element.dc = i
            self.assertEqual(self.element.dc, i)


class TestElementFastext(TestElement):

    cls: Fastext
    shape = (40, )

    def test_set_checksum(self):
        for i in range(0, 0x10000, 199):
            self.element.checksum = i
            self.assertEqual(self.element.checksum, i)
