import itertools
import unittest

from teletext.elements import *



class TestElement(unittest.TestCase):

    cls = Element
    shape = (2,)
    sized = False
    needsmrag = False

    def make_element(self, array):
        args = []
        if not self.sized:
            args.append(self._array.shape)
        args.append(array)
        if self.needsmrag:
            mrag = Mrag()
            mrag.magazine = 1
            mrag.row = 0
            args.append(mrag)
        return self.cls(*args)

    def setUp(self):
        self._array = np.zeros(self.shape, dtype=np.uint8)
        self.element = self.make_element(self._array)

    def test_type(self):
        self.assertIsInstance(self.element, self.cls)

    def test_wrong_shape(self):
        with self.assertRaises(IndexError):
            array = np.zeros((1,1), dtype=np.uint8)
            self.make_element(array)

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
        with self.assertRaises(NotImplementedError):
            self.element.errors()

    def test_bytes(self):
        self.assertEqual(self._array.tobytes(), self.element.bytes)


class TestElementParity(TestElement):

    cls = ElementParity
    shape = (2, )

    def test_errors(self):
        self.assertTrue(all(self.element.errors))
        self._array[:] = 0x20 # correct parity
        self.assertFalse(any(self.element.errors))


class TestElementHamming(TestElementParity):

    cls = ElementHamming
    shape = (2, )

    def test_errors(self):
        self.assertTrue(all(self.element.errors))
        self._array[:] = 0x15  # correct hamming8
        self.assertFalse(any(self.element.errors))


class TestMrag(TestElementHamming):

    cls = Mrag
    shape = (2, )
    sized = True

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


class TestPage(TestElementHamming):

    cls = Page
    shape = (2,)


class TestHeader(TestPage):

    cls = Header
    shape = (40,)
    sized = True


class TestPageLink(TestPage):

    cls = PageLink
    shape = (6,)
    sized = True
    needsmrag = True


class TestDesignationCode(TestElementHamming):

    cls = DesignationCode
    shape = (1, )

    def test_set_dc(self):
        for i in range(16):
            self.element.dc = i
            self.assertEqual(self.element.dc, i)


class TestFastext(TestDesignationCode):

    cls = Fastext
    shape = (40,)
    sized = True
    needsmrag = True

    def test_errors(self):
        pass # TODO
