import unittest

import numpy as np

from teletext.coding import parity_encode, parity_decode, parity_errors, hamming8_encode, hamming8_decode, \
    hamming8_correctable_errors, hamming8_uncorrectable_errors


class ParityEncodeTestCase(unittest.TestCase):

    def _test_array(self, array: np.ndarray):

        encoded = parity_encode(array)
        self.assertEqual(encoded.dtype, np.int)
        self.assertEqual(array.shape, encoded.shape, 'Encoded data has wrong shape')

        #bitcounts = np.sum(np.unpackbits(encoded, axis=1), axis=1)
        #self.assertTrue(all(bitcounts & 1), 'Encoded data has wrong parity.')

        errors = parity_errors(encoded)
        self.assertFalse(any(errors), 'Encoded data has false errors.')

        decoded = parity_decode(encoded)
        self.assertEqual(decoded.dtype, np.int)
        self.assertTrue(all(decoded == array), 'Decoded data does not match original.')

        for b in range(8):
            oneerr = encoded ^ (1 << b)
            errors = parity_errors(oneerr)
            self.assertTrue(all(errors), 'Error not detected in encoded data.')

    def test_full_array(self):
        self._test_array(array=np.array(range(0x80), dtype=np.uint8))

    def test_unit_arrays(self):
        for i in range(0x80):
            self._test_array(array=np.array([i], dtype=np.uint8))

    def test_array_type(self):
        encoded = parity_encode(np.array(range(0x80), dtype=np.uint8))
        self.assertIsInstance(encoded, np.ndarray)
        #self.assertEqual(encoded.dtype, np.int)

    def test_list_type(self):
        encoded = parity_encode(list(range(0x80)))
        self.assertIsInstance(encoded, np.ndarray)
        #self.assertEqual(encoded.dtype, np.int)

    def test_unit_array_type(self):
        encoded = parity_encode(np.array([0], dtype=np.uint8))
        self.assertIsInstance(encoded, np.ndarray)
        #self.assertEqual(encoded.dtype, np.int)

    def test_unit_list_type(self):
        encoded = parity_encode([0])
        self.assertIsInstance(encoded, np.ndarray)
        #self.assertEqual(encoded.dtype, np.int)

    def test_int_type(self):
        encoded = parity_encode(0)
        #self.assertIsInstance(encoded, np.int64)

    def test_full_list(self):
        data = list(range(0x80))
        encoded = parity_encode(data)
        self.assertEqual(encoded.shape, (len(data),), 'Encoded data has wrong shape')

    def test_unit_list(self):
        data = [0]
        encoded = parity_encode(data)
        self.assertEqual(encoded.shape, (1, ), 'Encoded data has wrong shape')

    def test_ints(self):
        for i in range(0x80):
            encoded = parity_encode(i)

            errors = parity_errors(encoded)
            self.assertFalse(errors, 'Encoded data has false errors.')

            decoded = parity_decode(encoded)
            self.assertEqual(decoded, i, 'Decoded data does not match original.')

            for b in range(8):
                oneerr = encoded ^ (1 << b)
                errors = parity_errors(oneerr)
                self.assertTrue(errors, 'Error not detected in encoded data.')


class Hamming8TestCase(unittest.TestCase):

    def test_all(self):

        def h8_manual(d):
            d1 = d & 1
            d2 = (d >> 1) & 1
            d3 = (d >> 2) & 1
            d4 = (d >> 3) & 1

            p1 = (1 + d1 + d3 + d4) & 1
            p2 = (1 + d1 + d2 + d4) & 1
            p3 = (1 + d1 + d2 + d3) & 1
            p4 = (1 + p1 + d1 + p2 + d2 + p3 + d3 + d4) & 1

            return (p1 | (d1 << 1) | (p2 << 2) | (d2 << 3)
                    | (p3 << 4) | (d3 << 5) | (p4 << 6) | (d4 << 7))

        for i in range(0x10):
            self.assertTrue(hamming8_encode(i) == h8_manual(i))

        data = np.arange(0x10, dtype=np.uint8)
        encoded = hamming8_encode(data)
        self.assertTrue(all(hamming8_decode(encoded) == data))
        self.assertTrue(not any(hamming8_correctable_errors(encoded)))
        self.assertTrue(not any(hamming8_uncorrectable_errors(encoded)))

        for b1 in range(8):
            oneerr = encoded ^ (1 << b1)
            self.assertTrue(all(hamming8_decode(oneerr) == data))
            self.assertTrue(all(hamming8_correctable_errors(oneerr)))
            self.assertTrue(not any(hamming8_uncorrectable_errors(oneerr)))
            for b2 in range(8):
                if b2 != b1:
                    twoerr = oneerr ^ (1 << b2)
                    self.assertTrue(not any(hamming8_correctable_errors(twoerr)))
                    self.assertTrue(all(hamming8_uncorrectable_errors(twoerr)))



