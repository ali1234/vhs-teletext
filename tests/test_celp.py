import unittest

import numpy as np

try:
    from teletext.celp import CELPDecoder, LtpCodebook, hamming7_dec
except ImportError:
    CELPDecoder = None


@unittest.skipIf(CELPDecoder is None, 'CELP decoding not available (needs the audio extra).')
class TestLtpCodebook(unittest.TestCase):

    def setUp(self):
        self.ltp = LtpCodebook(40)
        self.ltp.insert(np.arange(LtpCodebook.max_lag, dtype=np.double))

    def test_long_lag(self):
        # lag >= subframe length: a plain copy of the past excitation
        np.testing.assert_array_equal(self.ltp.get(50), self.ltp.history[-50:-10])

    def test_max_lag(self):
        np.testing.assert_array_equal(self.ltp.get(147), self.ltp.history[:40])

    def test_short_lag(self):
        # lag < subframe length: the last lag samples are repeated
        np.testing.assert_array_equal(self.ltp.get(20), np.tile(self.ltp.history[-20:], 2))
        np.testing.assert_array_equal(self.ltp.get(30)[30:], self.ltp.history[-30:-20])

    def test_insert(self):
        self.ltp.insert(np.full(40, -1.0))
        self.assertEqual(self.ltp.history.shape, (LtpCodebook.max_lag, ))
        np.testing.assert_array_equal(self.ltp.history[-40:], -1)


@unittest.skipIf(CELPDecoder is None, 'CELP decoding not available (needs the audio extra).')
class TestHamming(unittest.TestCase):

    def test_corrects_single_errors(self):
        valid = {d: [p for p in range(8) if hamming7_dec[d, p] == d] for d in range(16)}
        for d, parity in valid.items():
            self.assertEqual(len(parity), 1)
            for bit in range(7):
                word = ((d << 3) | parity[0]) ^ (1 << bit)
                self.assertEqual(hamming7_dec[word >> 3, word & 7], 0x10 | d)


@unittest.skipIf(CELPDecoder is None, 'CELP decoding not available (needs the audio extra).')
class TestDecoder(unittest.TestCase):

    def frames(self, n):
        rng = np.random.default_rng(1)
        return rng.integers(0, 256, size=(n, 19), dtype=np.uint8)

    def decode(self, **kwargs):
        dec = CELPDecoder(**kwargs)
        dec.last_lsf = None
        dec.last_z = np.zeros((10, ))
        dec.ltp_codebook = LtpCodebook(dec.subframe_length)
        return np.concatenate([dec.generate_audio(f) for f in self.frames(10)])

    def test_excitations(self):
        for excitation in CELPDecoder.excitation_codebooks:
            with self.subTest(excitation=excitation):
                audio = self.decode(excitation=excitation)
                self.assertEqual(audio.shape, (10 * 160, ))
                self.assertEqual(audio.dtype, np.int16)

    def test_pame_excitation(self):
        dec = CELPDecoder(excitation='pame')
        # index below the lag: pulses every lag samples starting at the index
        np.testing.assert_array_equal(np.nonzero(dec.excitation(5, 30))[0], [5, 35])
        # next lag/2 indexes: pulses every lag/2 samples
        np.testing.assert_array_equal(np.nonzero(dec.excitation(100 + 3, 100))[0], [3])
        # pulse outside the subframe gives a silent vector
        self.assertFalse(dec.excitation(60, 100).any())
        # remaining indexes select the clipped codebook
        np.testing.assert_array_equal(dec.excitation(200, 100), CELPDecoder.excitation_codebooks['clipped'][50])
        for idx, lag in ((0, 20), (255, 20), (255, 147)):
            self.assertTrue(np.all(np.isfinite(dec.excitation(idx, lag))))

    def test_deterministic(self):
        np.testing.assert_array_equal(self.decode(), self.decode())


@unittest.skipIf(CELPDecoder is None, 'CELP decoding not available (needs the audio extra).')
class TestPacketTypes(unittest.TestCase):

    def packet(self, row, service, data):
        from teletext.packet import Packet
        from teletext.coding import hamming8_encode
        p = Packet()
        p.mrag.magazine = 4
        p.mrag.row = row
        p._array[2] = hamming8_encode(service)
        p._array[3] = hamming8_encode(1)
        p._array[4:] = data
        return p

    def test_text(self):
        text = np.frombuffer(b'Audetel Test000001 Zero fade          ', dtype=np.uint8)
        parity = np.array([bin(n).count('1') & 1 for n in range(128)])
        text = text | ((1 - parity[text]) << 7).astype(np.uint8)
        self.assertTrue(CELPDecoder.is_text(self.packet(30, 0, text)))
        audio = np.random.default_rng(1).integers(0, 256, size=38, dtype=np.uint8)
        self.assertFalse(CELPDecoder.is_text(self.packet(30, 0, audio)))

    def test_text_is_silent(self):
        text = np.full(38, 0x20, dtype=np.uint8)  # 0x20 has odd parity
        dec = CELPDecoder()
        frames = list(dec.decode_packet_stream([self.packet(30, 0, text)]))
        self.assertEqual(len(frames), 2)
        self.assertFalse(np.any(frames[0]))

    def test_two_channel(self):
        audio = np.random.default_rng(1).integers(0, 256, size=38, dtype=np.uint8)
        two = self.packet(31, 5, audio)
        one = self.packet(30, 0, audio)
        self.assertTrue(CELPDecoder.is_two_channel(two))
        self.assertFalse(CELPDecoder.is_two_channel(one))
        self.assertEqual(len(list(CELPDecoder().decode_packet_stream([two]))), 1)
        self.assertEqual(len(list(CELPDecoder().decode_packet_stream([one]))), 2)
        self.assertEqual(len(list(CELPDecoder().decode_packet_stream([two], frame=1))), 1)
