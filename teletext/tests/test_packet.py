import itertools
import unittest

from teletext.packet import *


class TestPacket(unittest.TestCase):

    packet = Packet()

    def setUp(self):
        pass

    def test_type(self):
        self.packet.mrag.row = 0
        self.assertEqual(self.packet.type, 'header')
        self.packet.mrag.row = 1
        self.assertEqual(self.packet.type, 'display')
        self.packet.mrag.row = 27
        self.assertEqual(self.packet.type, 'fastext')
        self.packet.mrag.row = 28
        self.assertEqual(self.packet.type, 'page enhancement')
        self.packet.mrag.row = 29
        self.assertEqual(self.packet.type, 'magazine enhancement')
        self.packet.mrag.row = 31
        self.assertEqual(self.packet.type, 'independent data')
        self.packet.mrag.row = 30
        self.assertEqual(self.packet.type, 'unknown')
        self.packet.mrag.magazine = 8
        self.assertEqual(self.packet.type, 'broadcast')
