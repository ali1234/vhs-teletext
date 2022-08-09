import unittest

from teletext.spellcheck import *
from teletext.elements import Displayable


class TestSpellCheck(unittest.TestCase):

    def setUp(self):
        self.sc = SpellChecker(language='en_GB')

    def test_case_match(self):
        src = 'AaAaA'
        word = 'bbbbb'
        self.assertEqual(self.sc.case_match(word, src), 'BbBbB')

    def test_suggest(self):
        # correctly spelled word should be unchanged
        self.assertEqual(self.sc.suggest('hello'), 'hello')
        # incorrectly spelled word with known substitutions should be fixed
        self.assertEqual(self.sc.suggest('dello'), 'hello')

    def test_spellcheck(self):
        d = Displayable((17,), 'dello dello dello'.encode('ascii'))
        self.sc.spellcheck(d)
        self.assertEqual(d.to_ansi(colour=False), 'hello hello hello')
