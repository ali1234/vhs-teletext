import itertools

import enchant

from .coding import parity_encode


class SpellChecker(object):

    common_errors = set(itertools.chain.from_iterable(
        itertools.permutations(s, 2) for s in ('eij', 'rstuk', 'yz', 'kgo', 'nm', 'dh')
    ))

    def __init__(self, language='en_GB'):
        self.dictionary = enchant.Dict(language)

    def check_pair(self, x, y):
        if x == y or (x, y) in self.common_errors:
            return 0
        return 1

    def weighted_hamming(self, a, b):
        return sum(self.check_pair(x, y) for x,y in zip(a, b))

    def case_match(self, word, src):
        return ''.join(c.lower() if d.islower() else c.upper() for c, d in zip(word, src))

    def suggest(self, word):
        if len(word) > 2:
            lcword = word.lower()
            if not self.dictionary.check(lcword):
                for suggestion in self.dictionary.suggest(lcword):
                    if len(suggestion) == len(lcword) and self.weighted_hamming(suggestion.lower(), lcword) == 0:
                        return self.case_match(suggestion, word)
        return word

    def spellcheck(self, displayable):
        words = ''.join(c if c.isalpha() else ' ' for c in displayable.to_ansi(colour=False)).split(' ')

        words = [self.suggest(w) for w in words]

        line = ' '.join(words).encode('ascii')
        for n, b in enumerate(line):
            if b != ord(b' '):
                displayable[n] = parity_encode(b)


def spellcheck_packets(packets, language='en_GB'):

    sc = SpellChecker(language)

    for p in packets:
        t = p.type
        if t == 'display':
            sc.spellcheck(p.displayable)
        elif t == 'header':
            sc.spellcheck(p.header.displayable)
        yield p
