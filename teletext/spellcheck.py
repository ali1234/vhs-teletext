import enchant

from .coding import parity_encode

class SpellChecker(object):

    def __init__(self, language):
        self.dictionary = enchant.Dict(language)

    def check_pair(self, x, y):
        x = x.lower()
        y = y.lower()
        if x == y:
            return 0
        for s in ['eij', 'rstuk', 'yz', 'kgo', 'nm', 'dh']:
            if x in s and y in s:
                return 0
        return 1

    def weighted_hamming(self, a, b):
        return sum([self.check_pair(x, y) for x,y in zip(a, b)])

    def case_match(self, word, src):
        return ''.join([c.lower() if d.islower() else c.upper() for c, d in zip(word, src)])

    def spellcheck(self, displayable):
        words = ''.join(c if c.isalpha() else ' ' for c in displayable.to_ansi(colour=False)).split(' ')

        for n,w in enumerate(words):
            if len(w) > 2 and not self.dictionary.check(w.lower()):
                s = list(filter(lambda x: len(x) == len(w) and self.weighted_hamming(x, w) == 0, self.dictionary.suggest(w.lower())))
                if len(s) > 0:
                    words[n] = self.case_match(s[0], w)

        line = ' '.join(words).encode('ascii')
        for n, b in enumerate(line):
            if b != ord(b' '):
                displayable[n] = parity_encode(b)

    def spellcheck_iter(self, packet_iter):
        for p in packet_iter:
            t = p.type
            if t == 'display':
                self.spellcheck(p.displayable)
            elif t == 'header':
                self.spellcheck(p.header.displayable)
            yield p
