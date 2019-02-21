import enchant

from .coding import parity_encode

class SpellChecker(object):

    def __init__(self, language='en_GB'):
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

    def suggest(self, word):
        if len(word) > 2 and not self.dictionary.check(word.lower()):
            for suggestion in self.dictionary.suggest(word.lower()):
                if len(suggestion) == len(word) and self.weighted_hamming(suggestion, word) == 0:
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
