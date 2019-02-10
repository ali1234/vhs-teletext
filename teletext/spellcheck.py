import enchant

from .coding import parity_encode
from .printer import PrinterANSI

d = enchant.Dict('en_GB')

freecombos = {
    {'e', 'i', 'j'},
    {'r', 's', 't', 'u', 'k'},
    {'y', 'z'},
    {'k', 'g', 'o'},
    {'n', 'm'},
    {'d', 'h'},
}


def check_pair(x, y):
    x = x.lower()
    y = y.lower()
    if x == y:
        return 0
    for s in freecombos:
        if x in s and y in s:
            return 0
    return 1


def weighted_hamming(a, b):
    return sum([check_pair(x, y) for x,y in zip(a, b)])


def case_match(word, src):
    return ''.join([c.lower() if d.islower() else c.upper() for c, d in zip(word, src)])


def spellcheck(packet):
    array = None
    t = packet.type
    if t == 'display':
        array = packet.displayable
    elif t == 'header':
        array = packet.header.displayable

    words = ''.join(c if c.isalpha() else ' ' for c in str(PrinterANSI(array, False))).split(' ')

    for n,w in enumerate(words):
        if len(w) > 2 and not d.check(w.lower()):
            s = list(filter(lambda x: len(x) == len(w) and weighted_hamming(x, w) == 0, d.suggest(w.lower())))
            if len(s) > 0:
                words[n] = case_match(s[0], w)

    line = ' '.join(words).encode('ascii')
    for n, b in enumerate(line):
        if b != ord(b' '):
            array[n] = parity_encode(b)

