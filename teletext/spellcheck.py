import enchant

from teletext.packet import DisplayPacket, HeaderPacket, PrinterANSI

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
    if type(packet) == DisplayPacket or type(packet) == HeaderPacket:
        words = str(PrinterANSI(packet.displayable, False))
        words = ''.join([c if c.isalnum() else ' ' for c in words])
        words = words.split(' ')

        for n,w in enumerate(words):
            if len(w) > 2 and not d.check(w.lower()):
                s = filter(lambda x: len(x) == len(w) and weighted_hamming(x, w) == 0, d.suggest(w.lower()))
                if len(s) > 0:
                    words[n] = case_match(s[0], w)

        words = ' '.join(words)
        for n,c in enumerate(words):
            if c != ' ':
                packet.displayable[n] = ord(c)

