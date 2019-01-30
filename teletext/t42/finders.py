

class Finder(object):

    groups = {
        'c': b'abcdefghijklmnopqrstuvwxyz ',
        'C': b'ABCDEFGHIJKLMNOPQRSTUVWXYZ ',
        'D': b'MTWFS',
        'd': b'mtwfs',
        'A': b'OUEHRA',
        'a': b'ouehra',
        'Y': b'NEDUIT',
        'y': b'neduit',
        'M': b'JFMASOND',
        'm': b'jfmasond',
        'O': b'AEPUCO',
        'o': b'aepuco',
        'N': b'NBRYNLGPTVC',
        'n': b'nbrynlgptvc',
        'Z': b'12345678',
        'T': b'0123456789ABCDEFabcdef',
        'U': b'0123456789ABCDEFabcdef',
        'F': b' 0123',
        'f': b'0123456789',
        'H': b' 012',
        'h': b'0123456789',
        'L': b'012345',
        'l': b'0123456789',
        'S': b'012345',
        's': b'0123456789',
        'e': b'',  # exact match
        ' ': b'\x00\x01\x02\x03\x04\x05\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f ', # whitespace/spacing attributes
    }


    def __init__(self, match1, match2, name, years, channels):
        self.match1 = match1
        self.match2 = match2
        self.name = name
        self.years = years

    def match(self, b):
        s = (b&0x7f).tostring()
        rank = 0
        for n in range(32):
            if self.match2[n] == 'e' and s[n] == self.match1[n]:
                rank += 2
            elif s[n] in self.groups[self.match2[n]]:
                rank += 1
        return rank

    def fixup(self, b):
        for n in range(32):
            if self.match2[n] == 'e':
                b[n] = ord(self.match1[n])
            #elif self.match2[n] == ' ' and chr(b[n]) not in self.groups[' ']:
            #    b[n] = ord(' ')
        return b

    def info(self, b):
        tmp = {}
        for n in range(32):
            tmp[self.match2[n]] = s[n]

Finders = [
Finder("CEEFAX 217 \x09Wed 25 Dec\x03 18:29/53",
       "eeeeeeeZTUee"+"DayeFfeMone"+"eHheLleSs",
       "BBC", (0,1996), ['BBC1', 'BBC2']),

Finder("CEEFAX 1 217 Wed 25 Dec\x0318:29/53",
       "eeeeeeeeeZTUeDayeFfeMoneHhe"+"LleSs",
       "BBC1", (1996,3000), ['BBC1']),

Finder("CEEFAX 2 217 Wed 25 Dec\x0318:29/53",
       "eeeeeeeeeZTUeDayeFfeMone"+"HheLleSs",
       "BBC2", (1996,3000), ['BBC2']),

Finder("Central  217 Wed 25 Dec 18:29:53",
       "eeeeeeeeeZTUeDayeFfeMoneHheLleSs",
       "Central", (0, 3000), ['ITV']),

Finder("\x02   ITV SUBTITLES               ",
       "e"+"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
       "ITV Subs.", (0, 3000), ['ITV']),

Finder("\x01\x1d\x07 DBI STATUS PAGE   \x1c  2059:27",
       "e"+"e"+"e"+"eeeeeeeeeeeeeeeeeeee"+"eeHhLleSs",
       "ITV DBI Stat.", (0, 3000), ['ITV']),

Finder("                         2059:27",
       "eeeeeeeeeeeeeeeeeeeeeeeeeHhLleSs",
       "ITV DBI Blank", (0, 3000), ['ITV']),

Finder("                                ",
       "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
       "Subs Blank", (0, 3000), ['BBC1', 'BBC2', 'ITV', 'C4', 'Five']),

Finder("\x01789 DBI TEST PAGE 789\x07  2059:27",
       "e"+"ZTUeeeeeeeeeeeeeeeZTUe"+"eeHhLleSs",
       "ITV DBI Test.", (0, 3000), ['ITV']),

Finder("\x01   DBI/CH4 - BCAST2  \x09\x07 2059:27",
       "e"+"ZTUeeeeeeeeeeeeeeeeeee"+"e"+"eHhLleSs",
       "C4 DBI Test.", (0, 3000), ['C4']),

Finder(" 500     mon 12  may     2059:27",
       "eZTUeeeeedayeFfeemoneeeeeHhLleSs",
       "Five", (1997,1997), ['Five']),

Finder("\x06   5 text   \x07255 02 May\x031835:21",
       "e"+"eeeeeeeeeeeee"+"ZTUeFfeMone"+"HhLleSs",
       "Five", (1997, 2006), ['Five']),

Finder("Five 500  27 Nov        20:59.27",
       "eeeeeZTUeeFfeMonemoneeeeHheLleSs",
       "Five", (1999,3000), ['Five']),

Finder("SOFTEL D1 SUBTITLE INSERTER     ",
       "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
       "Five Subs.", (0,3000), ['Five']),

Finder("\x04\x1d\x03Teletext\x07 \x1c100 May05\x0318:29:53",
       "e"+"e"+"e"+"eeeeeeeee"+"ee"+"ZTUeMonFfe"+"HheLleSs",
       "Teletext Ltd.", (1993, 3000), ['ITV', 'C4']),

Finder("\x04\x1d\x03Teletext \x03\x1c100 May05\x0318:29:53",
       "e"+"e"+"e"+"eeeeeeeeee"+"e"+"ZTUeMonFfe"+"HheLleSs",
       "Teletext Ltd. (Five)", (1999, 3000), ['Five']),

Finder("\x04\x1d\x03Teletext\x07 \x1c100\x03May05\x0318:29:53",
       "e"+"e"+"e"+"eeeeeeeee"+"ee"+"ZTUe"+"MonFfe"+"HheLleSs",
       "Teletext Ltd. (Five)", (1999, 3000), ['Five']),

Finder("4-Tel 307 Sun 26 May\x03C4\x0718:29:53",
       "eeeeeeZTUeDayeFfeMone"+"eee"+"HheLleSs",
       "4-Tel", (0, 3000), ['C4']),

Finder("PLEASE REFER TO PAGE 100 2001:01",
       "eeeeeeeeeeeeeeeeeeeeeeeeeHhLleSs",
       "Oracle Filler", (0,1992), ['ITV', 'C4']),

Finder("ORACLE 200 Sun27 Dec\x03ITV\x032001:01",
       "eeeeeeeZTUeDayFfeMone"+"eeee"+"HhLleSs",
       "Oracle (ITV)", (0,1992), ['ITV']),

Finder("Teletext on 4 100 Jan25\x0320:01:01",
       "eeeeeeeeeeeeeeZTUeMonFfe"+"HheLleSs",
       "Teletext Ltd. (C4 - Early)", (1993,1993), ['C4']),

Finder("100\x02ARD/ZDF\x07Mo 26.12.88\x0222:00:00",
       "ZTUe"+"eeeeeeee"+"Daeeeeeeeeee"+"HheLleSs",
       "ARD", (0,3000), ['ARD']),

Finder("102 BELTEK              22:07:06",
       "ZTU CCCCCCCCCCCCCCCCCCC HheLleSs",
       "TVR", (0,3000), ['ARD']),

]

