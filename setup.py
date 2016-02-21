#!/usr/bin/env python

import os

from distutils.core import setup

from teletext.vbi.training import save_pattern

debruijn = os.path.join('teletext', 'vbi', 'data', 'debruijn.dat')
try:
    open(debruijn)
except:
    save_pattern(debruijn)


setup(name='vhs-teletext',
        version='0.1',
        author='Alistair Buxton',
        author_email='a.j.buxton@gmail.com',
        url='http://github.com/ali1234/vhs-teletext',
        packages=['teletext', 'teletext.vbi', 'teletext.t42', 'teletext.misc'],
        package_data={'teletext.vbi': ['data/debruijn.dat']},
        scripts=['deconvolve', 't42pipe', 'vbiview', 'vbicat', 'training'],
        requires=['numpy', 'scipy'],
    )
