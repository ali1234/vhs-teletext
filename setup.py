from setuptools import setup

setup(
    name='vhs-teletext',
    version='3.0',
    author='Alistair Buxton',
    author_email='a.j.buxton@gmail.com',
    url='http://github.com/ali1234/vhs-teletext',
    packages=['teletext', 'teletext.vbi', 'teletext.t42', 'teletext.misc'],
    package_data={'teletext.vbi': ['data/debruijn.dat', 'data/parity.dat', 'data/hamming.dat']},
    scripts=['t42pipe', 't42interactive', 't42service', 'vbiview', 'vbicat'],
    entry_points={
        'console_scripts': [
            'deconvolve = teletext.vbi.deconvolve:deconvolve',
            'training = teletext.vbi.training:training',
        ]
    },
    install_requires=['numpy', 'scipy', 'click', 'pycuda', 'scikit-cuda', 'tqdm', 'pyenchant'],
)
