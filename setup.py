from setuptools import setup

setup(
    name='teletext',
    version='3.0',
    author='Alistair Buxton',
    author_email='a.j.buxton@gmail.com',
    url='http://github.com/ali1234/vhs-teletext',
    packages=['teletext', 'teletext.vbi', 'teletext.misc'],
    package_data={'teletext.vbi': ['data/debruijn.dat', 'data/parity.dat', 'data/hamming.dat']},
    scripts=['vbiview'],
    entry_points={
        'console_scripts': [
            'deconvolve = teletext.cli:deconvolve',
            'training = teletext.vbi.training:training',
            't42interactive = teletext.interactive:interactive',
            't42service = teletext.service:service',
            't42pipe = teletext.cli:pipe',
            't42html = teletext.printer:html',
            'vbicat = teletext.vbi.util:vbicat',
        ]
    },
    install_requires=['numpy', 'scipy', 'click', 'pycuda', 'scikit-cuda', 'tqdm', 'pyenchant'],
)
