from setuptools import setup

setup(
    name='teletext',
    version='3.1.99',
    author='Alistair Buxton',
    author_email='a.j.buxton@gmail.com',
    url='http://github.com/ali1234/vhs-teletext',
    packages=['teletext', 'teletext.vbi', 'teletext.cli', 'teletext.gui'],
    package_data={
        'teletext.vbi': [
            'data/debruijn.dat',
            'data/vhs/parity.dat',
            'data/vhs/hamming.dat',
            'data/vhs/full.dat',
            'data/betamax/parity.dat',
            'data/betamax/hamming.dat',
            'data/betamax/full.dat'
        ],
        'teletext.gui': [
            'decoder.qml',
            'editor.ui',
        ]
    },
    entry_points={
        'console_scripts': [
            'teletext = teletext.cli.teletext:teletext',
        ],
        'gui_scripts': [
            'ttviewer = teletext.gui.editor:main',
        ],
    },
    install_requires=[
        'numpy', 'scipy', 'click', 'tqdm',  'pyzmq', 'watchdog',
        'windows-curses;platform_system=="Windows"',
    ],
    extras_require={
        'spellcheck': ['pyenchant'],
        'CUDA': ['pycuda', 'scikit-cuda'],
        'viewer': ['PyOpenGL'],
        'profiler': ['plop'],
        'qt': ['PyQt5', 'matplotlib'],
    }
)
