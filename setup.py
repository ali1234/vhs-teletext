from setuptools import setup

setup(
    name='teletext',
    version='3.0.0',
    author='Alistair Buxton',
    author_email='a.j.buxton@gmail.com',
    url='http://github.com/ali1234/vhs-teletext',
    packages=['teletext', 'teletext.vbi'],
    package_data={
        'teletext.vbi': [
            'data/debruijn.dat',
            'data/parity.dat',
            'data/hamming.dat',
            'data/full.dat'
        ]
    },
    entry_points={
        'console_scripts': [
            'teletext = teletext.cli:teletext',
        ],
        'gui_scripts': [
            'vbiviewer = teletext.vbi.gui.main:main',
        ],
    },
    install_requires=[
        'numpy', 'scipy', 'click', 'tqdm',
        'windows-curses;platform_system=="Windows"'
    ],
    extras_require={
        'spellcheck': ['pyenchant'],
        'CUDA': ['pycuda', 'scikit-cuda'],
        'viewer': ['PyOpenGL'],
        'profiler': ['plop'],
    }
)
