from setuptools import setup

setup(
    name='teletext',
    version='3.0.0',
    author='Alistair Buxton',
    author_email='a.j.buxton@gmail.com',
    url='http://github.com/ali1234/vhs-teletext',
    packages=['teletext', 'teletext.vbi'],
    package_data={'teletext.vbi': ['data/debruijn.dat', 'data/parity.dat', 'data/hamming.dat']},
    entry_points={
        'console_scripts': [
            'teletext = teletext.cli:teletext',
            'training = teletext.vbi.training:training',
            't42service = teletext.service:service',
            't42html = teletext.printer:html',
        ]
    },
    install_requires=['numpy', 'scipy', 'click', 'tqdm', 'windows-curses;platform_system=="Windows"'],
    extras_require={
        'spellcheck': ['pyenchant'],
        'CUDA': ['pycuda', 'scikit-cuda'],
        'viewer': ['PyOpenGL'],
    }
)
