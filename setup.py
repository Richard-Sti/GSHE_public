from setuptools import setup

setup(
    name='GR_spin_hall',
    version='0.1',
    description='Numerical gravitational spin Hall equations',
    url='https://github.com/Richard-Sti/GR_spin_hall',
    author='Richard Stiskalek',
    author_email='richard.stiskalek@protonmail.com',
    license='MIT License',
    packages=['GR_spin_hall'],
    install_requires=['scipy',
                      'numpy',
                      'sympy',
                      'matplotlib'],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9']
)
