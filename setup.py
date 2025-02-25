from setuptools import find_packages, setup


BUILD_REQ = ["numpy", "scipy"]
INSTALL_REQ = BUILD_REQ
INSTALL_REQ += ["pandas",
                "pycbc",
                "astropy"
                ],

setup(
    name="GSHEWaveform",
    version="0.1",
    description="GSHE Waveform Tools",
    url="https://github.com/Richard-Sti/GSHE",
    author="Richard Stiskalek",
    author_email="richard.stiskalek@protonmail.com",
    license="GPL-3.0",
    packages=find_packages(),
    python_requires=">=3.8",
    build_requires=BUILD_REQ,
    setup_requires=BUILD_REQ,
    install_requires=INSTALL_REQ,
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9"]
)
