from setuptools import setup, Extension
import numpy as np

with open("linestuffup/_version.py", "r") as f:
    exec(f.read())

with open("README.md", "r") as f:
    long_desc = f.read()


setup(
    name = "LineStuffUp",
    version = __version__,
    description =  "3D nonlinear alignment for microscopy and neuroimaging",
    long_description = long_desc,
    long_description_content_type='text/markdown',
    author = 'Max Shinn',
    author_email = 'm.shinn@ucl.ac.uk',
    maintainer = 'Max Shinn',
    maintainer_email = 'm.shinn@ucl.ac.uk',
    license = 'MIT',
    python_requires='>=3.7',
    url='https://github.com/mwshinn/LineStuffUp',
    packages = ['linestuffup'],
    install_requires = ["numpy", "scipy", "napari", "magicgui", "scikit-image", "imageio", "imageio-ffmpeg", "threadpoolctl"],
    classifiers = [
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Multimedia :: Graphics",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Development Status :: 4 - Beta",
    ]
)

