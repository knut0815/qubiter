# Qubiter at GitHub

## Tutorial

The following Jupyter notebook is a
good introduction to Qubiter's basic features. Other notebooks
in Qubiter's jupyter_notebook folder
discuss more advanced features:

* https://nbviewer.jupyter.org/github/artiste-qb-net/qubiter/blob/master/qubiter/jupyter_notebooks/Say_Hello_World_With_Qubiter.ipynb

The following automatically generated notebook summarizes all other notebooks:

* https://nbviewer.jupyter.org/github/artiste-qb-net/qubiter/blob/master/qubiter/jupyter_notebooks/SUMMARY.ipynb

See also, at the readthedocs.org
website, the beautiful documentation generated automatically by the Sphinx
software based on Qubiter's docstrings and code:

* https://qubiter.readthedocs.io/en/latest/

## Installation

In order to install this forked version of Qubiter follow these steps:

1) Clone Qubiter source files from github:
git clone https://github.com/yourball/qubiter.git

2) Change to the source directory and run Qubiter installation:

``pip install .``

3) Install PYCSD package (Python wrapper of cosine-sine decomposition originally written for fortran77).
To do so clone source files: git clone https://github.com/nice-tools/pycsd.git
And run from the source folder:

``pip install .``

4) Install gfortran since pycsd package heavily relies on fortran77 compiler.
On Ubuntu run:

``sudo apt-get install gfortran``

On Mac go to https://www.webmo.net/support/fortran_osx.html and follow steps for g77 compiler installation.


## What is Qubiter?

The Qubiter project aims to  provide eventually a full suite of tools, written mostly in Python, for designing and simulating quantum circuits on classical computers. (So it will address only the needs of gate model, not annealer, quantum computer engineers). We or others could start a similar project for annealers.

An earlier C++ computer program also called Qubiter (see http://www.ar-tiste.com/qubiter.html), written by Robert R. Tucci, did only quantum compiling. This newer project includes a quantum CSD compiler similar to the earlier Qubiter, based on the (Cosine-Sine) CS Decompostion of Linear Algebra, but written in Python. But this new project also includes much more than that.

We've included classes for reading and writing quantum circuit files. Also for expanding circuits with gates that have multiple controls into circuits with only CNOTs and single qubit rotation gates. Also for embedding a circuit inside a larger one. And, last but not least, we've included a simulator.

The simulator hasn't been bench-marked but should be pretty fast, because it relies on Numpy, which is a Python wrapper for C code.

Besides being amply documented with docstrings, each class has a main method at the end giving examples of its usage (and testing it). Plus we've included a large and ever increasing collection of Jupyter notebooks that teach some physics and how to use Qubiter at the same time.

The quantum circuits are saved as text files, which allows easy exchange between QC engineers.

The quantum circuits are draw in ASCII (not in postscript or in a proprietary format). We hope we can convince you that ASCII drawings of quantum circuits are surprisingly clear, expressive, and convenient, really all you need, plus, unlike other formats, they are super easy to edit. Using other formats might require you to master difficult subjects like postscript in order to write/edit circuit files. This is totally unnecessary!

Quantum Fog at GitHub (see https://github.com/artiste-qb-net/quantum-fog) is a twin project started by the same people. We hope that eventually Quantum Fog will call Qubiter to perform some tasks, like quantum compiling and simulating.

All of Qubiter at GitHub except for the contents of the quantum_CSD_compiler folder is licensed under the BSD license (3 clause version) with an added clause at the end, taken almost verbatim from the Apache 2.0 license, granting additional Patent rights. See `LICENSE.md`.

The contents of Qubiter's quantum_CSD_compiler folder are licensed under the GPLv2 (Linux) license.

## Contributors

(Alphabetical Order)
* Dekant, Henning
* Tregillus, Henry
* Tucci, Robert
* Yin, Tao
