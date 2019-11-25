#!/usr/bin/env python
# coding: utf-8

# # Quantum CSD Compiling Intro
#
# The purpose of this notebook is to give a quick introduction to Qubiter's
# CSD quantum compiler capabilities.
#
# By a quantum complier, we mean
# a computer program that takes as input an arbitrary unitary matrix $U$ of dimension $N=2^n$
# and returns a SEO (sequence of elementary operations, in this case CNOTs and single qubit
# rotations) that equals $U$. There are various kinds of quantum
# compilers. Suppose $U$ is of the form $U=e^{-itH}$, where $t$ is time and $H$ is
# the Hamiltonian operator. If $H$ has a form which is known a priori, a situation
# that is common in Physics and Chemistry, then a popular way of expanding $U$
# is by using the Trotter-Suzuki approximation. If the form of $H$ is not
# known a priori as is common in Artificial Intelligence, then
# we recommend using the CS (Cosine-Sine) decomposition of Linear Algebra.
# Both methods are already implemented in Qubiter, but this notebook is about
# the CSD.
#
# Doing CSD quantum compiling with Qubiter requires using the classes in the quantum_CSD_compiler
# folder, which will only work properly if you install, besides all the Qubiter
# Python files and a Python distro that includes numpy and scipy (for example, Anaconda),
# some binary libraries prepared by Artiste-q.net which include
# a Python wrapper for a LAPACK subroutine
# called cuncsd.f that performs CSDs. How to install those binary libraries
# is explained elsewhere in this site. Henceforth, we will assume
# all the necessary files have been installed on your computer if you want to redo the calculations.
#
# The quantum_CSD_compiler folder includes a pdf called csd-intro.pdf that gives
# an introduction to the CSD.
#
# Some external references:
#
#
# 1. R.R. Tucci, A Rudimentary Quantum Compiler(2cnd Ed.)
#     https://arxiv.org/abs/quant-ph/9902062
#
# 2. Qubiter 1.11, a C++ program whose first version was released together
#     with Ref.1 above. Qubiter 1.11 is included in the
#     quantum_CSD_compiler/LEGACY folder of this newer, pythonic version of Qubiter
#
# 3. R.R. Tucci, Quantum Fast Fourier Transform Viewed as a Special Case of Recursive Application of Cosine-Sine Decomposition, https://arxiv.org/abs/quant-ph/0411097
#
#

# Qubiter applies CSD recursively
# to build a tree of node matrices. The product of those node matrices,
# if read in the correct order, is equal to the input matrix $U$.
#
# As an example, let us use for $U$ a 3 qubit quantum Fourier matrix.
# We can create an object of class Tree with $U$
# as input as follows


import os
import sys
import argparse


import numpy as np
import math
from scipy.stats import unitary_group
import pandas as pd

# os.chdir("../../")
sys.path.insert(0, os.getcwd())

# Qubiter libs
from qubiter.FouSEO_writer import *
from qubiter.CGateSEO_writer import *
from qubiter.CGateExpander import *
from qubiter.quantum_CSD_compiler.MultiplexorExpander import *
from qubiter.quantum_CSD_compiler.Tree import *
from qubiter.quantum_CSD_compiler.DiagUnitarySEO_writer import *
from qubiter.quantum_CSD_compiler.MultiplexorSEO_writer import *
from qubiter.quantum_CSD_compiler.DiagUnitaryExpander import *
from qubiter.device_specific.Qubiter_to_AnyQasm import *

import qubiter.device_specific.chip_couplings_ibm as ibm
from qubiter.device_specific.Qubiter_to_IBMqasm import Qubiter_to_IBMqasm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configuration for decomposition', add_help=True)
    parser.add_argument('--num_qubits', '-n', type=int, default=2,  help="Number of qubits")
    parser.add_argument('--qasm_output', '-q', type=store_true, help="Conversion to QASM format")
    args = parser.parse_args()
    print('Current directory: ', os.getcwd())

    num_bits = args.num_qubits
    print("Initializing random unitary U ..., num_qubits: ", num_bits)
    init_unitary_mat = unitary_group.rvs(2 ** num_bits)
    emb = CktEmbedder(num_bits, num_bits)
    file_prefix = "decomposition_demo"
    t = Tree(True, file_prefix, emb, init_unitary_mat, verbose=False)
    t.close_files()

    # The above code automatically creates an expansion of $U$
    # into DIAG and MP_Y lines. Next we print the Picture file that was created.


    file = './qubiter/quantum_CSD_compiler/'+file_prefix + "_2_ZLpic.txt"
    df = pd.read_csv(file, delim_whitespace=True, header=None)


    style = "exact"
    xer = MultiplexorExpander(file_prefix, num_bits, style, verbose=False)
    xer = DiagUnitaryExpander(file_prefix + "_X1", num_bits, style)
    xer = MultiplexorExpander(file_prefix + "_X2", num_bits, style, verbose=False)
    exp = CGateExpander(file_prefix + "_X3", num_bits)

    file = "./qubiter/quantum_CSD_compiler/" + file_prefix + "_X4_2_ZLpic.txt"
    log = open(file)
    for line in log:
        print(line)


    # Let us also print the corresponding English file that was created.

    file = "./qubiter/quantum_CSD_compiler/" + file_prefix + "_X4_2_eng.txt"
    log = open(file)
    for line in log:
        print(line)

    if args.qasm_output:
        aqasm_name = "IBMqasm"
        #  We can adopt qubit couplings for a specific hardware architecture (e.g. IBM5YorktownTenerife)
        c_to_tars = None  # ibm.ibmq5YorktownTenerife_c_to_tars
        Qubiter_to_IBMqasm(
            file_prefix + "_X4",
            num_bits,
            aqasm_name=aqasm_name,
            c_to_tars=c_to_tars,
            write_qubiter_files=True,
        )
