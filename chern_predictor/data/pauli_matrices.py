# Copyright (c) 2020-2023, Paul Baireuther
# All rights reserved.

# Third party libraries
import numpy as np

# Pauli matrices and related matrices
s0 = np.array([[1, 0], [0, 1]])
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])
sp = np.array([[0, 1], [0, 0]])
sm = np.array([[0, 0], [1, 0]])
