# Copyright (c) 2020-2023, Paul Baireuther
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU Affero General Public License as published by the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License along with this
# program. If not, see <https://www.gnu.org/licenses/>


# Python standard library
from math import ceil
from typing import Tuple, Union

# Third party libraries
import numpy as np
from scipy.sparse import coo_array
from scipy.sparse.linalg import eigsh

# This project
from chern_predictor.data.pauli_matrices import s0, sm, sp, sz


def hopping(
    to_nx: int,
    to_ny: int,
    from_nx: int,
    from_ny: int,
    Xi: np.ndarray,
    Delta: np.ndarray,
    hopping_cutoff: float,
) -> np.ndarray:
    dx = int(to_nx - from_nx)
    dy = int(to_ny - from_ny)
    if dx**2 + dy**2 > hopping_cutoff**2 + 1e-12:
        return 0j * s0
    else:
        return Xi[dx, dy] * sz + Delta[dx, dy] * sp + np.conj(Delta[-dx, -dy]) * sm


def make_square(
    edge_length: int,
    hopping_cutoff: float,
    xi: np.ndarray,
    delta: np.ndarray,
    sparse: bool = True,
) -> Union[coo_array, np.ndarray]:
    """Construct a tight binding Hamiltonian of a square lattice with hard wall boundary
    conditions.

    Parameters
    ----------
    edge_length: The system size will be `edge_length` by `edge_length`
    hopping_cutoff: The longest allowed hopping distance

    Returns
    -------
    Hamiltonian matrix
    """

    rows, cols, vals = [], [], []
    for nx in range(edge_length):
        for ny in range(edge_length):
            site_index = nx * edge_length + ny
            rows.extend([2 * site_index, 2 * site_index + 1])
            cols.extend([2 * site_index, 2 * site_index + 1])
            vals.extend([xi[0, 0], -xi[0, 0]])

    hopping_cutoff_rounded_up = int(ceil(hopping_cutoff))
    for nx_from in range(edge_length):
        nx_to_max = min(nx_from + hopping_cutoff_rounded_up + 1, edge_length)
        for ny_from in range(edge_length):
            ny_to_max = min(ny_from + hopping_cutoff_rounded_up + 1, edge_length)
            for nx_to in range(max(nx_from - hopping_cutoff_rounded_up, 0), nx_to_max):
                for ny_to in range(
                    max(ny_from - hopping_cutoff_rounded_up, 0), ny_to_max
                ):
                    if nx_from != nx_to or ny_from != ny_to:
                        hop = hopping(
                            nx_to, ny_to, nx_from, ny_from, xi, delta, hopping_cutoff
                        )
                        site_from_index = nx_from * edge_length + ny_from
                        site_to_index = nx_to * edge_length + ny_to
                        dx = int(nx_to - nx_from)
                        dy = int(ny_to - ny_from)
                        if dx**2 + dy**2 > hopping_cutoff**2 + 1e-12:
                            pass
                        else:
                            rows.extend(
                                [
                                    2 * site_to_index,
                                    2 * site_to_index,
                                    2 * site_to_index + 1,
                                    2 * site_to_index + 1,
                                ]
                            )
                            cols.extend(
                                [
                                    2 * site_from_index,
                                    2 * site_from_index + 1,
                                    2 * site_from_index,
                                    2 * site_from_index + 1,
                                ]
                            )
                            vals.extend([hop[0, 0], hop[0, 1], hop[1, 0], hop[1, 1]])

    vals = np.array(vals)
    rows = np.array(rows)
    cols = np.array(cols)
    ham = coo_array(
        (vals, (rows, cols)),
        shape=(2 * edge_length**2, 2 * edge_length**2),
        dtype=complex,
    )

    if sparse:
        return ham
    else:
        return ham.todense()


def calc_ldos(ham: np.ndarray, en_lim: float) -> np.ndarray:
    """This function calculates the local density of states

    Parameters
    ----------
    ham: The sparse representation of the Hamiltonians
    en_lim: The maximal energy (absolute value) of energies to be considered

    Returns
    -------
    local density of states
    """

    evals, evecsT = np.linalg.eigh(ham)
    evecs = evecsT.T

    N, N = ham.shape

    ldos = np.zeros(N)
    for i, val in enumerate(evals):
        if abs(val) < en_lim:
            ldos += np.abs(evecs[i]) ** 2
    ldos = np.sum(ldos.reshape(N // 2, 2), axis=1)

    edge_length = int(np.sqrt(N // 2))
    return ldos.reshape(edge_length, edge_length)


def calc_ldos_sparse(
    ham: coo_array,
    en_lim: float,
    en_resolved: bool = False,
    kmin: int = 6,
    seed: int = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """This function calculates the local density of states.

    Parameters
    ----------
    ham: The sparse representation of the Hamiltonians
    en_lim: The maximal energy (absolute value) of energies to be considered
    en_resolved: If True, the LDOS is returned as a stack of weights corresponding to the
    eigenvalues
    kmin: minimal number of eigenvalues to be calculated
    verbose: if True, some feedback is printed

    Returns
    -------
    eigenvalues, local density of states
    """

    rng = np.random.RandomState(seed)

    n_ham, _ = ham.shape
    n_sites = int(n_ham // 2)

    evals = [0]
    while np.max(np.abs(evals)) <= en_lim:
        tolerance = 1e-6
        success = False
        while not success:
            try:
                evals, evecsT = eigsh(
                    ham, k=kmin, sigma=0, v0=rng.rand(n_ham), tol=tolerance
                )
                success = True
            except:
                tolerance *= 2
                print(
                    "evals did not converge, increasing tolerance to {}".format(
                        tolerance
                    )
                )

        if verbose:
            print(kmin, end=", ")
        kmin *= 2
    evecs = evecsT.T

    sorted_args = np.argsort(evals)
    evals = evals[sorted_args]
    evecs = evecs[sorted_args]

    # select all eigenvalues with modulus smaller than the energy limit
    condition = np.abs(evals) <= en_lim
    evals = evals[condition]
    evecs = evecs[condition]

    weights = np.sum(np.abs(evecs.reshape(-1, n_sites, 2)) ** 2, axis=2)
    if en_resolved:
        return evals, weights
    else:
        if len(evals) > 0:
            ldos = np.mean(weights, axis=0)
        else:
            ldos = np.zeros(n_sites)
        return evals, ldos
