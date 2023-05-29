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
from typing import Tuple, Union

# Third party libraries
import numpy as np

# This project
from chern_predictor.data.hamiltonian_coefficients import hopping_function


def energy(
    kx: np.ndarray, ky: np.ndarray, Xi: np.ndarray, Delta: np.ndarray
) -> np.ndarray:
    """This function calculates the energies on a (kx, ky) grid

    Parameters
    ----------
    kx: The kx values
    ky: The ky values
    Xi: The hoppings
    Delta: The pairings

    Returns
    -------
    Energies on the (kx, ky) grid.
    """

    kx = kx[:, None]
    ky = ky[None, :]

    return np.sqrt(Xi_k(kx, ky, Xi, Delta) ** 2 + abs(Delta_k(kx, ky, Xi, Delta)) ** 2)


# sub functions
def Xi_k(
    kx: Union[float, np.ndarray],
    ky: Union[float, np.ndarray],
    Xi: np.ndarray,
    Delta: np.ndarray,
) -> Union[float, np.ndarray]:
    N = Xi.shape[0] // 2
    tot = 0 + 0j
    for nx in range(-N, N + 1):
        for ny in range(-N, N + 1):
            tot += hopping_function(nx, ny, Xi, Delta)[0, 0] * np.exp(
                1j * nx * kx + 1j * ny * ky
            )
    assert np.sum(np.abs(tot - np.real(tot))) < 1e-9
    return np.real(tot)


def Delta_k(
    kx: Union[float, np.ndarray],
    ky: Union[float, np.ndarray],
    Xi: np.ndarray,
    Delta: np.ndarray,
) -> Union[float, np.ndarray]:
    N = Delta.shape[0] // 2
    tot = 0 + 0j
    for nx in range(-N, N + 1):
        for ny in range(-N, N + 1):
            tot += hopping_function(nx, ny, Xi, Delta)[0, 1] * np.exp(
                1j * nx * kx + 1j * ny * ky
            )
    return tot


def is_converged(val: float, val_previous: float, margin: float) -> bool:
    """This function checks if a sequential optimization has converged.

    Parameters
    ----------
    val: current value
    val_previous: previous value
    margin: convergence margin

    Returns
    -------
    True, if the convergence criterion has been fullfilled.
    """
    if abs(val) < 1e-15:
        return True
    return abs(val - val_previous) / abs(val) < margin


def calc_gap(
    xi: np.ndarray,
    delta: np.ndarray,
    seed: int,
    min_absolute_gap: float = 0.0,
    conv_margin: float = 0.01,
    n_ks_min: int = 20,
    n_ks_max: int = 1280,
    verbose: bool = False,
) -> Tuple[float, float]:
    """This function calculates the gap in the spectrum around zero

    Parameters
    ----------
    xi: Hoppings
    delta: Pairings
    seed: Seeds the random k-value selection.
    min_absolute_gap: The minimal required gap in absolute units; For Hamiltonians with gap below
                      the minimal gap, the calculation is not completed.
    conv_margin: If the delta(gap)/gap is smaller than this between two adjacent steps,
                 the calculations is considered finished
    n_ks_min: a lower limit how many kx, ky values to consider
    n_ks_max: an upper limit how many kx, ky values to consider
    verbose: If true, some feedback is printed

    Returns
    -------
    The gap in the spectrum once relative to the bandwidth, once absolute
    """

    rng = np.random.RandomState(seed)

    n_ks = n_ks_min
    gap_relative_previous = 1e9
    while n_ks <= n_ks_max:
        kxs, kys = (
            2 * np.pi * rng.rand(n_ks) - np.pi,
            2 * np.pi * rng.rand(n_ks) - np.pi,
        )
        energies = energy(kxs, kys, xi, delta)
        gap_absolute = 2 * np.min(np.abs(energies))
        gap_relative = gap_absolute / (2.0 * np.max(energies))

        if gap_absolute < min_absolute_gap:
            break
        else:
            if is_converged(gap_relative, gap_relative_previous, conv_margin):
                if verbose:
                    print(
                        "n_ks = {}, relative gap = {}, absolute gap = {}".format(
                            n_ks, gap_relative, gap_absolute
                        )
                    )
                    print("Calculation has converged.")
                break
        gap_relative_previous = gap_relative
        if verbose:
            print(
                "n_ks = {}, relative gap = {}, absolute gap = {}".format(
                    n_ks, gap_relative, gap_absolute
                )
            )
        n_ks *= 2
    return gap_relative, gap_absolute
