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
from chern_predictor.data.pauli_matrices import sm, sp, sz


# Generate hopping coefficients with the correct symmetries
def generate_ham_coefficients(
    hopping_scale: float,
    hopping_cutoff: float,
    max_relative_pairing_strength: Union[None, float],
    rng: np.random.RandomState,
    epsilon: float = 1e-6,
    verbose: bool = False,
) -> Tuple[np.array, np.array]:
    """
    This function generates two sets of hopping coefficients Xi and Delta. Xi are hoppings
    between sites of the same type, and Delta between particle and hole sites.

    Parameters
    ----------
    hopping_scale: The length scale over which the hopping amplitude decays as
                   ~exp(-distance/hopping_lengthscale)
    hopping_cutoff: The maximal hopping distance, i.e. dx^2 + dy^2 <= hopping_cutoff^2.
    max_relative_pairing_strength: if not None, it enforces that the hopping amplitudes are larger the pairing
                    amplitudes times the max_relative_pairing_strength, according to the two norm
    rng: An instance of the numpy random number generator, already seeded.
    epsilon: a buffer, all hoppings within `hopping_cutoff + epsilon` will be considered
    verbose: If true, some feedback is printed

    Returns
    -------
    xi: Hoppings between two electron sites or two hole sites
    delta: Hoppings between electron and hole sites
    relative_pairing_strength: sum of abs pairings squared divided by sum of hoppings squared
    """

    hopping_cutoff_rounded_up = int(np.ceil(hopping_cutoff))

    # The starting point are three arrays with uniform random values
    weak_pairing = False
    while not weak_pairing:
        xi0 = (
            2.0 * rng.rand(hopping_cutoff_rounded_up + 1, hopping_cutoff_rounded_up + 1)
            - 1.0
        )
        delta0_modulus = rng.rand(
            hopping_cutoff_rounded_up + 1, hopping_cutoff_rounded_up + 1
        )
        delta0_phase = (
            2.0
            * np.pi
            * rng.rand(hopping_cutoff_rounded_up + 1, hopping_cutoff_rounded_up + 1)
        )
        delta0 = delta0_modulus * np.exp(1j * delta0_phase)

        # Next introduce an exponentially decaying hopping amplitude, and an isotropic hopping
        # cutoff
        for nx in range(hopping_cutoff_rounded_up + 1):
            for ny in range(hopping_cutoff_rounded_up + 1):
                if nx > 0 or ny > 0:
                    dist = np.sqrt(nx**2 + ny**2)
                    if dist < hopping_cutoff + epsilon:
                        factor = np.exp(-dist / hopping_scale)
                        xi0[nx, ny] *= factor
                        delta0[nx, ny] *= factor
                    else:
                        xi0[nx, ny] = 0
                        delta0[nx, ny] = 0

        # Next, implement some of the symmetry requirements
        for nx in range(hopping_cutoff_rounded_up + 1):
            for ny in range(nx + 1, hopping_cutoff_rounded_up + 1):
                delta0[nx, ny] = -1j * np.conj(delta0[ny, nx])
        delta0[0, 0] = 0 + 0j

        # Here it is important to preserve the amplitudes
        delta0[0, :] = np.sign(np.real(delta0[0, :])) * np.abs(delta0[0, :])
        delta0[:, 0] = 1j * np.sign(np.imag(delta0[:, 0])) * np.abs(delta0[:, 0])
        for nx in range(hopping_cutoff_rounded_up + 1):
            delta0[nx, nx] = np.abs(delta0[nx, nx]) * np.exp(
                1j * np.angle(np.real(delta0[nx, nx]) - 1j * np.real(delta0[nx, nx]))
            )

        # Finally, construct two new matrices that are large enough to accommodate both positive
        # and negative hoppings and fill it in accordance with the symmetry requirements.
        xi = np.zeros(
            (2 * hopping_cutoff_rounded_up + 1, 2 * hopping_cutoff_rounded_up + 1),
            dtype=float,
        )
        delta = np.zeros(
            (2 * hopping_cutoff_rounded_up + 1, 2 * hopping_cutoff_rounded_up + 1),
            dtype=complex,
        )
        for dx in range(-hopping_cutoff_rounded_up, hopping_cutoff_rounded_up + 1):
            for dy in range(-hopping_cutoff_rounded_up, hopping_cutoff_rounded_up + 1):
                if abs(dx) > abs(dy):
                    t = xi0[abs(dx), abs(dy)]
                    d = delta0[abs(dx), abs(dy)]
                else:
                    t = xi0[abs(dy), abs(dx)]
                    d = -1j * np.conj(delta0[abs(dy), abs(dx)])
                if dx < 0:
                    d = np.conj(d)
                if dy < 0:
                    d = -np.conj(d)
                xi[dx, dy] = np.real(t)
                delta[dx, dy] = d
        # check weak pairing condition
        if max_relative_pairing_strength is None:
            weak_pairing = True
        else:
            relative_pairing_strength = np.sqrt(
                np.sum(np.abs(delta) ** 2) / (np.sum(xi**2) - xi[0, 0] ** 2)
            )
            weak_pairing = relative_pairing_strength <= max_relative_pairing_strength
            if verbose and not weak_pairing:
                print("pairing was too strong")

    return xi, delta, relative_pairing_strength


def hopping_function(dx: int, dy: int, xi: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """This function, for a hopping of distance (dx, dy), given the hamiltonian coefficients Xi
    and Delta, returns the hopping matrix."""
    return xi[dx, dy] * sz + delta[dx, dy] * sp + np.conj(-delta[dx, dy]) * sm
