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

# Third party libraries
import numpy as np

# This project
from chern_predictor.data.pauli_matrices import s0, sm, sp, sz


def onsite(site, Xi, U):
    nx, ny = site.pos
    nx, ny = int(nx), int(ny)
    return (Xi[0, 0] + U[nx, ny]) * sz


def hop(site_to, site_from, hopping_cutoff, Xi, Delta):
    nx1, ny1 = site_from.pos
    nx2, ny2 = site_to.pos
    dx = int(nx2 - nx1)
    dy = int(ny2 - ny1)
    if abs(dx) > hopping_cutoff or abs(dy) > hopping_cutoff:
        return 0j * s0
    else:
        return Xi[dx, dy] * sz + Delta[dx, dy] * sp + np.conj(Delta[-dx, -dy]) * sm


def make_square_by_hand(
    edge_length: int, hopping_cutoff: float, xi, delta
) -> np.ndarray:
    """Builds Hamiltonian of a square shaped tight binding system.

    Parameters
    ----------
    edge_length: The system size will be `edge_length` by `edge_length`
    hopping_cutoff: The longest allowed hopping distance

    Returns
    -------
    Hamiltonian matrix
    """

    def hop_manual(to_nx, to_ny, from_nx, from_ny, Xi, Delta, hopping_cutoff):
        nx1, ny1 = from_nx, from_ny
        nx2, ny2 = to_nx, to_ny
        dx = int(nx2 - nx1)
        dy = int(ny2 - ny1)
        if abs(dx) > hopping_cutoff or abs(dy) > hopping_cutoff:
            return 0j * s0
        else:
            return Xi[dx, dy] * sz + Delta[dx, dy] * sp + np.conj(Delta[-dx, -dy]) * sm

    ham = np.zeros(shape=(2 * edge_length**2, 2 * edge_length**2), dtype=np.complex)
    for nx in range(edge_length):
        for ny in range(edge_length):
            site_index = nx * edge_length + ny
            ham[2 * site_index, 2 * site_index] = xi[0, 0]
            ham[2 * site_index + 1, 2 * site_index + 1] = -xi[0, 0]

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
                        myhop = hop_manual(
                            nx_to, ny_to, nx_from, ny_from, xi, delta, hopping_cutoff
                        )
                        site_from_index = nx_from * edge_length + ny_from
                        site_to_index = nx_to * edge_length + ny_to
                        ham[2 * site_to_index, 2 * site_from_index] = myhop[0, 0]
                        ham[2 * site_to_index, 2 * site_from_index + 1] = myhop[0, 1]
                        ham[2 * site_to_index + 1, 2 * site_from_index] = myhop[1, 0]
                        ham[2 * site_to_index + 1, 2 * site_from_index + 1] = myhop[
                            1, 1
                        ]

    return ham
