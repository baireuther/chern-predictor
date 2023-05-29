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


# Third party libraries
import numpy as np
from scipy.integrate import dblquad, fixed_quad


def calc_chern_number(
    Xi: np.ndarray,
    Delta: np.ndarray,
    conv_margin: float = 1e-2,
    n_quad_min: int = 100,
    n_quad_max: int = 1600,
    verbose: bool = False,
) -> float:
    """Calculation of the Chern number using the Hamiltonian in momentum space without disorder.

    Parameters
    ----------
    Xi: Hoppings between two electron sites or two hole sites.
    Delta: Hoppings between electron and hole sites.
    conv_margin: A float, characterizing when the calculation is converged.
    n_quad_min: Minimal order of quadrature integration.
    n_quad_max: Maximal order of quadrature integration.
    verbose: If 'True', some feedback will be printed.

    Returns
    -------
    Chern number
    """

    def integrand1(kx, ky):
        if type(kx) == float:
            kx = np.array([kx])
        if type(ky) == float:
            ky = np.array([ky])
        res = integrand(kx, ky, Xi, Delta)
        return res

    def integrand2(ky):
        res, _ = fixed_quad(integrand1, -np.pi, np.pi, args=(ky,), n=n_quad)
        return res

    n_quad = n_quad_min // 2
    chern_number = None
    converged = False
    conv_hist = []
    while n_quad <= n_quad_max:

        chern_number_prev_step = chern_number
        chern_number, _ = fixed_quad(integrand2, -np.pi, np.pi, n=n_quad)
        conv_hist.append(chern_number)

        if (
            # Criterion 1: Converges to a value
            chern_number_prev_step
            and np.abs(chern_number - chern_number_prev_step) < conv_margin
        ):
            # Criterion 2: Converges to an integer
            if np.abs(chern_number - np.rint(chern_number)) < conv_margin:
                converged = True
                break

        if verbose:
            print(
                f"Order of quadrature integration = {n_quad}, value of Chern number = {chern_number}"
            )

        n_quad *= 2

    if not converged:
        # Fall back to the slower adaptive integration
        chern_number, _ = dblquad(
            integrand1, a=-np.pi, b=np.pi, gfun=-np.pi, hfun=np.pi, epsrel=1e-3
        )
        print(
            f"WARNING: "
            f"The calculation of the Chern number using quadrature integration did not converge. "
            f"Falling back to adaptive integration. The intermediate estimates where {conv_hist}. "
            f"Using adaptive integration {chern_number} was obtained."
        )

    return int(np.rint(chern_number))


def integrand(
    kx: np.ndarray, ky: np.ndarray, Xi: np.ndarray, Delta: np.ndarray
) -> np.array:
    """This function implements the formula: h * (dh/dkx x dh/dky) / (4 * pi * |h|^3), where
    h(k) = (Re[Delta(k)], -Im[Delta(k)], Xi(k))."""

    kx = kx[None, :]
    ky = ky[:, None]
    N = Xi.shape[0] // 2

    # First calculate h(k)
    hx = np.sum(
        np.array(
            [
                np.real(Delta[nx, ny]) * np.cos(kx * nx + ky * ny)
                - np.imag(Delta[nx, ny]) * np.sin(kx * nx + ky * ny)
                for nx in range(-N, N + 1)
                for ny in range(-N, N + 1)
            ]
        ),
        axis=0,
    )
    hy = np.sum(
        [
            -np.real(Delta[nx, ny]) * np.sin(kx * nx + ky * ny)
            - np.imag(Delta[nx, ny]) * np.cos(kx * nx + ky * ny)
            for nx in range(-N, N + 1)
            for ny in range(-N, N + 1)
        ],
        axis=0,
    )
    hz = np.sum(
        [
            Xi[nx, ny] * np.cos(kx * nx + ky * ny)
            for nx in range(-N, N + 1)
            for ny in range(-N, N + 1)
        ],
        axis=0,
    )
    h = np.array([hx, hy, hz])

    # Then calculate dh(k)/dkx
    hxdkx = np.sum(
        [
            -np.real(Delta[nx, ny]) * np.sin(kx * nx + ky * ny) * nx
            - np.imag(Delta[nx, ny]) * np.cos(kx * nx + ky * ny) * nx
            for nx in range(-N, N + 1)
            for ny in range(-N, N + 1)
        ],
        axis=0,
    )
    hydkx = np.sum(
        [
            -np.real(Delta[nx, ny]) * np.cos(kx * nx + ky * ny) * nx
            + np.imag(Delta[nx, ny]) * np.sin(kx * nx + ky * ny) * nx
            for nx in range(-N, N + 1)
            for ny in range(-N, N + 1)
        ],
        axis=0,
    )
    hzdkx = np.sum(
        [
            -Xi[nx, ny] * np.sin(kx * nx + ky * ny) * nx
            for nx in range(-N, N + 1)
            for ny in range(-N, N + 1)
        ],
        axis=0,
    )
    dhdkx = np.array([hxdkx, hydkx, hzdkx])

    # Then calculate dh(k)/dky
    hxdky = np.sum(
        [
            -np.real(Delta[nx, ny]) * np.sin(kx * nx + ky * ny) * ny
            - np.imag(Delta[nx, ny]) * np.cos(kx * nx + ky * ny) * ny
            for nx in range(-N, N + 1)
            for ny in range(-N, N + 1)
        ],
        axis=0,
    )
    hydky = np.sum(
        [
            -np.real(Delta[nx, ny]) * np.cos(kx * nx + ky * ny) * ny
            + np.imag(Delta[nx, ny]) * np.sin(kx * nx + ky * ny) * ny
            for nx in range(-N, N + 1)
            for ny in range(-N, N + 1)
        ],
        axis=0,
    )
    hzdky = np.sum(
        [
            -Xi[nx, ny] * np.sin(kx * nx + ky * ny) * ny
            for nx in range(-N, N + 1)
            for ny in range(-N, N + 1)
        ],
        axis=0,
    )
    dhdky = np.array([hxdky, hydky, hzdky])

    # And finally, put it all together
    h_norm = np.sqrt(np.sum(np.multiply(h, h), axis=0))[None, :, :]
    ncross = np.cross(dhdkx, dhdky, axisa=0, axisb=0, axisc=0)

    return np.sum(np.multiply(h / h_norm**3, ncross), axis=0) / (4 * np.pi)
