from typing import List

import numpy as np
import matplotlib.pyplot as plt


LATTICE_SIZE = 50
#     e0   e1   e2    e3    e4   e5    e6    e7    e8
E = [
    [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0],
    [0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0],
]
W = [4 / 9] + [1 / 9] * 4 + [1 / 36] * 4
DX = 1.0
DT = 1.0
C = DX / DT
TAU = 10.0


def si(i: int, ux: List[np.ndarray], uy: List[np.ndarray]) -> np.ndarray:
    udotu = ux * ux + uy * uy
    edotu = E[0][i] * ux + E[1][i] * uy
    return (
        3.0 * edotu / C
        + 9.0 * edotu ** 2 / (2.0 * C ** 2)
        - 3.0 * udotu / (2.0 * C ** 2)
    )


def calculate_rho(ns) -> np.ndarray:
    r = np.zeros(ns[0].shape)
    for n in ns:
        r += n
    return r


def calculate_ui(ns, rho: np.ndarray, axis: int = 0) -> np.ndarray:
    r = np.zeros(ns[0].shape)
    for i, n in enumerate(ns):
        r += C * E[axis][i] * n
    return r / rho


def calculate_nieq(i: int, ux: List[np.ndarray], uy: List[np.ndarray], rho: np.ndarray):
    wi = W[i]
    return wi * rho * (1 + si(i, ux, uy))


def collide(ns: List[np.ndarray], neq: List[np.ndarray]) -> List[np.ndarray]:
    r = []
    for i, n in enumerate(ns):
        r.append(n - (n - neq[i]) / TAU)
    return r


def stream(ns: List[np.ndarray]) -> List[np.ndarray]:
    r = []
    ncol, nrow = ns[0].shape
    for i, n in enumerate(ns):
        # row shift means y direction. Also y is inverted
        shift_row, shift_col = -int(E[1][i]), int(E[0][i])
        # shift values according to unit vectors
        n_new = np.roll(n, (shift_row, shift_col), axis=(0, 1))
        # fill in zeros at empty places otherwise roll
        # uses continuous BC
        if shift_col == -1:
            n_new[:, -1] = 0
        elif shift_col == 1:
            n_new[:, 0] = 0
        if shift_row == -1:
            n_new[-1, :] = 0
        elif shift_row == 1:
            n_new[0, :] = 0
        r.append(n_new)
    # mid-grid BC
    # e1 -> e3 at east wall
    r[3][:, -1] += ns[1][:, -1]
    # e2 -> e4 at north wall
    r[4][0, :] += ns[2][0, :]
    # e3 -> e1 at west wall
    r[1][:, 0] += ns[3][:, 0]
    # e4 -> e2 at south wall
    r[2][-1, :] += ns[4][-1, :]
    # e5 -> e7 at east and north wall
    r[7][:, -1] += ns[5][:, -1]
    r[7][0, :-1] += ns[5][0, :-1]
    # e6 -> e8 at west and north wall
    r[8][:, 0] += ns[6][:, 0]
    r[8][0, 1:] += ns[6][0, 1:]
    # e7 -> e5 at west and south wall
    r[5][:, 0] += ns[7][:, 0]
    r[5][-1, 1:] += ns[7][-1, 1:]
    # e8 -> e6 at east and south wall
    r[6][:, -1] += ns[8][:, -1]
    r[6][-1, :-1] += ns[8][-1, :-1]

    return r


def test_calculate_ui():
    specimen = [np.ones((2, 2))] * 9
    expected_ux = np.zeros((2, 2))
    expected_uy = np.zeros((2, 2))
    actual_ux = calculate_ui(specimen, 1.0, 0)
    actual_uy = calculate_ui(specimen, 1.0, 1)
    np.testing.assert_array_equal(
        actual_ux, expected_ux, "ux calculation false"
    )
    np.testing.assert_array_equal(
        actual_uy, expected_uy, "uy calculation false"
    )


def test_calculate_rho():
    specimen = [np.ones((2, 2))] * 9
    expected = np.ones((2, 2)) * 9.0
    actual = calculate_rho(specimen)
    np.testing.assert_array_equal(actual, expected, "rho calculation false")


def test_stream():
    specimen = [np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])] * 9
    expected = [
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        np.array([[1, 1, 2], [4, 4, 5], [7, 7, 8]]),
        np.array([[4, 5, 6], [7, 8, 9], [7, 8, 9]]),
        np.array([[2, 3, 3], [5, 6, 6], [8, 9, 9]]),
        np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6]]),
        np.array([[1, 4, 5], [4, 7, 8], [7, 8, 9]]),
        np.array([[5, 6, 3], [8, 9, 6], [7, 8, 9]]),
        np.array([[1, 2, 3], [2, 3, 6], [5, 6, 9]]),
        np.array([[1, 2, 3], [4, 1, 2], [7, 4, 5]]),
    ]
    actual = stream(specimen)
    for i in range(len(specimen)):
        np.testing.assert_array_equal(
            actual[i], expected[i], err_msg=f"For array at index {i}"
        )


def main():
    # STEP 1: Initialize
    rho = np.random.rand(
        LATTICE_SIZE, LATTICE_SIZE
    )  # np.ndarray((LATTICE_SIZE, LATTICE_SIZE))
    ux = np.ndarray((LATTICE_SIZE, LATTICE_SIZE))
    uy = np.ndarray((LATTICE_SIZE, LATTICE_SIZE))
    # Discrete probablilities for each nine directions for each axis
    ns = []
    ns.append(np.ndarray((LATTICE_SIZE, LATTICE_SIZE)))
    ns.append(np.ndarray((LATTICE_SIZE, LATTICE_SIZE)))
    ns.append(np.ndarray((LATTICE_SIZE, LATTICE_SIZE)))
    ns.append(np.ndarray((LATTICE_SIZE, LATTICE_SIZE)))
    ns.append(np.ndarray((LATTICE_SIZE, LATTICE_SIZE)))
    ns.append(np.ndarray((LATTICE_SIZE, LATTICE_SIZE)))
    ns.append(np.ndarray((LATTICE_SIZE, LATTICE_SIZE)))
    ns.append(np.ndarray((LATTICE_SIZE, LATTICE_SIZE)))
    ns.append(np.ndarray((LATTICE_SIZE, LATTICE_SIZE)))
    neq = []
    neq.append(np.ndarray((LATTICE_SIZE, LATTICE_SIZE)))
    neq.append(np.ndarray((LATTICE_SIZE, LATTICE_SIZE)))
    neq.append(np.ndarray((LATTICE_SIZE, LATTICE_SIZE)))
    neq.append(np.ndarray((LATTICE_SIZE, LATTICE_SIZE)))
    neq.append(np.ndarray((LATTICE_SIZE, LATTICE_SIZE)))
    neq.append(np.ndarray((LATTICE_SIZE, LATTICE_SIZE)))
    neq.append(np.ndarray((LATTICE_SIZE, LATTICE_SIZE)))
    neq.append(np.ndarray((LATTICE_SIZE, LATTICE_SIZE)))
    neq.append(np.ndarray((LATTICE_SIZE, LATTICE_SIZE)))

    # Figure related plumbing
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(
        rho, extent=[0, LATTICE_SIZE, 0, LATTICE_SIZE], vmin=0, vmax=1
    )

    for i in range(10):
        fig.canvas.draw()
        # STEP 2: Streaming
        ns = stream(ns)

        # STEP 3: Compute macroscopic entities
        rho = calculate_rho(ns)
        ux = calculate_ui(ns, rho, axis=0)
        uy = calculate_ui(ns, rho, axis=1)
        im.set_data(rho)

        # STEP 4: Compute equilibrium number density
        for i in range(len(neq)):
            neq[i] = calculate_nieq(i, ux[i], uy[i], rho)

        # STEP 5: Collision
        ns = collide(ns, neq)


def tests():
    test_stream()
    test_calculate_ui()
    test_calculate_rho()


if __name__ == "__main__":
    main()
    # tests()
