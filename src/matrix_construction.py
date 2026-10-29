import numpy as np

def _signed_symmetric_factors(C, tol=1e-10):
    """ Eigenvector decomposition"""
    eigvals, eigvecs = np.linalg.eigh(C)
    pos_mask = eigvals > tol
    neg_mask = eigvals < -tol

    K_plus = np.sqrt(eigvals[pos_mask])[:, np.newaxis] * eigvecs[:, pos_mask].T
    K_minus = np.sqrt(-eigvals[neg_mask])[:, np.newaxis] * eigvecs[:, neg_mask].T

    if K_plus.size == 0:
        K_plus = np.zeros((0, C.shape[0]))
    if K_minus.size == 0:
        K_minus = np.zeros((0, C.shape[0]))

    return K_plus, K_minus


def construct_B0_B1_memory_efficient(Cx, Cy, tol=1e-10):
    """ Construct the memory efficient factors"""
    K0_plus, K0_minus = _signed_symmetric_factors(Cx, tol=tol)
    K1_plus, K1_minus = _signed_symmetric_factors(Cy, tol=tol)
    K0_plus = np.sqrt(2.0) * K0_plus
    K0_minus = np.sqrt(2.0) * K0_minus

    eig_B0 = 2.0 * np.concatenate(
        [
            np.outer(np.sum(K0_plus**2, axis=1), np.sum(K1_plus**2, axis=1)).ravel(),
            np.outer(np.sum(K0_minus**2, axis=1), np.sum(K1_minus**2, axis=1)).ravel(),
        ]
    )
    eig_B1 = 2.0 * np.concatenate(
        [
            np.outer(np.sum(K0_plus**2, axis=1), np.sum(K1_minus**2, axis=1)).ravel(),
            np.outer(np.sum(K0_minus**2, axis=1), np.sum(K1_plus**2, axis=1)).ravel(),
        ]
    )

    return K0_plus, K0_minus, K1_plus, K1_minus, eig_B0, eig_B1


def compute_L(eig_B0, eig_B1, reg):
    """ Compute the thoretical L for variatonal methods"""
    lambda_max = 0.0 if eig_B0.size == 0 else np.max(eig_B0)
    lambda_min = 0.0 if eig_B1.size == 0 else np.min(eig_B1)
    return lambda_max / (lambda_min + reg) + 1.0


def compute_accelerated_L(eig_B0, eig_B1, reg):
    """Compute the thoretical L for accelerated method"""
    return max(1.0, compute_L(eig_B0, eig_B1, reg) - 1.0)


def construct_B0_B1_efficient(Cx, Cy, tol=1e-10):
    n, m = Cx.shape[0], Cy.shape[0]

    eig_a, vec_a = np.linalg.eigh(Cx)
    eig_b, vec_b = np.linalg.eigh(Cy)

    global_eig = -2.0 * np.outer(eig_a, eig_b).ravel()

    pos_indices = np.where(global_eig > tol)[0]
    neg_indices = np.where(global_eig < -tol)[0]

    B1 = []
    for idx in pos_indices:
        i, j = divmod(idx, m)
        B1.append(np.sqrt(global_eig[idx]) * np.kron(vec_a[:, i], vec_b[:, j]))
    B1 = np.array(B1) if B1 else np.zeros((0, n * m))

    B0 = []
    for idx in neg_indices:
        i, j = divmod(idx, m)
        B0.append(np.sqrt(-global_eig[idx]) * np.kron(vec_a[:, i], vec_b[:, j]))
    B0 = np.array(B0) if B0 else np.zeros((0, n * m))

    eig_B0 = np.abs(global_eig[neg_indices])
    eig_B1 = global_eig[pos_indices]

    return B0, B1, eig_B0, eig_B1
