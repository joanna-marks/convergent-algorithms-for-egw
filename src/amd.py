"""
This file contains code implementing the Averaged Mirror Descent algorithm f
or solving the Gromov-Wasserstein problem. 
To implement Mirror Descent update we follow the Quadratic entroipic GW (Algorithm 2)
from Scetbon et al. 2022.
It incudes a step-size parameter 'alpha'; 
setting alpha=1 recovers the standard Mirror Descent algorithm.
"""

import numpy as np
from ot.bregman import sinkhorn_epsilon_scaling, sinkhorn_knopp


# selected Sinkhorn solvers from POT library
SINKHORN_SOLVERS = {
    "sinkhorn_epsilon_scaling": sinkhorn_epsilon_scaling,
    "sinkhorn": sinkhorn_knopp,
}


def Sinkhorn(C, reg, wtx, wty, delta=1e-9, num_iter=1000, method="sinkhorn"):
    """
    Inputs: cost matrix (C), 
            regularization parameter (reg), 
            weights (wtx,wty),
            Sinkron tolerance (delta),
            Sinkhron maximum iterations (num_iter)
            Sinkhron solver (method - to be chosen from SINHORN_SOLVERS)

    Returns: optimal coupling
    """
    if method not in SINKHORN_SOLVERS:
        raise ValueError(f"Unknown Sinkhorn method {method!r}; choose one of {sorted(SINKHORN_SOLVERS)}.")
    C_shifted = C - np.min(C)
    return SINKHORN_SOLVERS[method](
        wtx,
        wty,
        C_shifted,
        reg,
        numItermax=num_iter,
        stopThr=delta,
        warn=False,
    )


def factorize_for_mirror_descent(C):
    """ Eigendecomposition of the cost matrix"""
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    return eigenvectors @ np.diag(eigenvalues), eigenvectors.T


def step_mirror(L, reg, wtx, wty, A_1, A_2, B_1, B_2, Pi_old, alpha, sinkhorn_delta=1e-9, sinkhorn_num_iter=1000,
                 sinkhorn_method="sinkhorn"):
    """ 
    Computes one step of the Averaged Mirror Descent (AMD)

    Inputs: previous step transformed cost (L),
            regularization parameter (reg), 
            weights (wtx,wty),
            factorized matrices (A_1, A_2 -- constructed from Cx),
            factorized matrices (B_1, B_2 -- constructed from Cy),
            coupling at previous iteration (Pi_old)
            the AMD step-size (alpha),
            Sinkhorn solver paramameters (detla, sinkhorn_num_iter, sinkhron_method)

    Return: the coupling returned by Sinkhron solve tilde{x}_j (Pi_new),
            next step transformed cost (L),
            new coupling x_j (P)
    """

    Pi_new = Sinkhorn(L, reg, wtx, wty, delta=sinkhorn_delta, num_iter=sinkhorn_num_iter, method=sinkhorn_method)
    P = alpha * Pi_new + (1 - alpha) * Pi_old
    C_trans = np.dot(np.dot(B_2, P.T), A_1)
    L = -4 * np.dot(np.dot(B_1, C_trans), A_2).T
    return Pi_new, L, P


def MirrorDescent(reg, wtx, wty, A_1, A_2, B_1, B_2, P_init, alpha, K,
    tol=1e-6, store_history=True, sinkhorn_delta=1e-9, sinkhorn_num_iter=1000, sinkhorn_method="sinkhorn",
    *, metrics_fn=None, stationarity_fn=None,
):
    """
    The Averaged Mirror Descent method

    Inputs: regularization parameter (reg), 
            weights (wtx, wty),
            factorized matrices (A_1, A_2 -- constructed from Cx),
            factorized matrices (B_1, B_2 -- constructed from Cy),
            initial coupling (P_init),
            averaging step-size (alpha),
            maximum number of iterations (K)
            stopping tolerance (tol)
            Sinkhorn solver paramameters (detla, sinkhorn_num_iter, sinkhron_method)
            saving parameters (metrics_fn, stationarity_fn -- decides if we store all of the couplings or if we store stationarity as a scaler)
    
            
    Returns: array of couplings at each iteration (couplings),
            array of results of inner Sinkhrn solvers (couplings_tilde)
    """
    x_j = P_init.copy()
    C_trans = np.dot(np.dot(B_2, x_j.T), A_1)
    L = -4 * np.dot(np.dot(B_1, C_trans), A_2).T

    if store_history:
        couplings = [P_init.copy()]
        couplings_tilde = []
    if metrics_fn is not None:
        metrics_fn(P_init.copy())
     
    flag = 0 # to flag if tolerance resarched
    for _ in range(K):
        x_tilde, L, x_j1 = step_mirror(
            L,
            reg,
            wtx,
            wty,
            A_1,
            A_2,
            B_1,
            B_2,
            x_j,
            alpha,
            sinkhorn_delta=sinkhorn_delta,
            sinkhorn_num_iter=sinkhorn_num_iter,
            sinkhorn_method=sinkhorn_method,
        )

        #compute primal stationarity proxy |x_j - tilde{x}_j |^2
        stationarity = np.sum(np.abs(x_j - x_tilde)) ** 2

        if store_history:
            couplings.append(x_j1.copy())
            couplings_tilde.append(x_tilde.copy())

        if metrics_fn is not None:
            metrics_fn(x_j1.copy())
      
        if stationarity_fn is not None:
            stationarity_fn(float(stationarity))

        x_j = x_j1

        if _ % 50 == 0 or _ == K - 1 or stationarity < tol:
                    print(f"Iteration {_}, Stationarity: {stationarity}")

        # set early breaking based on the stationarity value
        if stationarity < tol:
            flag = 1
            break

        

    if not store_history:
        couplings = [x_j.copy()]
        couplings_tilde = []

    return couplings, couplings_tilde
