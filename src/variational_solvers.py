"""
In this file we implement the variational methods from Rioux et al 2026 using their memory-efficient implementation.
The dual method is just the Variational (1-0.5) method i.e. the Variational method with outer step-size set to 1 and inner step-size set to 1/2.
"""

import numpy as np
from ot.bregman import sinkhorn_epsilon_scaling, sinkhorn_knopp

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



"""
THE FUNCTIONS BELOW IMPEMENT THE MEMORY EFFICIENT COMPUTATIONS (as described in Rioux et al 2026)

"""
def _infer_problem_sizes(K0_plus, K0_minus, K1_plus, K1_minus):
    ## given factors infer number of points
    N0 = K0_plus.shape[1] if K0_plus.shape[1] != 0 else K0_minus.shape[1]
    N1 = K1_plus.shape[1] if K1_plus.shape[1] != 0 else K1_minus.shape[1]
    return N0, N1


def apply_B0_memory_efficient(plan, K0_plus, K0_minus, K1_plus, K1_minus):
    ## B0 @ coupling
    N0, N1 = _infer_problem_sizes(K0_plus, K0_minus, K1_plus, K1_minus)
    X = np.reshape(plan, (N0, N1))
    block_1 = np.sqrt(2.0) * (K0_plus @ X @ K1_plus.T).ravel()
    block_2 = np.sqrt(2.0) * (K0_minus @ X @ K1_minus.T).ravel()
    return np.concatenate([block_1, block_2])


def apply_B1_memory_efficient(plan, K0_plus, K0_minus, K1_plus, K1_minus):
    ## B1 @ coupling
    N0, N1 = _infer_problem_sizes(K0_plus, K0_minus, K1_plus, K1_minus)
    X = np.reshape(plan, (N0, N1))
    block_1 = np.sqrt(2.0) * (K0_plus @ X @ K1_minus.T).ravel()
    block_2 = np.sqrt(2.0) * (K0_minus @ X @ K1_plus.T).ravel()
    return np.concatenate([block_1, block_2])


def apply_B0T_memory_efficient(u, K0_plus, K0_minus, K1_plus, K1_minus):
    ## B0.T @ u
    r1 = K0_plus.shape[0] * K1_plus.shape[0]
    U1 = np.reshape(u[:r1], (K0_plus.shape[0], K1_plus.shape[0]))
    U2 = np.reshape(u[r1:], (K0_minus.shape[0], K1_minus.shape[0]))
    return np.sqrt(2.0) * (K0_plus.T @ U1 @ K1_plus + K0_minus.T @ U2 @ K1_minus).ravel()


def apply_B1T_memory_efficient(v, K0_plus, K0_minus, K1_plus, K1_minus):
    ## B1.T @ v
    r1 = K0_plus.shape[0] * K1_minus.shape[0]
    V1 = np.reshape(v[:r1], (K0_plus.shape[0], K1_minus.shape[0]))
    V2 = np.reshape(v[r1:], (K0_minus.shape[0], K1_plus.shape[0]))
    return np.sqrt(2.0) * (K0_plus.T @ V1 @ K1_minus + K0_minus.T @ V2 @ K1_plus).ravel()


def get_plan_var_memory_efficient(u, v, reg, wtx, wty, K0_plus, K0_minus, K1_plus, K1_minus):
    ## compute coupling given dual variables
    Nx, Ny = len(wtx), len(wty)
    c = apply_B1T_memory_efficient(v, K0_plus, K0_minus, K1_plus, K1_minus)
    c -= apply_B0T_memory_efficient(u, K0_plus, K0_minus, K1_plus, K1_minus)
    P = Sinkhorn(np.reshape(c, (Nx, Ny)), reg, wtx, wty)
    return P.ravel()


def sup_v_step_memory_efficient(u, v, reg, wtx, wty, K0_plus, K0_minus, K1_plus, K1_minus):
    # one maxmisation step
    plan = get_plan_var_memory_efficient(u, v, reg, wtx, wty, K0_plus, K0_minus, K1_plus, K1_minus)
    grad = v - apply_B1_memory_efficient(plan, K0_plus, K0_minus, K1_plus, K1_minus)
    return plan, grad


def sup_solve_memory_efficient(u, v, reg, wtx, wty, K0_plus, K0_minus, K1_plus, K1_minus, gamma, maxiter, tol=1e-6, if_track=False):
    """
    Solves the inner problem using the simple gradient descent from Algorithm 6 (Rioux et al 2026)
    
    Inputs: current dual variable (u,v),
            weights (wtx, wty),
            factors from memory efficient decomposition (K0_plus, K0_minus, K1_plus, K1_minus)
            inner stepsize (gamma),
            maxiumum number of inner iterations (maxiter),
            stopping tolerance (tol),
            if print stetment permitted: (if_track)
    
    Returns: next step v (v),
            next step coupling (plan),
            if inner tolerance reached (bool(flag_inner == 1)),
            number of iterations used (iterations)
    """
    flag_inner = 0
    iterations = 0
    for _ in range(maxiter):
        plan, grad = sup_v_step_memory_efficient(u, v, reg, wtx, wty, K0_plus, K0_minus, K1_plus, K1_minus)
        v -= gamma * grad
        iterations += 1
        stationarity = float(np.sum(grad ** 2))
        if stationarity < tol:
            flag_inner = 1
            break
    return v, plan, bool(flag_inner == 1), iterations, stationarity



def VariationalSolverMemoryEfficient(K0_plus, K0_minus, K1_plus, K1_minus, wtx, wty, J, K, 
                                     gamma, eta, reg, if_track=False, inner_tol=1e-6, 
                                     store_history=True, *, outer_tol=None, store_scalars=None, metrics_fn=None,
):
    """
    Implemtation of the simple gradient method i.e. Algorithm 6 in Rioux et al 2026

    Inputs: factors from memory efficient decomposition (K0_plus, K0_minus, K1_plus, K1_minus)
            weights (wtx, wty),
            inner maximum number of iterations (J),
            outer meximum number of iterations (K),
            inner stepsize (gamma),
            oter stepsize (eta = 1/L),
            inner stopping tolerance (inner_tol),
            outer stopping tolerance (outer_tol)
            if print stetment permitted: (if_track),
            storing and saving functions (store_history store_scalars, metrics_fn)
        
        Returns: history - depends on the saving mode but returns the most important infromation about the run, which includes the following:
                stationarity values, couplings (either final time or all coupling histories), numer of iterations for inner and outer solve,
                if the inner solve and the outer solve reached the desired tolerance, the total nubmer of sinkhron calls
    """
    store_scalars = store_history if store_scalars is None else store_scalars
    # initialize with product coupling
    P = wtx[:, None] * wty[None, :]
    x = P.ravel()

    #compute u_0, u_1
    u = apply_B0_memory_efficient(x, K0_plus, K0_minus, K1_plus, K1_minus)
    v = apply_B1_memory_efficient(x, K0_plus, K0_minus, K1_plus, K1_minus)

    if store_history:
        history = {"u": [u.copy()], "v": [v.copy()], "x": [x.copy()], "Couplings": [P.copy()]}
    if metrics_fn is not None:
        metrics_fn(P.copy())

    #tracking set-up
    stopping_residual = []
    inner_converged = []
    sinkhorn_calls = []
    flag = 0
    num_iterations = 0

    r_B0 = K0_plus.shape[0] * K1_plus.shape[0] + K0_minus.shape[0] * K1_minus.shape[0]
    r_B1 = K0_plus.shape[0] * K1_minus.shape[0] + K0_minus.shape[0] * K1_plus.shape[0]

    for i in range(K):
        if r_B0 != 0:
            if r_B1 == 0:
                ## if B1 is zero then we don't need to solve the inner problem
                plan = get_plan_var_memory_efficient(u, v, reg, wtx, wty, K0_plus, K0_minus, K1_plus, K1_minus)
                grad = u - apply_B0_memory_efficient(plan, K0_plus, K0_minus, K1_plus, K1_minus)
                u = u - eta * grad
                inner_converged.append(True)
                sinkhorn_calls.append(1)
            else:
                # solve inner problem, then take gradient step
                v, _, inner_flag, inner_iters, inner_stationarity = sup_solve_memory_efficient(
                    u, v, reg, wtx, wty, K0_plus, K0_minus, K1_plus, K1_minus, gamma, J, inner_tol, if_track
                )
                plan = get_plan_var_memory_efficient(u, v, reg, wtx, wty, K0_plus, K0_minus, K1_plus, K1_minus)
                grad = u - apply_B0_memory_efficient(plan, K0_plus, K0_minus, K1_plus, K1_minus)
                u = u - eta * grad
                inner_converged.append(inner_flag)
                sinkhorn_calls.append(inner_iters + 1)
        else:
            ## if B0 is empty then only take maximisation steps
            plan, grad = sup_v_step_memory_efficient(u, v, reg, wtx, wty, K0_plus, K0_minus, K1_plus, K1_minus)
            v = v - gamma * grad
            inner_converged.append(True)
            sinkhorn_calls.append(1)

        #compute primal stationarity proxy |u_j - u_{j+1} |^2 or equivalently | \tilde{\nabla{\ell}(u_j)} |^2
        stationarity = float(np.sum(grad ** 2))
        stopping_residual.append(stationarity)
        num_iterations = i + 1

        if store_history:
            history["u"].append(u.copy())
            history["v"].append(v.copy())
            history["x"].append(plan.copy())
            history["Couplings"].append(plan.reshape((wtx.shape[0], wty.shape[0])))
        if metrics_fn is not None:
            metrics_fn(plan.reshape((wtx.shape[0], wty.shape[0])))

        if inner_flag != 1 and if_track:
            print(f"Iteration {i}, inner tolerance not met, inner stationarity:{inner_stationarity}")

        if if_track and i % 50 == 0 or i == K - 1 or stationarity < (inner_tol if outer_tol is None else outer_tol):
                print(f"Iteration {i}, Squared L2 gradient: {stationarity}")

        #break early if tolerance met
        if stationarity < (inner_tol if outer_tol is None else outer_tol):
            flag = 1
            break

    if flag != 1 and if_track:
        print("Tolerance not met")

    ## return only final iterates (to save memory if history not used)
    if not store_history:
        history = {
            "u": [u.copy()],
            "v": [v.copy()],
            "x": [plan.copy()],
            "Couplings": [plan.reshape((wtx.shape[0], wty.shape[0]))],
        }

    history["stopping_residual"] = stopping_residual if store_scalars else stopping_residual[-1:]
    history["inner_converged"] = inner_converged if store_scalars else inner_converged[-1:]
    history["sinkhorn_calls"] = sinkhorn_calls if store_scalars else sinkhorn_calls[-1:]
    history["num_iterations"] = num_iterations
    history["converged"] = bool(flag == 1)

    return history


def AcceleratedVariationalSolverMemoryEfficient(K0_plus, K0_minus, K1_plus, K1_minus, eig_B0, wtx,wty,
    J, K, L, gamma_v, reg, if_track=False, inner_tol=1e-6, store_history=True,
    *, outer_tol=None, store_scalars=None, metrics_fn=None,
):
    """
        Implemtation of the simple gradient method i.e. Algorithm 6 in Rioux et al 2026
    
        Inputs: factors from memory efficient decomposition (K0_plus, K0_minus, K1_plus, K1_minus)
                weights (wtx, wty),
                inner maximum number of iterations (J),
                outer meximum number of iterations (K),
                Lipschitz constant which determines stepping sequances (L),
                inner stepsize (gamma_v),
                inner stopping tolerance (inner_tol),
                outer stopping tolerance (outer_tol)
                if print stetment permitted: (if_track),
                storing and saving functions (store_history store_scalars, metrics_fn)
            
        Returns: history - depends on the saving mode but returns the most important infromation about the run, which includes the following:
                    stationarity values, couplings (either final time or all coupling histories), numer of iterations for inner and outer solve,
                    if the inner solve and the outer solve reached the desired tolerance, the total nubmer of sinkhron calls
    """
    store_scalars = store_history if store_scalars is None else store_scalars
    # initialize with product coupling
    P = wtx[:, None] * wty[None, :]
    x = P.ravel()

    #compute s_0, v_1
    s = apply_B0_memory_efficient(x, K0_plus, K0_minus, K1_plus, K1_minus)
    v = apply_B1_memory_efficient(x, K0_plus, K0_minus, K1_plus, K1_minus)
    z = s.copy()
    u = z.copy()

    #compute beta and R0
    beta = 1 / (2 * L)
    R0 = np.sqrt(np.sum(eig_B0)) if eig_B0.size else 0.0

    if store_history:
        history = {"u": [], "z": [z.copy()], "v": [v.copy()], "x": [x.copy()], "Couplings": [P.copy()]}
    if metrics_fn is not None:
        metrics_fn(P.copy())

    stopping_residual = []
    inner_converged = []
    sinkhorn_calls = []
    flag = 0
    num_iterations = 0

    r_B0 = K0_plus.shape[0] * K1_plus.shape[0] + K0_minus.shape[0] * K1_minus.shape[0]
    r_B1 = K0_plus.shape[0] * K1_minus.shape[0] + K0_minus.shape[0] * K1_plus.shape[0]

    for k in range(K):
        gamma = (k + 1) / (4 * L)
        tau = 2 / (k + 3)
        if r_B0 != 0:
            if r_B1 == 0:
                ## if B1 is zero then we don't need to solve the inner problem
                plan = get_plan_var_memory_efficient(z, v, reg, wtx, wty, K0_plus, K0_minus, K1_plus, K1_minus)
                inner_converged.append(True)
                sinkhorn_calls.append(1)
            else:
                v, _, inner_flag, inner_iters, _inner_stationarity = sup_solve_memory_efficient(
                    z, v, reg, wtx, wty, K0_plus, K0_minus, K1_plus, K1_minus, gamma_v, J, inner_tol, if_track
                )
                plan = get_plan_var_memory_efficient(z, v, reg, wtx, wty, K0_plus, K0_minus, K1_plus, K1_minus)
                inner_converged.append(inner_flag)
                sinkhorn_calls.append(inner_iters + 1)

            #compute gradient and take acceleration step  
            grad = z - apply_B0_memory_efficient(plan, K0_plus, K0_minus, K1_plus, K1_minus)
            u = z - min(1, R0 / (np.linalg.norm(z - beta * grad) + inner_tol)) * beta * grad
            s = s - min(1, R0 / (np.linalg.norm(s - gamma * grad) + inner_tol)) * gamma * grad
            z = tau * s + (1.0 - tau) * u
        else:
            ## if B0 is empty then only take maximisation steps
            u = apply_B0_memory_efficient(x, K0_plus, K0_minus, K1_plus, K1_minus)
            plan, grad = sup_v_step_memory_efficient(z, v, reg, wtx, wty, K0_plus, K0_minus, K1_plus, K1_minus)
            v = v - gamma_v * grad
            inner_converged.append(True)
            sinkhorn_calls.append(1)

        #compute primal stationarity proxy |u_j - z_j |^2 or equivalently | \tilde{\nabla{\ell}(u_j)} |^2
        stationarity = float(np.sum(grad ** 2))
        stopping_residual.append(stationarity)
        num_iterations = k + 1

        if store_history:
            history["u"].append(u.copy())
            history["z"].append(z.copy())
            history["v"].append(v.copy())
            history["x"].append(plan.copy())
            history["Couplings"].append(plan.reshape((wtx.shape[0], wty.shape[0])))

        if metrics_fn is not None:
            metrics_fn(plan.reshape((wtx.shape[0], wty.shape[0])))


        if inner_flag != 1 and if_track:
                print(f"Iteration {k}, inner tolerance not met")

        if if_track and k % 50 == 0 or k == K - 1 or stationarity < (inner_tol if outer_tol is None else outer_tol):
                print(f"Iteration {k}, Squared L2 gradient: {stationarity}")

        #break early if tolerance met
        if stationarity < (inner_tol if outer_tol is None else outer_tol):
            flag = 1
            break

    
    if flag != 1 and if_track:
        print("Tolerance not met")
    
    if not store_history:
        history = {
            "u": [u.copy()],
            "z": [z.copy()],
            "v": [v.copy()],
            "x": [plan.copy()],
            "Couplings": [plan.reshape((wtx.shape[0], wty.shape[0]))],
        }
    history["stopping_residual"] = stopping_residual if store_scalars else stopping_residual[-1:]
    history["inner_converged"] = inner_converged if store_scalars else inner_converged[-1:]
    history["sinkhorn_calls"] = sinkhorn_calls if store_scalars else sinkhorn_calls[-1:]
    history["num_iterations"] = num_iterations
    history["converged"] = bool(flag == 1)

    return history



def DualSolverVariationalMemoryEfficient(K0_plus, K0_minus, K1_plus, K1_minus, wtx, wty, J=50, K=200, reg=0.01,
                                         if_track=False, inner_tol=1e-6, gamma=0.5, store_history=True,
                                         *, outer_tol=None, store_scalars=None, metrics_fn=None):
    """
    Variational (1-gamma) method, simply run VariationalSolver with eta=1 and the given gamma.
    """
    return VariationalSolverMemoryEfficient(K0_plus, K0_minus, K1_plus, K1_minus, wtx, wty, J, K,
                                            gamma=gamma, eta=1.0, reg=reg, if_track=if_track, inner_tol=inner_tol,
                                            store_history=store_history, outer_tol=outer_tol,
                                            store_scalars=store_scalars, metrics_fn=metrics_fn)



"""
BELOW ARE FUNCITONS NEEDED FOR NON-MOMORY EFFICIENT IMPLEMENTAITON OF THE SOLVERS
"""

def get_plan_var(u, v, reg, wtx, wty, B0, B1):
    Nx, Ny = len(wtx), len(wty)
    c = B1.T @ v - B0.T @ u
    P = Sinkhorn(np.reshape(c, (Nx, Ny)), reg, wtx, wty)
    return P.ravel()


def sup_v_step(u, v, reg, wtx, wty, B0, B1):
    plan = get_plan_var(u, v, reg, wtx, wty, B0, B1)
    grad = v - B1 @ plan
    return plan, grad

def sup_solve(u, v, reg, wtx, wty, B0, B1, stepsize, maxiter, tol=1e-6, if_track=False):
    flag = 0
    iterations = 0
    for _ in range(maxiter):
        plan, grad = sup_v_step(u, v, reg, wtx, wty, B0, B1)
        v -= stepsize * grad
        iterations += 1
        stationarity = float(np.sum(grad ** 2))
        if stationarity < tol:
            flag = 1
            break
    if flag != 1 and if_track:
        print(f"Inner tolerance not met, Squared L2 gradient: {stationarity}")
    return v, plan, bool(flag == 1), iterations

def VariationalSolver(B0, B1, wtx, wty, J, K, gamma, eta, reg, if_track=False, inner_tol=1e-6, store_history=True,
                       *, outer_tol=None, store_scalars=None, metrics_fn=None):

    store_scalars = store_history if store_scalars is None else store_scalars
    P = wtx[:, None] * wty[None, :]
    u = B0 @ P.ravel()
    v = B1 @ P.ravel()

    if store_history:
        history = {"u": [u.copy()], "v": [v.copy()], "x": [P.ravel().copy()], "Couplings": [P.copy()]}
    if metrics_fn is not None:
        metrics_fn(P.copy())
    stopping_residual = []
    inner_converged = []
    sinkhorn_calls = []
    flag = 0
    num_iterations = 0
    for i in range(K):
        if np.shape(B0)[0] != 0:
            if np.shape(B1)[0] == 0:
                plan = get_plan_var(u, v, reg, wtx, wty, B0, B1)
                grad = u - B0 @ plan
                u = u - eta * grad
                inner_converged.append(True)
                sinkhorn_calls.append(1)
            else:
                v, _, inner_flag, inner_iters = sup_solve(u, v, reg, wtx, wty, B0, B1, gamma, J, inner_tol, if_track)
                plan = get_plan_var(u, v, reg, wtx, wty, B0, B1)
                grad = u - B0 @ plan
                u = u - eta * grad
                inner_converged.append(inner_flag)
                sinkhorn_calls.append(inner_iters + 1)
        else:
            plan, grad = sup_v_step(u, v, reg, wtx, wty, B0, B1)
            v = v - gamma * grad
            inner_converged.append(True)
            sinkhorn_calls.append(1)

        stationarity = float(np.sum(grad ** 2))
        stopping_residual.append(stationarity)
        num_iterations = i + 1
        if store_history:
            history["u"].append(u.copy())
            history["v"].append(v.copy())
            history["x"].append(plan.copy())
            history["Couplings"].append(plan.reshape((wtx.shape[0], wty.shape[0])))
        if metrics_fn is not None:
            metrics_fn(plan.reshape((wtx.shape[0], wty.shape[0])))

        if if_track and i % 50 == 0 or i == K - 1 or stationarity < (inner_tol if outer_tol is None else outer_tol):
            print(f"Iteration {i}, Squared L2 gradient: {stationarity}")

        if stationarity < (inner_tol if outer_tol is None else outer_tol):
            flag = 1
            break

    if flag != 1 and if_track:
        print("Tolerance not met")
    if not store_history:
        history = {
            "u": [u.copy()],
            "v": [v.copy()],
            "x": [plan.copy()],
            "Couplings": [plan.reshape((wtx.shape[0], wty.shape[0]))],
        }
    history["stopping_residual"] = stopping_residual if store_scalars else stopping_residual[-1:]
    history["inner_converged"] = inner_converged if store_scalars else inner_converged[-1:]
    history["sinkhorn_calls"] = sinkhorn_calls if store_scalars else sinkhorn_calls[-1:]
    history["num_iterations"] = num_iterations
    history["converged"] = bool(flag == 1)
    return history

def AcceleratedVariationalSolver(B0, B1, wtx, wty, J, K, L, gamma_v, reg, if_track=False, inner_tol=1e-6, store_history=True,
                                  *, outer_tol=None, store_scalars=None, metrics_fn=None):
    """Stop on the squared L2 gradient norm; outer_tol overrides tol for outer stopping only.

    See VariationalSolver for what store_scalars/metrics_fn do independently of
    store_history.
    """
    store_scalars = store_history if store_scalars is None else store_scalars
    P = wtx[:, None] * wty[None, :]
    x = P.ravel()
    s = B0 @ x
    v = B1 @ x
    z = s.copy()
    u = z.copy()
    beta = 1 / (2 * L)
    R0 = np.linalg.norm(B0)
    if store_history:
        history = {"u": [], "z": [z.copy()], "v": [v.copy()], "x": [x.copy()], "Couplings": [P.copy()]}
    if metrics_fn is not None:
        metrics_fn(P.copy())
    stopping_residual = []
    inner_converged = []
    sinkhorn_calls = []
    flag = 0
    num_iterations = 0

    for k in range(K):
        gamma = (k + 1) / (4 * L)
        tau = 2 / (k + 3)
        if np.shape(B0)[0] != 0:
            if np.shape(B1)[0] == 0:
                plan = get_plan_var(z, v, reg, wtx, wty, B0, B1)
                inner_converged.append(True)
                sinkhorn_calls.append(1)
            else:
                v, _, inner_flag, inner_iters = sup_solve(z, v, reg, wtx, wty, B0, B1, gamma_v, J, inner_tol, if_track)
                plan = get_plan_var(z, v, reg, wtx, wty, B0, B1)
                inner_converged.append(inner_flag)
                sinkhorn_calls.append(inner_iters + 1)
            grad = z - B0 @ plan
            u = z - min(1, R0 / (np.linalg.norm(z - beta * grad) + inner_tol)) * beta * grad
            s = s - min(1, R0 / (np.linalg.norm(s - gamma * grad) + inner_tol)) * gamma * grad
            z = tau * s + (1.0 - tau) * u
        else:
            plan, grad = sup_v_step(z, v, reg, wtx, wty, B0, B1)
            v = v - gamma_v * grad
            inner_converged.append(True)
            sinkhorn_calls.append(1)

        stationarity = float(np.sum(grad ** 2))
        stopping_residual.append(stationarity)
        num_iterations = k + 1
        if store_history:
            history["u"].append(u.copy())
            history["z"].append(z.copy())
            history["v"].append(v.copy())
            history["x"].append(plan.copy())
            history["Couplings"].append(plan.reshape((wtx.shape[0], wty.shape[0])))
        if metrics_fn is not None:
            metrics_fn(plan.reshape((wtx.shape[0], wty.shape[0])))

        if if_track and k % 50 == 0 or k == K - 1 or stationarity < (inner_tol if outer_tol is None else outer_tol):
            print(f"Iteration {k}, Squared L2 gradient: {stationarity}")

        if stationarity < (inner_tol if outer_tol is None else outer_tol):
            flag = 1
            break     

    if flag != 1 and if_track:
        print("Tolerance not met")
    if not store_history:
        history = {
            "u": [u.copy()],
            "z": [z.copy()],
            "v": [v.copy()],
            "x": [plan.copy()],
            "Couplings": [plan.reshape((wtx.shape[0], wty.shape[0]))],
        }
    history["stopping_residual"] = stopping_residual if store_scalars else stopping_residual[-1:]
    history["inner_converged"] = inner_converged if store_scalars else inner_converged[-1:]
    history["sinkhorn_calls"] = sinkhorn_calls if store_scalars else sinkhorn_calls[-1:]
    history["num_iterations"] = num_iterations
    history["converged"] = bool(flag == 1)
    return history


def DualSolverVariational(B0, B1, wtx, wty, J=50, K=200, reg=0.01, if_track=False, inner_tol=1e-6,
                           gamma=1.0, store_history=True, *, outer_tol=None, store_scalars=None, metrics_fn=None):
    """ Variational (1-gamma) with the outer step-size fixed at 1.0 and the given inner gamma."""
    return VariationalSolver(B0, B1, wtx, wty, J, K, gamma=gamma, eta=1.0, reg=reg, if_track=if_track,
                             inner_tol=inner_tol, store_history=store_history, outer_tol=outer_tol,
                             store_scalars=store_scalars, metrics_fn=metrics_fn)
