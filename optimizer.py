import numpy as np
import cvxpy as cp
from abc import ABC, abstractmethod
from utils import *

class Solver(ABC):
    def __init__(self, n_in, n_out, dim, verbose):
        self.n_in = n_in
        self.n_out = n_out
        self.dim = dim
        self.d_in = dim ** n_in
        self.d_out = dim ** n_out
        self.verbose = verbose
    
    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def get_solution(self):
        pass


class SolverSDP(Solver):
    def __init__(self, n_in, n_out, dim, verbose, p_init_grid, p_fine_grid, n_rounds):
        super().__init__(n_in, n_out, dim, verbose)
        self.p_init_grid = p_init_grid
        self.p_fine_grid = p_fine_grid  
        self.n_rounds = n_rounds
        self.p_samples = []
    
    def make_problem(self):
        D = self.d_out * self.d_in        
        J = cp.Variable((D, D), hermitian=True)
        t = cp.Variable()  
        constraints = []
        constraints += [J >> 0]
        constraints += [t >= 0, t <= 1] 
        
        I_in = np.eye(self.d_in, dtype=complex)
        I_out = np.eye(self.d_out, dtype=complex)
        for i in range(self.d_in):
            for j in range(self.d_in):
                s = 0
                for a in range(self.d_out):
                    s += J[a * self.d_in + i, a * self.d_in + j]
                constraints += [s == I_in[i, j]]
        
        I2, X, Y, Z = qubit_paulis()
        Sx_out = collective_spin(X, n_qubits=self.n_out)
        Sy_out = collective_spin(Y, n_qubits=self.n_out)
        Sz_out = collective_spin(Z, n_qubits=self.n_out)
        Sx_in = collective_spin(X, n_qubits=self.n_in)
        Sy_in = collective_spin(Y, n_qubits=self.n_in)
        Sz_in = collective_spin(Z, n_qubits=self.n_in)
        gens = [(Sx_out, Sx_in), (Sy_out, Sy_in), (Sz_out, Sz_in)]
        for (S_out, S_in) in gens:
            A = np.kron(S_out, I_in) - np.kron(I_out, S_in.T)
            constraints += [A @ J - J @ A == 0]
        
        X_vars = []
        for p in self.p_samples:
            rho = rho_diag(float(p))
            rho_in = rho_tensor_power(rho, self.n_in)  
            ideal_rho_out = rho_tensor_power(rho, self.n_out)  
            alpha_c = cp.Constant(ideal_rho_out)
            
            K = cp.Constant(np.kron(I_out, rho_in.T))  
            M = K @ J                               
            
            sigma_blocks = []
            for a in range(self.d_out):
                row = []
                for b in range(self.d_out):
                    expr = 0
                    for i in range(self.d_in):
                        expr += M[a * self.d_in + i, b * self.d_in + i]
                    row.append(expr)
                sigma_blocks.append(row)
            sigma = cp.bmat(sigma_blocks)
            sigma = (sigma + sigma.H) / 2 

            Xk = cp.Variable((self.d_out, self.d_out), complex=True)
            X_vars.append(Xk)

            block = cp.bmat([[alpha_c, Xk],
                            [Xk.H,    sigma]])
            constraints += [block >> 0]
            constraints += [cp.real(cp.trace(Xk)) >= t]

        prob = cp.Problem(cp.Maximize(t), constraints)
        
        return prob, J, t
        
    def solve_one_round(self, solver_preference=("MOSEK", "SCS"),):
        prob, J, t = self.make_problem()
        
        chosen_solver = None
        for s in solver_preference:
            if s in cp.installed_solvers():
                chosen_solver = s
                break
        if chosen_solver is None:
            raise RuntimeError(
                "No suitable SDP solver found. Install one of: MOSEK, SCS, CVXOPT, Clarabel (depending on CVXPY support)."
            )

        prob.solve(solver=chosen_solver, verbose=self.verbose)
        if J.value is None or t.value is None:
            raise RuntimeError("Solver failed to return a solution (J.value or t.value is None).")

        return float(t.value), np.array(J.value)
    
    def solve(self):
        self.p_samples = sorted(set(np.linspace(0.5, 1.0, self.p_init_grid).tolist()))

        best = None
        for it in range(self.n_rounds):
            t_opt, J_opt = self.solve_one_round()
            
            p_fine = np.linspace(0.5, 1.0, self.p_fine_grid)
            fvals = []
            for p in p_fine:
                rho = rho_diag(float(p))
                rho_in = rho_tensor_power(rho, self.n_in)
                ideal_rho_out = rho_tensor_power(rho, self.n_out)
                sigma = apply_choi_numpy(J_opt, rho_in, self.d_out, self.d_in)
                fvals.append(fidelity_root(ideal_rho_out, sigma))

            fvals = np.array(fvals)
            idx = int(np.argmin(fvals))
            p_worst = float(p_fine[idx])
            f_worst = float(fvals[idx])

            if self.verbose:
                print(f"[Round {it}] SDP t_opt (fidelity lower bound on samples) = {t_opt**2:.8f}")
                print(f"         worst on fine grid: p={p_worst:.6f}, F={f_worst**2:.8f}")

            best = (t_opt, J_opt, self.p_samples, p_worst, f_worst)

            if min(abs(p_worst - np.array(self.p_samples))) < 1e-6:
                break
            self.p_samples.append(p_worst)
            self.p_samples = sorted(set(self.p_samples))

        return best[1]

    def get_solution(self):
        J = self.solve()
        return J


class SolverSDPTwoPoint(SolverSDP):
    def __init__(self, n_in, n_out, dim, verbose):
        super().__init__(n_in, n_out, dim, verbose, p_init_grid=2, p_fine_grid=1, n_rounds=1)
        