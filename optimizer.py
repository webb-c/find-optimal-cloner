import numpy as np
import cvxpy as cp
import os
import math
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
        
        ptr_J = cp.partial_trace(J, (self.d_out, self.d_in), axis=0)
        constraints += [ptr_J == I_in]
        
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
            
            sigma = cp.partial_trace(M, (self.d_out, self.d_in), axis=1)
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


class SolverSDPPerm(Solver):
    """Permutation + SU(2) symmetry reduced SDP.
    """
    def __init__(self, n_in, n_out, dim=2, verbose=False, p_init_grid=5, p_fine_grid=51, n_rounds=3):
        super().__init__(n_in, n_out, dim, verbose)
        if dim != 2:
            raise ValueError("SolverSDPPerm currently implements qubit(SU(2)) case only: dim must be 2.")

        self.p_init_grid = p_init_grid
        self.p_fine_grid = p_fine_grid
        self.n_rounds = n_rounds
        self.p_samples = []

        # spin sectors + multiplicities
        self.j2_in_list = j2_list_for_n_qubits(n_in)
        self.j2_out_list = j2_list_for_n_qubits(n_out)
        self.mult_in = {j2: mult_qubits(n_in, j2) for j2 in self.j2_in_list}
        self.mult_out = {j2: mult_qubits(n_out, j2) for j2 in self.j2_out_list}

        # Precompute SU(2)-commutant projectors for each (j_out,j_in)
        self.P = {}  # (j2_out,j2_in) -> list of (L2, Π)
        for j2o in self.j2_out_list:
            for j2i in self.j2_in_list:
                self.P[(j2o, j2i)] = su2_commutant_projectors(j2o, j2i)

    def _rho_blocks(self, n: int, p: float, j2_list):
        return {j2: rho_block_diag_in_spin_irrep(n, j2, p) for j2 in j2_list}

    def _rho_blocks_grid(self, n: int, j2_list, p_grid):
        p_grid = np.asarray(p_grid, dtype=float).reshape(-1, 1)
        q_grid = 1.0 - p_grid
        out = {}
        for j2 in j2_list:
            m2_vals = np.arange(j2, -j2 - 1, -2, dtype=int).reshape(1, -1)
            exp_p = ((n + m2_vals) // 2)
            exp_q = ((n - m2_vals) // 2)
            out[j2] = (p_grid ** exp_p) * (q_grid ** exp_q)
        return out

    def _diagonal_transition_matrix_from_choi(self, J: np.ndarray, j2_out: int, j2_in: int):
        d_out = int(j2_out) + 1
        d_in = int(j2_in) + 1
        J = np.asarray(J, dtype=complex)
        if J.shape != (d_out * d_in, d_out * d_in):
            raise ValueError(
                f"Block for (j2_out={j2_out}, j2_in={j2_in}) has shape {J.shape}, "
                f"expected {(d_out * d_in, d_out * d_in)}."
            )
        J4 = J.reshape(d_out, d_in, d_out, d_in)
        T = np.einsum("aiai->ai", J4)
        T = ((T + T.conj()) / 2.0).real
        return np.asarray(T, dtype=float)

    def diagonal_transition_cache(self, J_blocks):
        cache = {}
        for j2o in self.j2_out_list:
            d_out = j2o + 1
            for j2i in self.j2_in_list:
                d_in = j2i + 1
                if (j2o, j2i) not in J_blocks:
                    cache[(j2o, j2i)] = np.zeros((d_out, d_in), dtype=float)
                    continue
                cache[(j2o, j2i)] = self._diagonal_transition_matrix_from_choi(J_blocks[(j2o, j2i)], j2o, j2i)
        return cache

    def fidelity_curve_blocks(self, J_blocks, p_grid=None, transition_cache=None):
        if p_grid is None:
            p_grid = np.linspace(0.5, 1.0, self.p_fine_grid)
        p_grid = np.asarray(p_grid, dtype=float)
        if p_grid.ndim != 1:
            raise ValueError("p_grid must be a 1D array.")

        T = self.diagonal_transition_cache(J_blocks) if transition_cache is None else transition_cache
        rho_in_grid = self._rho_blocks_grid(self.n_in, self.j2_in_list, p_grid)
        rho_out_grid = self._rho_blocks_grid(self.n_out, self.j2_out_list, p_grid)

        n_p = p_grid.size
        root_curve = np.zeros(n_p, dtype=float)
        for j2o in self.j2_out_list:
            d_out = j2o + 1
            sigma_diag = np.zeros((n_p, d_out), dtype=float)
            for j2i in self.j2_in_list:
                sigma_diag += self.mult_in[j2i] * (rho_in_grid[j2i] @ T[(j2o, j2i)].T)
            sigma_diag = np.clip(sigma_diag, 0.0, None)
            target_diag = np.clip(rho_out_grid[j2o], 0.0, None)
            root_curve += self.mult_out[j2o] * np.sum(np.sqrt(target_diag * sigma_diag), axis=1)
        fidelity_curve = root_curve ** 2
        return p_grid, root_curve, fidelity_curve

    def make_problem(self):
        t = cp.Variable()
        constraints = [t >= 0, t <= 1]

        coeff = {}   
        Jexpr = {}   

        for j2o in self.j2_out_list:
            for j2i in self.j2_in_list:
                proj_list = self.P[(j2o, j2i)]
                terms = []
                for (L2, Pi) in proj_list:
                    x = cp.Variable(nonneg=True)
                    coeff[(j2o, j2i, L2)] = x
                    terms.append(x * cp.Constant(Pi))
                # 이제 Jexpr은 mu_out * J 인 스케일된 변수 \tilde{J}를 나타냅니다.
                Jexpr[(j2o, j2i)] = sum(terms)

        for j2i in self.j2_in_list:
            d_in = j2i + 1
            lhs = 0
            for j2o in self.j2_out_list:
                d_out = j2o + 1
                ptr = cp.partial_trace(Jexpr[(j2o, j2i)], (d_out, d_in), axis=0)
                # Jexpr이 이미 mu_out을 포함하므로, 여기서는 mult_out을 곱하지 않습니다.
                lhs += ptr 
            constraints += [lhs == np.eye(d_in, dtype=complex)]

        for p in self.p_samples:
            rho_in = self._rho_blocks(self.n_in, float(p), self.j2_in_list)
            alpha_out = self._rho_blocks(self.n_out, float(p), self.j2_out_list)

            sigma = {}
            for j2o in self.j2_out_list:
                d_out = j2o + 1
                sig = 0
                for j2i in self.j2_in_list:
                    d_in = j2i + 1
                    K = np.kron(np.eye(d_out, dtype=complex), rho_in[j2i].T)
                    M = cp.Constant(K) @ Jexpr[(j2o, j2i)]
                    sig_part = cp.partial_trace(M, (d_out, d_in), axis=1)
                    sig += self.mult_in[j2i] * sig_part
                # sigma[j2o] 역시 mu_out * \sigma 인 \tilde{\sigma} 가 됩니다.
                sigma[j2o] = (sig + sig.H) / 2

            fid_sum = 0
            for j2o in self.j2_out_list:
                d_out = j2o + 1
                X = cp.Variable((d_out, d_out), complex=True)
                # LMI 블록 전체에 mu_out을 곱하여 스케일링합니다.
                A = cp.Constant(alpha_out[j2o] * self.mult_out[j2o]) 

                block = cp.bmat([[A, X],
                                    [X.H, sigma[j2o]]])
                constraints += [block >> 0]
                # X도 mu_out이 곱해진 \tilde{X}이므로 mult_out을 곱하지 않습니다.
                fid_sum += cp.real(cp.trace(X)) 

            constraints += [fid_sum >= t]

        prob = cp.Problem(cp.Maximize(t), constraints)
        return prob, coeff, t
    
    def solve_one_round(self, solver_preference=("MOSEK", "SCS")):
        prob, coeff, t = self.make_problem()

        chosen = None
        for s in solver_preference:
            if s in cp.installed_solvers():
                chosen = s
                break
        if chosen is None:
            raise RuntimeError("No suitable solver found. Install one of MOSEK or SCS (etc.).")

        prob.solve(solver=chosen, verbose=self.verbose)
        if t.value is None:
            raise RuntimeError("Solver failed: t.value is None")

        J_blocks = {}
        for j2o in self.j2_out_list:
            for j2i in self.j2_in_list:
                d_out = j2o + 1
                d_in = j2i + 1
                J = np.zeros((d_out * d_in, d_out * d_in), dtype=complex)
                for (L2, Pi) in self.P[(j2o, j2i)]:
                    x = coeff[(j2o, j2i, L2)].value
                    if x is None:
                        raise RuntimeError("Solver failed: some coefficient is None")
                    # 최적화 변수 x를 원래의 물리적 스케일로 복원하기 위해 mult_out으로 나눕니다.
                    J += (float(x) / self.mult_out[j2o]) * Pi 
                J_blocks[(j2o, j2i)] = (J + J.conj().T) / 2
        return float(t.value), J_blocks
    
    def _apply_channel_blocks_numpy(self, J_blocks, p: float):
        rho_in = self._rho_blocks(self.n_in, float(p), self.j2_in_list)
        sigma = {}
        for j2o in self.j2_out_list:
            d_out = j2o + 1
            sig = np.zeros((d_out, d_out), dtype=complex)
            for j2i in self.j2_in_list:
                d_in = j2i + 1
                J = J_blocks[(j2o, j2i)]
                K = np.kron(np.eye(d_out, dtype=complex), rho_in[j2i].T)
                M = K @ J
                sig_part = partial_trace_numpy(M, d_out, d_in, axis=1)
                sig += self.mult_in[j2i] * sig_part
            sigma[j2o] = (sig + sig.conj().T) / 2
        return sigma

    def _root_fidelity_full_numpy(self, p: float, sigma_blocks):
        alpha = self._rho_blocks(self.n_out, float(p), self.j2_out_list)
        f = 0.0
        for j2o in self.j2_out_list:
            f_j = fidelity_root_numpy(alpha[j2o], sigma_blocks[j2o])
            f += self.mult_out[j2o] * f_j
        return float(f)

    def solve(self):
        # initial samples
        self.p_samples = sorted(set(np.linspace(0.5, 1.0, self.p_init_grid).tolist()))

        best = None
        for it in range(self.n_rounds):
            t_opt, J_opt = self.solve_one_round() # call self.make_problem() in this method; and self.make_problem() uses self.p_samples.

            # evaluate on fine grid using the diagonal transfer-matrix fast path
            p_fine = np.linspace(0.5, 1.0, self.p_fine_grid)
            _, root_curve, _ = self.fidelity_curve_blocks(J_opt, p_fine)
            fvals = np.asarray(root_curve, dtype=float)

            idx = int(np.argmin(fvals))
            p_worst = float(p_fine[idx])
            f_worst = float(fvals[idx])

            if self.verbose:
                print(f"[Round {it}] SDP t_opt (root fidelity LB on samples) = {t_opt:.8f} (F≈{t_opt**2:.8f})")
                print(f"         worst on fine grid: p={p_worst:.6f}, rootF={f_worst:.8f} (F≈{f_worst**2:.8f})")

            best = (t_opt, J_opt, list(self.p_samples), p_worst, f_worst)
            
            if min(abs(p_worst - np.array(self.p_samples))) < 1e-6:
                break
            self.p_samples.append(p_worst)
            self.p_samples = sorted(set(self.p_samples))

        return best[1]
    
    
    @staticmethod
    def _cg_coeff(j2: int, m2: int, ms2: int, J2: int) -> float:
        j = j2 / 2.0
        m = m2 / 2.0
        denom = 2.0 * j + 1.0

        if J2 == j2 + 1:  # J = j + 1/2
            if ms2 == +1:
                return math.sqrt((j + m + 1.0) / denom)
            else:  # ms2 == -1
                return math.sqrt((j - m + 1.0) / denom)

        if J2 == j2 - 1:  # J = j - 1/2
            if ms2 == +1:
                return -math.sqrt((j - m) / denom)
            else:  # ms2 == -1
                return math.sqrt((j + m) / denom)

        return 0.0

    @staticmethod
    def _schur_basis_qubits(n: int):
        if n < 1:
            raise ValueError("n must be >= 1")

        def j2_list(nq: int):
            return list(range(nq, nq % 2 - 1, -2))

        def mult(nq: int, j2: int) -> int:
            from math import comb
            k = (nq - j2) // 2
            return comb(nq, k) - (comb(nq, k - 1) if k - 1 >= 0 else 0)

        if n == 1:
            V = np.eye(2, dtype=complex)
            labels = [(1, 1, 0), (1, -1, 0)]  # |0> = m=+1/2, |1> = m=-1/2
            return V, labels

        V_prev, labels_prev = SolverSDPPerm._schur_basis_qubits(n - 1)
        d_prev = V_prev.shape[0]
        d = 2 * d_prev

        # map (j2,m2,alpha) -> vector in computational basis
        vec_prev = {(j2, m2, a): V_prev[:, k] for k, (j2, m2, a) in enumerate(labels_prev)}

        # multiplicities in previous step
        mult_prev = {}
        for (j2, m2, a) in labels_prev:
            mult_prev[j2] = max(mult_prev.get(j2, -1), a)
        for j2 in mult_prev:
            mult_prev[j2] += 1

        ket_up = np.array([1.0, 0.0], dtype=complex)  # |0> => ms2=+1
        ket_dn = np.array([0.0, 1.0], dtype=complex)  # |1> => ms2=-1

        vec_new = {}  # (J2,M2,alpha_new) -> vector
        alpha_counter = {J2: 0 for J2 in j2_list(n)}

        for j2 in j2_list(n - 1):
            for a in range(mult_prev.get(j2, 0)):
                # J2 = j2 + 1
                J2 = j2 + 1
                if J2 in alpha_counter:
                    alpha_new = alpha_counter[J2]
                    alpha_counter[J2] += 1
                    for M2 in range(J2, -J2 - 1, -2):
                        v = np.zeros(d, dtype=complex)
                        for ms2, ket in [(+1, ket_up), (-1, ket_dn)]:
                            m2 = M2 - ms2
                            if abs(m2) <= j2 and (j2, m2, a) in vec_prev:
                                c = SolverSDPPerm._cg_coeff(j2, m2, ms2, J2)
                                v += c * np.kron(vec_prev[(j2, m2, a)], ket)
                        vec_new[(J2, M2, alpha_new)] = v

                # J2 = j2 - 1
                J2 = j2 - 1
                if j2 > 0 and J2 in alpha_counter:
                    alpha_new = alpha_counter[J2]
                    alpha_counter[J2] += 1
                    for M2 in range(J2, -J2 - 1, -2):
                        v = np.zeros(d, dtype=complex)
                        for ms2, ket in [(+1, ket_up), (-1, ket_dn)]:
                            m2 = M2 - ms2
                            if abs(m2) <= j2 and (j2, m2, a) in vec_prev:
                                c = SolverSDPPerm._cg_coeff(j2, m2, ms2, J2)
                                v += c * np.kron(vec_prev[(j2, m2, a)], ket)
                        vec_new[(J2, M2, alpha_new)] = v

        # Canonical reorder: j2 desc, alpha asc, m2 desc
        cols = []
        labels = []
        for J2 in j2_list(n):
            mJ = mult(n, J2)
            for a in range(mJ):
                for M2 in range(J2, -J2 - 1, -2):
                    cols.append(vec_new[(J2, M2, a)])
                    labels.append((J2, M2, a))

        V = np.column_stack(cols)
        V = V / np.linalg.norm(V, axis=0, keepdims=True) # normalize
        
        return V, labels

    def _blocks_to_full_choi_schur(self, J_blocks: dict) -> np.ndarray:
        d_out = 2 ** self.n_out
        d_in = 2 ** self.n_in

        V_out, labels_out = self._schur_basis_qubits(self.n_out)
        V_in,  labels_in  = self._schur_basis_qubits(self.n_in)

        map_out = {lab: i for i, lab in enumerate(labels_out)}
        map_in  = {lab: i for i, lab in enumerate(labels_in)}

        J_s = np.zeros((d_out * d_in, d_out * d_in), dtype=complex)

        for j2o in self.j2_out_list:
            do = j2o + 1
            m2o_vals = [j2o - 2 * k for k in range(do)]  # m2 desc
            for j2i in self.j2_in_list:
                di = j2i + 1
                m2i_vals = [j2i - 2 * k for k in range(di)]
                J_red = J_blocks[(j2o, j2i)]  # shape (do*di, do*di)

                for ao in range(self.mult_out[j2o]):
                    out_idx = [map_out[(j2o, m2, ao)] for m2 in m2o_vals]
                    for ai in range(self.mult_in[j2i]):
                        in_idx = [map_in[(j2i, m2, ai)] for m2 in m2i_vals]

                        # global indices matching local ordering (out_m major, in_m minor)
                        glob = []
                        for o in out_idx:
                            base = o * d_in
                            for i_ in in_idx:
                                glob.append(base + i_)
                        glob = np.asarray(glob, dtype=int)

                        J_s[np.ix_(glob, glob)] += J_red

        return (J_s + J_s.conj().T) / 2.0

    def blocks_to_full_choi(self, J_blocks: dict) -> np.ndarray:
        d_out = 2 ** self.n_out
        d_in = 2 ** self.n_in

        V_out, _ = self._schur_basis_qubits(self.n_out)
        V_in,  _ = self._schur_basis_qubits(self.n_in)

        J_s = self._blocks_to_full_choi_schur(J_blocks)

        V_tot = np.kron(V_out, V_in)  # Schur -> computational
        J_full = V_tot @ J_s @ V_tot.conj().T
        return (J_full + J_full.conj().T) / 2.0

    def get_solution(self):
        J_blocks = self.solve()              # reduced dict
        J_full = self.blocks_to_full_choi(J_blocks)
        return J_full

    def get_solution_blocks(self):
        return self.solve()

class SolverSDPPermTwoPoint(SolverSDPPerm):
    def __init__(self, n_in, n_out, dim=2, verbose=False):
        super().__init__(n_in, n_out, dim, verbose, p_init_grid=2, p_fine_grid=1, n_rounds=1)


class SolverSDPPermSpectrum(SolverSDPPerm):
    def __init__(self, n_in, n_out, dim=2, verbose=False, spectrum=None):
        super().__init__(n_in, n_out, dim, verbose, p_init_grid=1, p_fine_grid=1, n_rounds=1)
        self.spectrum = spectrum

    # override
    def solve(self):
        self.p_samples = [self.spectrum]

        t_opt, J_opt = self.solve_one_round() # call self.make_problem() in this method; and self.make_problem() uses self.p_samples.
        _, root_curve, _ = self.fidelity_curve_blocks(J_opt, np.asarray([float(self.spectrum)], dtype=float))
        fvals = float(root_curve[0])
        best = (t_opt, J_opt, self.p_samples, self.p_samples[0], fvals)

        return best[1]


class SolverLocalIrrepSDP(SolverSDPPerm):
    def __init__(self, n_in, n_out, dim=2, verbose=False, p_init_grid=5, p_fine_grid=51, n_rounds=3, irrep_in=None, irrep_out=None):
        super().__init__(n_in=n_in, n_out=n_out, dim=dim, verbose=verbose, p_init_grid=p_init_grid, p_fine_grid=p_fine_grid, n_rounds=n_rounds)
        if dim != 2:
            raise ValueError("SolverLocalIrrepSDP currently supports only qubits (dim=2).")
        if irrep_in is None or irrep_out is None:
            raise ValueError("SolverLocalIrrepSDP requires both irrep_in and irrep_out.")

        self.j2_in, self.partition_in = parse_qubit_irrep_label(irrep_in, n_in)
        self.j2_out, self.partition_out = parse_qubit_irrep_label(irrep_out, n_out)
        self.local_d_in = self.j2_in + 1
        self.local_d_out = self.j2_out + 1
        self.local_projectors = list(self.P[(self.j2_out, self.j2_in)])
        self.local_basis = self._build_local_basis_channels()

        self._local_choi = None
        self._weights = None
        self._worst_p = None
        self._worst_root_fidelity = None
        self._fidelity_curve_p = None
        self._fidelity_curve_root = None
        self._sampled_root_lb = None

    def _build_local_basis_channels(self):
        basis = {}
        I_in = np.eye(self.local_d_in, dtype=complex)
        for (L2, Pi) in self.local_projectors:
            ptr = partial_trace_numpy(Pi, self.local_d_out, self.local_d_in, axis=0)
            alpha = float(np.trace(ptr).real) / self.local_d_in
            if alpha <= 0:
                raise RuntimeError(f"Invalid partial-trace coefficient for L2={L2}: alpha={alpha}")
            residual = np.linalg.norm(ptr - alpha * I_in)
            if residual > 1e-7:
                raise RuntimeError(
                    f"Projector for L2={L2} does not give scalar partial trace; residual={residual:.3e}"
                )
            basis[L2] = (Pi / alpha + (Pi / alpha).conj().T) / 2.0
        return basis

    def _local_input_state(self, p: float):
        try:
            rho_in = normalized_spin_irrep_state(self.n_in, self.j2_in, float(p))
            return rho_in
        except ValueError:
            return None

    def _local_output_state(self, p: float):
        try:
            rho_out = normalized_spin_irrep_state(self.n_out, self.j2_out, float(p))
            return rho_out
        except ValueError:
            return None
        # return normalized_spin_irrep_state(self.n_out, self.j2_out, float(p))

    def make_problem(self):
        q = {L2: cp.Variable(nonneg=True) for (L2, _) in self.local_projectors}
        t = cp.Variable()
        constraints = [t >= 0, t <= 1, cp.sum(list(q.values())) == 1]

        Jexpr = 0
        for (L2, _) in self.local_projectors:
            Jexpr += q[L2] * cp.Constant(self.local_basis[L2])

        constraints += [cp.partial_trace(Jexpr, (self.local_d_out, self.local_d_in), axis=0) == np.eye(self.local_d_in, dtype=complex)]

        for p in self.p_samples:
            rho_in = self._local_input_state(float(p))
            rho_out = self._local_output_state(float(p))
            if rho_out is None:
                continue
            M = cp.Constant(np.kron(np.eye(self.local_d_out, dtype=complex), rho_in.T)) @ Jexpr
            sigma = cp.partial_trace(M, (self.local_d_out, self.local_d_in), axis=1)
            sigma = (sigma + sigma.H) / 2

            X = cp.Variable((self.local_d_out, self.local_d_out), complex=True)
            block = cp.bmat([[cp.Constant(rho_out), X], [X.H, sigma]])
            constraints += [block >> 0]
            constraints += [cp.real(cp.trace(X)) >= t]

        prob = cp.Problem(cp.Maximize(t), constraints)
        return prob, q, t

    def solve_one_round(self, solver_preference=("MOSEK", "SCS")):
        prob, q, t = self.make_problem()

        chosen = None
        for s in solver_preference:
            if s in cp.installed_solvers():
                chosen = s
                break
        if chosen is None:
            raise RuntimeError("No suitable SDP solver found. Install one of MOSEK or SCS.")

        prob.solve(solver=chosen, verbose=self.verbose)
        if t.value is None:
            raise RuntimeError("Solver failed: t.value is None")

        weights = {}
        J = np.zeros((self.local_d_out * self.local_d_in, self.local_d_out * self.local_d_in), dtype=complex)
        for (L2, _) in self.local_projectors:
            val = q[L2].value
            if val is None:
                raise RuntimeError(f"Solver failed: q[{L2}] is None")
            weights[L2] = float(val)
            J += float(val) * self.local_basis[L2]
        J = (J + J.conj().T) / 2.0
        return float(t.value), weights, J

    def fidelity_curve(self, J: np.ndarray, p_grid=None):
        if p_grid is None:
            p_grid = np.linspace(0.5, 1.0, self.p_fine_grid)
        roots = []
        for p in p_grid:
            rho_in = self._local_input_state(float(p))
            rho_out = self._local_output_state(float(p))
            if rho_in is None or rho_out is None:
                continue
            sigma = apply_local_choi_numpy(J, rho_in)
            roots.append(fidelity_root_numpy(rho_out, sigma))
        return np.asarray(p_grid, dtype=float), np.asarray(roots, dtype=float)

    def solve(self):
        if self._local_choi is not None:
            return self._local_choi

        self.p_samples = sorted(set(np.linspace(0.5, 1.0, self.p_init_grid).tolist()))

        best_t = None
        best_weights = None
        best_J = None
        best_worst_p = None
        best_worst_root = None
        best_curve_p = None
        best_curve_root = None

        for it in range(self.n_rounds):
            t_opt, weights, J_opt = self.solve_one_round()
            p_curve, root_curve = self.fidelity_curve(J_opt)
            idx = int(np.argmin(root_curve))
            p_worst = float(p_curve[idx])
            root_worst = float(root_curve[idx])

            if self.verbose:
                print(
                    f"[Round {it}] local-irrep root-fidelity LB on samples = {t_opt:.8f} (F≈{t_opt**2:.8f})"
                )
                print(
                    f"           worst on fine grid: p={p_worst:.6f}, rootF={root_worst:.8f} (F≈{root_worst**2:.8f})"
                )
                print(
                    "           mixture weights: "
                    + ", ".join([f"L2={L2}:{weights[L2]:.6f}" for L2 in sorted(weights)])
                )

            best_t = t_opt
            best_weights = weights
            best_J = J_opt
            best_worst_p = p_worst
            best_worst_root = root_worst
            best_curve_p = p_curve
            best_curve_root = root_curve

            if min(abs(p_worst - np.asarray(self.p_samples))) < 1e-6:
                break
            self.p_samples.append(p_worst)
            self.p_samples = sorted(set(self.p_samples))

        self._sampled_root_lb = float(best_t)
        self._weights = dict(best_weights)
        self._local_choi = np.asarray(best_J, dtype=complex)
        self._worst_p = float(best_worst_p)
        self._worst_root_fidelity = float(best_worst_root)
        self._fidelity_curve_p = np.asarray(best_curve_p, dtype=float)
        self._fidelity_curve_root = np.asarray(best_curve_root, dtype=float)
        return self._local_choi

    def get_solution(self):
        return self.solve()

    def get_solution_blocks(self):
        self.solve()
        return np.asarray(self._local_choi, dtype=complex)

    def get_result(self):
        self.solve()
        return local_irrep_result_to_dict(self)

    def describe_result(self):
        result = self.get_result()
        lines = []
        lines.append(
            f"Local irrep solver result: input {result['partition_in']} (j2={result['j2_in']}) -> output {result['partition_out']} (j2={result['j2_out']})"
        )
        lines.append(f"Worst-case root fidelity ≈ {result['worst_root_fidelity']:.10f}")
        lines.append(f"Worst-case fidelity      ≈ {result['worst_fidelity']:.10f}")
        lines.append(f"Worst spectrum p         ≈ {result['worst_p']:.10f}")
        lines.append("Optimal covariant local map:")
        lines.append("  T = sum_L q_L T_L")
        for L2 in sorted(result['weights']):
            lines.append(f"    q_(L2={L2}) = {result['weights'][L2]:.10f}")
        return "\n".join(lines)

    def save_result(self, path: str):
        save_local_irrep_result(path, self.get_result(), self.get_solution_blocks())


class SolverGlobalIrrepSDP(SolverSDPPerm):
    def __init__(self, n_in, n_out, dim=2, verbose=False, p_init_grid=5, p_fine_grid=51, n_rounds=3, local_cache_dir="data/sdp_irrep", reuse_local_irrep=True):
        super().__init__(n_in=n_in, n_out=n_out, dim=dim, verbose=verbose, p_init_grid=p_init_grid, p_fine_grid=p_fine_grid, n_rounds=n_rounds)
        self.local_cache_dir = local_cache_dir
        self.reuse_local_irrep = reuse_local_irrep

        self.local_results = {}
        self._coefficients = None
        self._solution_blocks = None
        self._worst_p = None
        self._worst_root_fidelity = None
        self._sampled_root_lb = None

    def _cache_path_for_pair(self, j2_out: int, j2_in: int):
        part_in = str(j2_to_qubit_partition(self.n_in, j2_in)).replace(" ", "")
        part_out = str(j2_to_qubit_partition(self.n_out, j2_out)).replace(" ", "")
        filepath = f"{self.n_in}_to_{self.n_out}_{part_in}_to_{part_out}.npz"

        return os.path.join(self.local_cache_dir, filepath)

    def _local_cache_is_usable(self, result: dict, j2_out: int, j2_in: int):
        if result["n_in"] != self.n_in or result["n_out"] != self.n_out:
            return False
        if result["j2_in"] != j2_in or result["j2_out"] != j2_out:
            return False
        curve_p = np.asarray(result.get("fidelity_curve_p", np.array([], dtype=float)), dtype=float)
        return curve_p.size > 0 and abs(float(np.max(curve_p)) - 1.0) <= 1e-10

    def _compute_local_result(self, j2_out: int, j2_in: int):
        solver = SolverLocalIrrepSDP(
            n_in=self.n_in,
            n_out=self.n_out,
            dim=self.dim,
            verbose=self.verbose,
            p_init_grid=self.p_init_grid,
            p_fine_grid=self.p_fine_grid,
            n_rounds=self.n_rounds,
            irrep_in=j2_to_qubit_partition(self.n_in, j2_in),
            irrep_out=j2_to_qubit_partition(self.n_out, j2_out),
        )
        solver.solve()
        result = solver.get_result()
        if self.local_cache_dir is not None:
            save_local_irrep_result(self._cache_path_for_pair(j2_out, j2_in), result, solver.get_solution_blocks())
        
        result[block_key(j2_out, j2_in)] = solver.get_solution_blocks()
        return result

    def _get_local_result(self, j2_out: int, j2_in: int):
        key = (j2_out, j2_in)
        if key in self.local_results:
            return self.local_results[key]

        result = None
        if self.local_cache_dir is not None and self.reuse_local_irrep:
            path = self._cache_path_for_pair(j2_out, j2_in)
            if os.path.exists(path):
                try:
                    loaded = load_local_irrep_result(path)
                    if self._local_cache_is_usable(loaded, j2_out, j2_in):
                        result = loaded
                    elif self.verbose:
                        print(f"[sdp_irrep_global] Recomputing stale local cache: {path}")
                except Exception:
                    if self.verbose:
                        print(f"[sdp_irrep_global] Failed to load cache, recomputing: {path}")

        if result is None:
            result = self._compute_local_result(j2_out, j2_in)

        self.local_results[key] = result
        return result

    def _ensure_all_local_results(self):
        for j2_out in self.j2_out_list:
            for j2_in in self.j2_in_list:
                self._get_local_result(j2_out, j2_in)

    def save_local_results(self):
        self._ensure_all_local_results()
        if self.local_cache_dir is None:
            return
        os.makedirs(self.local_cache_dir, exist_ok=True)
        
        for (j2_out, j2_in), result in self.local_results.items():
            sol = result[block_key(j2_out, j2_in)]
            save_local_irrep_result(self._cache_path_for_pair(j2_out, j2_in), result, sol)

    def make_problem(self):
        self._ensure_all_local_results()

        t = cp.Variable()
        constraints = [t >= 0, t <= 1]
        coeff = {}
        Jexpr = {}

        for j2_out in self.j2_out_list:
            for j2_in in self.j2_in_list:
                y = cp.Variable(nonneg=True)
                coeff[(j2_out, j2_in)] = y
                J_loc = np.asarray(self.local_results[(j2_out, j2_in)][block_key(j2_out, j2_in)], dtype=complex)
                Jexpr[(j2_out, j2_in)] = y * cp.Constant(J_loc)

        for j2_in in self.j2_in_list:
            constraints += [cp.sum([coeff[(j2_out, j2_in)] for j2_out in self.j2_out_list]) == 1]

        for p in self.p_samples:
            rho_in = self._rho_blocks(self.n_in, float(p), self.j2_in_list)
            alpha_out = self._rho_blocks(self.n_out, float(p), self.j2_out_list)

            sigma = {}
            for j2_out in self.j2_out_list:
                d_out = j2_out + 1
                sig = 0
                for j2_in in self.j2_in_list:
                    d_in = j2_in + 1
                    K = np.kron(np.eye(d_out, dtype=complex), rho_in[j2_in].T)
                    M = cp.Constant(K) @ Jexpr[(j2_out, j2_in)]
                    sig_part = cp.partial_trace(M, (d_out, d_in), axis=1)
                    sig += self.mult_in[j2_in] * sig_part
                sigma[j2_out] = (sig + sig.H) / 2

            fid_sum = 0
            for j2_out in self.j2_out_list:
                d_out = j2_out + 1
                X = cp.Variable((d_out, d_out), complex=True)
                A = cp.Constant(alpha_out[j2_out] * self.mult_out[j2_out])
                block = cp.bmat([[A, X], [X.H, sigma[j2_out]]])
                constraints += [block >> 0]
                fid_sum += cp.real(cp.trace(X))

            constraints += [fid_sum >= t]

        prob = cp.Problem(cp.Maximize(t), constraints)
        return prob, coeff, t

    def solve_one_round(self, solver_preference=("MOSEK", "SCS")):
        prob, coeff, t = self.make_problem()

        chosen = None
        for s in solver_preference:
            if s in cp.installed_solvers():
                chosen = s
                break
        if chosen is None:
            raise RuntimeError("No suitable solver found. Install one of MOSEK or SCS (etc.).")

        prob.solve(solver=chosen, verbose=self.verbose)
        if t.value is None:
            raise RuntimeError("Solver failed: t.value is None")

        coeff_vals = {}
        J_blocks = {}
        for j2_out in self.j2_out_list:
            for j2_in in self.j2_in_list:
                y = coeff[(j2_out, j2_in)].value
                if y is None:
                    raise RuntimeError("Solver failed: some coefficient is None")
                coeff_vals[(j2_out, j2_in)] = float(y)
                J_loc = np.asarray(self.local_results[(j2_out, j2_in)][block_key(j2_out, j2_in)], dtype=complex)
                J_phys = (float(y) / self.mult_out[j2_out]) * J_loc
                J_blocks[(j2_out, j2_in)] = (J_phys + J_phys.conj().T) / 2.0
        return float(t.value), coeff_vals, J_blocks

    def solve(self):
        if self._solution_blocks is not None:
            return self._solution_blocks

        self._ensure_all_local_results()
        self.p_samples = sorted(set(np.linspace(0.5, 1.0, self.p_init_grid).tolist()))

        best = None
        for it in range(self.n_rounds):
            t_opt, coeff_opt, J_opt = self.solve_one_round()

            p_fine = np.linspace(0.5, 1.0, self.p_fine_grid)
            _, root_curve, _ = self.fidelity_curve_blocks(J_opt, p_fine)
            fvals = np.asarray(root_curve, dtype=float)

            idx = int(np.argmin(fvals))
            p_worst = float(p_fine[idx])
            f_worst = float(fvals[idx])

            if self.verbose:
                print(f"[Round {it}] global-irrep root-fidelity LB on samples = {t_opt:.8f} (F≈{t_opt**2:.8f})")
                print(f"         worst on fine grid: p={p_worst:.6f}, rootF={f_worst:.8f} (F≈{f_worst**2:.8f})")
                for j2_in in self.j2_in_list:
                    part_in = j2_to_qubit_partition(self.n_in, j2_in)
                    coeff_msg = []
                    for j2_out in self.j2_out_list:
                        part_out = j2_to_qubit_partition(self.n_out, j2_out)
                        coeff_msg.append(f"{part_out}:{coeff_opt[(j2_out, j2_in)]:.6f}")
                    print(f"         coeffs for input {part_in}: " + ", ".join(coeff_msg))

            best = (t_opt, coeff_opt, J_opt, p_worst, f_worst)
            if min(abs(p_worst - np.asarray(self.p_samples))) < 1e-6:
                break
            self.p_samples.append(p_worst)
            self.p_samples = sorted(set(self.p_samples))

        self._sampled_root_lb = float(best[0])
        self._coefficients = dict(best[1])
        self._solution_blocks = best[2]
        self._worst_p = float(best[3])
        self._worst_root_fidelity = float(best[4])
        return self._solution_blocks

    def get_solution_blocks(self):
        return self.solve()

    def get_solution(self):
        if self._solution_blocks is None:
            self.solve()
        J_full = self.blocks_to_full_choi(self._solution_blocks)
        return J_full

    def get_coefficients(self):
        if self._coefficients is None:
            self.solve()
        return dict(self._coefficients)

    def get_coefficients_by_partition(self):
        coeff = self.get_coefficients()
        out = {}
        for (j2_out, j2_in), val in coeff.items():
            out[(j2_to_qubit_partition(self.n_out, j2_out), j2_to_qubit_partition(self.n_in, j2_in))] = float(val)
        return out

    def describe_result(self):
        self.get_solution_blocks()
        lines = []
        lines.append(f"Global irrep-assembled solver result: N={self.n_in} -> M={self.n_out}")
        lines.append(f"Worst-case root fidelity ≈ {self._worst_root_fidelity:.10f}")
        lines.append(f"Worst-case fidelity      ≈ {self._worst_root_fidelity**2:.10f}")
        lines.append(f"Worst spectrum p         ≈ {self._worst_p:.10f}")
        if self._sampled_root_lb is not None:
            lines.append(f"Sampled-grid root-F LB   ≈ {self._sampled_root_lb:.10f}")
        lines.append("Assembly coefficients (sum over output irreps = 1 for each input irrep):")
        for j2_in in self.j2_in_list:
            part_in = j2_to_qubit_partition(self.n_in, j2_in)
            lines.append(f"  input {part_in}:")
            for j2_out in self.j2_out_list:
                part_out = j2_to_qubit_partition(self.n_out, j2_out)
                lines.append(f"    coeff[{part_out}] = {self.get_coefficients()[(j2_out, j2_in)]:.10f}")
        return "\n".join(lines)

    def save_result(self, path: str):
        self.get_solution_blocks()
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        save_dict = {
            "worst_p": np.array(self._worst_p, dtype=float),
            "worst_fidelity": np.array(self._worst_root_fidelity ** 2, dtype=float),
            "sampled_fidelity": np.array(self._sampled_root_lb ** 2, dtype=float),
        }   
        for k, J in self._solution_blocks.items():
            save_dict[block_key(*k)] = np.asarray(J, dtype=complex)
            save_dict[f"c_{k}"] = np.array(self._coefficients[k], dtype=float)
        np.savez_compressed(path, **save_dict)


class SolverFixedIrrepFamily(SolverLocalIrrepSDP):
    """
    For a fixed irrep pair lambda -> mu, build every fixed auxiliary *local SU(2)
    sector* available in V_mu \\otimes V_lambda^*. Each sector is indexed by j2_nu,
    i.e. by the projector label L2 from su2_commutant_projectors().

    Important:
    This class stores the exact reduced/local ansatz used by the optimizer
    (J_nu proportional to the corresponding projector sector), not the literal
    full Schur-space map Pi_mu (X \\otimes Pi_nu) Pi_mu including multiplicity-space
    factors. In particular, auxiliary labels are no longer restricted to partitions
    of n_out - n_in boxes. When that physical-size partition exists, it is exposed
    in the metadata as physical_partition_nu; otherwise a canonical representative
    partition_nu = (j2_nu,) is used.
    """
    def __init__(self, n_in, n_out, dim=2, verbose=False, p_init_grid=5, p_fine_grid=101, n_rounds=1, irrep_in=None, irrep_out=None, cache_dir="data/fixed_irrep"):
        super().__init__(n_in=n_in, n_out=n_out, dim=dim, verbose=verbose, p_init_grid=p_init_grid, p_fine_grid=p_fine_grid, n_rounds=n_rounds, irrep_in=irrep_in, irrep_out=irrep_out)

        self.cache_dir = cache_dir
        self.n_anc = self.n_out - self.n_in

        self.j2_nu_list = [int(L2) for (L2, _) in self.local_projectors]
        self.partition_nu_map = {int(j2_nu): tuple(canonical_qubit_aux_partition(j2_nu)) for j2_nu in self.j2_nu_list}
        self.physical_partition_nu_map = {}
        for j2_nu in self.j2_nu_list:
            if j2_nu in j2_list_for_n_qubits(self.n_anc):
                self.physical_partition_nu_map[int(j2_nu)] = tuple(j2_to_qubit_partition(self.n_anc, j2_nu))
            else:
                self.physical_partition_nu_map[int(j2_nu)] = None

        self._results_by_j2_nu = None
        self._blocks_by_j2_nu = None

    def _normalize_nu_label(self, irrep_nu):
        """Accept either j2_nu (int) or an arbitrary qubit partition label."""
        if isinstance(irrep_nu, (int, np.integer)):
            j2_nu = int(irrep_nu)
        else:
            j2_nu, _ = parse_auxiliary_qubit_irrep_label(irrep_nu)

        if j2_nu not in self.j2_nu_list:
            raise ValueError(
                f"nu={irrep_nu} is not admissible for "
                f"{self.partition_in} -> {self.partition_out}. "
                f"Allowed j2_nu values: {self.j2_nu_list}"
            )
        return j2_nu

    def admissible_nu_labels(self):
        return [
            {
                "j2_nu": int(j2_nu),
                "partition_nu": tuple(self.partition_nu_map[j2_nu]),
                "physical_partition_nu": self.physical_partition_nu_map[j2_nu],
                "matches_physical_ancilla": self.physical_partition_nu_map[j2_nu] is not None,
            }
            for j2_nu in self.j2_nu_list
        ]

    def _cache_path_for_nu(self, j2_nu: int):
        part_in = str(tuple(self.partition_in)).replace(" ", "")
        part_out = str(tuple(self.partition_out)).replace(" ", "")
        part_nu = str(tuple(self.partition_nu_map[j2_nu])).replace(" ", "")
        filename = f"{self.n_in}_to_{self.n_out}_{part_in}_to_{part_out}_nu_{part_nu}.npz"
        return os.path.join(self.cache_dir, filename)

    def _result_dict_for_fixed_nu(self, j2_nu: int):
        fixed_data = fixed_irrep_channel_local_choi(
            n_in=self.n_in,
            n_out=self.n_out,
            irrep_in=self.partition_in,
            irrep_out=self.partition_out,
            irrep_nu=self.partition_nu_map[j2_nu],
        )
        J = np.asarray(fixed_data["local_choi"], dtype=complex)

        self.p_samples = sorted(set(np.linspace(0.5, 1.0, self.p_init_grid).tolist()))
        sampled_roots = []
        for p in self.p_samples:
            rho_in = self._local_input_state(float(p))
            rho_out = self._local_output_state(float(p))
            if rho_in is None or rho_out is None:
                continue
            sigma = apply_local_choi_numpy(J, rho_in)
            sampled_roots.append(fidelity_root_numpy(rho_out, sigma))

        sampled_root_lb = 0.0 if len(sampled_roots) == 0 else float(np.min(np.asarray(sampled_roots, dtype=float)))

        p_curve, root_curve = self.fidelity_curve(J, p_grid=np.linspace(0.5, 1.0, self.p_fine_grid))
        idx = int(np.argmin(root_curve))
        worst_p = float(p_curve[idx])
        worst_root = float(root_curve[idx])

        weights = {int(L2): 1.0 if int(L2) == int(j2_nu) else 0.0 for L2 in self.local_basis.keys()}

        result = {
            "n_in": int(self.n_in),
            "n_out": int(self.n_out),
            "j2_in": int(self.j2_in),
            "j2_out": int(self.j2_out),
            "partition_in": tuple(self.partition_in),
            "partition_out": tuple(self.partition_out),
            "j2_nu": int(j2_nu),
            "partition_nu": tuple(fixed_data["partition_nu"]),
            "physical_partition_nu": fixed_data.get("physical_partition_nu"),
            "nu_matches_physical_ancilla": bool(fixed_data.get("nu_matches_physical_ancilla", False)),
            "worst_p": float(worst_p),
            "worst_fidelity": float(worst_root ** 2),
            "sampled_fidelity": float(sampled_root_lb ** 2),
            "weights": weights,
            "local_choi_basis": {int(L2): np.asarray(B, dtype=complex) for L2, B in self.local_basis.items()},
            "fidelity_curve_p": np.asarray(p_curve, dtype=float),
            "fidelity_curve_root": np.asarray(root_curve, dtype=float),
        }
        result[block_key(self.j2_out, self.j2_in)] = J
        return result

    def solve(self):
        """Build all admissible fixed-nu local sectors; returns j2_nu -> local block."""
        if self._blocks_by_j2_nu is not None:
            return self._blocks_by_j2_nu

        results = {}
        blocks = {}

        for j2_nu in self.j2_nu_list:
            result = self._result_dict_for_fixed_nu(j2_nu)
            results[int(j2_nu)] = result
            blocks[int(j2_nu)] = {
                (self.j2_out, self.j2_in): np.asarray(result[block_key(self.j2_out, self.j2_in)], dtype=complex)
            }

            if self.verbose:
                phys = result.get("physical_partition_nu")
                phys_msg = f", physical_nu={phys}" if phys is not None else ", physical_nu=None"
                print(
                    f"[fixed nu] {self.partition_in} -> {self.partition_out}, "
                    f"nu={result['partition_nu']} (j2={j2_nu}){phys_msg}, "
                    f"worst F≈{result['worst_fidelity']:.8f}"
                )

        self._results_by_j2_nu = results
        self._blocks_by_j2_nu = blocks
        return self._blocks_by_j2_nu

    def get_solution(self):
        return self.solve()

    def get_solution_blocks(self):
        return self.solve()

    def get_result(self, irrep_nu):
        self.solve()
        j2_nu = self._normalize_nu_label(irrep_nu)
        return self._results_by_j2_nu[j2_nu]

    def get_all_results(self):
        self.solve()
        return self._results_by_j2_nu

    def get_result_by_partition_nu(self):
        self.solve()
        return {
            tuple(self.partition_nu_map[j2_nu]): result
            for j2_nu, result in self._results_by_j2_nu.items()
        }

    def save_result(self, irrep_nu, path=None):
        self.solve()
        j2_nu = self._normalize_nu_label(irrep_nu)
        result = self._results_by_j2_nu[j2_nu]
        if path is None:
            path = self._cache_path_for_nu(j2_nu)
        save_local_irrep_result(path, result, result[block_key(self.j2_out, self.j2_in)])
        return path

    def save_all_results(self):
        self.solve()
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

        saved = {}
        for j2_nu in self.j2_nu_list:
            path = self._cache_path_for_nu(j2_nu)
            save_local_irrep_result(path, self._results_by_j2_nu[j2_nu], self._results_by_j2_nu[j2_nu][block_key(self.j2_out, self.j2_in)])
            saved[j2_nu] = path
        return saved

    def describe_results(self):
        self.solve()
        lines = []
        lines.append(
            f"Fixed-nu local-sector family for input {self.partition_in} (j2={self.j2_in}) "
            f"-> output {self.partition_out} (j2={self.j2_out})"
        )
        for j2_nu in self.j2_nu_list:
            result = self._results_by_j2_nu[j2_nu]
            lines.append(
                f"  nu={result['partition_nu']} (j2={j2_nu}, physical={result.get('physical_partition_nu')}): "
                f"worst_F≈{result['worst_fidelity']:.10f}, "
                f"worst_p≈{result['worst_p']:.10f}"
            )
        return "\n".join(lines)


def save_all_fixed_irrep_families(
    n_in,
    n_out,
    dim=2,
    verbose=False,
    p_init_grid=5,
    p_fine_grid=101,
    n_rounds=1,
    cache_dir="data/fixed_irrep",
    skip_empty=True,
):
    """
    For all input/output qubit irreps for (n_in, n_out), build SolverFixedIrrepFamily
    and save every admissible fixed-nu map.

    Returns
    -------
    summary : dict
        {
            "saved_files": [paths...],
            "saved_pairs": [((part_in), (part_out))...],
            "skipped_pairs": [((part_in), (part_out), reason)...],
        }
    """
    os.makedirs(cache_dir, exist_ok=True)

    input_parts = [tuple(j2_to_qubit_partition(n_in, j2)) for j2 in j2_list_for_n_qubits(n_in)]
    output_parts = [tuple(j2_to_qubit_partition(n_out, j2)) for j2 in j2_list_for_n_qubits(n_out)]

    saved_files = []
    saved_pairs = []
    skipped_pairs = []

    for part_in in input_parts:
        for part_out in output_parts:
            try:
                solver = SolverFixedIrrepFamily(
                    n_in=n_in,
                    n_out=n_out,
                    dim=dim,
                    verbose=verbose,
                    p_init_grid=p_init_grid,
                    p_fine_grid=p_fine_grid,
                    n_rounds=n_rounds,
                    irrep_in=part_in,
                    irrep_out=part_out,
                    cache_dir=cache_dir,
                )

                admissible = solver.admissible_nu_labels()
                if len(admissible) == 0:
                    msg = "no admissible nu"
                    skipped_pairs.append((part_in, part_out, msg))
                    if verbose:
                        print(f"[skip] {part_in} -> {part_out}: {msg}")
                    if skip_empty:
                        continue

                saved = solver.save_all_results()
                saved_paths = list(saved.values())

                saved_files.extend(saved_paths)
                saved_pairs.append((part_in, part_out))

                if verbose:
                    print(
                        f"[saved] {part_in} -> {part_out}: "
                        f"{len(saved_paths)} file(s), nus={admissible}"
                    )

            except Exception as e:
                skipped_pairs.append((part_in, part_out, str(e)))
                print(f"[error] {part_in} -> {part_out}: {e}")

    return {
        "saved_files": saved_files,
        "saved_pairs": saved_pairs,
        "skipped_pairs": skipped_pairs,
    }


class ExactFormulaFixedNuChannel:
    r"""
    Exact implementation of

        T_{lambda -> mu}^nu(X)
          = [dim V_lambda / (c_{lambda,nu}^mu dim Sp_nu dim V_mu)]
            Pi_mu (X \otimes Pi_nu) Pi_mu

    in the qubit Schur-Weyl setting.

    Internally this class constructs the exact channel on the full sector spaces

        H_lambda = Sp_lambda \otimes V_lambda,
        H_nu     = Sp_nu \otimes V_nu,
        H_mu     = Sp_mu \otimes V_mu,

    but the object intended for downstream analysis is the induced/effective local
    channel on spin spaces only, obtained by averaging over Sp_lambda and tracing
    out Sp_mu. The block stored under block_key(j2_out, j2_in) is therefore the
    Choi matrix on V_mu \otimes V_lambda, matching the convention used elsewhere
    in this optimizer.
    """

    def __init__(self, n_in, n_out, irrep_in, irrep_out, irrep_nu, verbose=False):
        if n_out < n_in:
            raise ValueError("Need n_out >= n_in.")
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.n_anc = int(n_out - n_in)
        self.verbose = bool(verbose)

        self.j2_in, self.partition_in = self._parse_partition(irrep_in, self.n_in)
        self.j2_out, self.partition_out = self._parse_partition(irrep_out, self.n_out)
        self.j2_nu, self.partition_nu = self._parse_partition(irrep_nu, self.n_anc)

        self.dim_Sp_in = mult_qubits(self.n_in, self.j2_in)
        self.dim_Sp_out = mult_qubits(self.n_out, self.j2_out)
        self.dim_Sp_nu = mult_qubits(self.n_anc, self.j2_nu)

        self.dim_V_in = self.j2_in + 1
        self.dim_V_out = self.j2_out + 1
        self.dim_V_nu = self.j2_nu + 1

        self.dim_sector_in = self.dim_Sp_in * self.dim_V_in
        self.dim_sector_out = self.dim_Sp_out * self.dim_V_out
        self.dim_sector_nu = self.dim_Sp_nu * self.dim_V_nu

        self.lr_coeff = int(lr_coeff_qubit_irreps(self.partition_in, self.partition_nu, self.partition_out))
        if self.lr_coeff == 0:
            raise ValueError(
                "LR coefficient is zero for "
                f"lambda={self.partition_in}, nu={self.partition_nu}, mu={self.partition_out}."
            )

        self.prefactor = float(self.dim_V_in) / float(self.lr_coeff * self.dim_Sp_nu * self.dim_V_out)
        self.physical_partition_nu = tuple(self.partition_nu)

        self._basis_in = None
        self._basis_out = None
        self._basis_nu = None
        self._sector_embedding = None
        self._sector_projector = None
        self._kraus = None
        self._sector_choi = None
        self._full_choi = None
        self._effective_local_kraus = None
        self._effective_local_choi = None

    @staticmethod
    def _parse_partition(label, n):
        if n == 0:
            if label in [(), [], "()", "empty", "trivial", 0, "0", "j2=0"]:
                return 0, ()
            raise ValueError("For n=0 ancilla, the only allowed irrep label is the trivial one.")
        return parse_qubit_irrep_label(label, n)

    @staticmethod
    def _cg_coeff(j2: int, m2: int, ms2: int, J2: int) -> float:
        j = j2 / 2.0
        m = m2 / 2.0
        denom = 2.0 * j + 1.0
        if J2 == j2 + 1:
            return math.sqrt((j + m + 1.0) / denom) if ms2 == +1 else math.sqrt((j - m + 1.0) / denom)
        if J2 == j2 - 1:
            return -math.sqrt((j - m) / denom) if ms2 == +1 else math.sqrt((j + m) / denom)
        return 0.0

    @classmethod
    def _schur_basis_qubits(cls, n: int):
        if n < 1:
            raise ValueError("n must be >= 1")

        def j2_list(nq: int):
            return list(range(nq, nq % 2 - 1, -2))

        def mult(nq: int, j2: int) -> int:
            from math import comb
            k = (nq - j2) // 2
            return comb(nq, k) - (comb(nq, k - 1) if k - 1 >= 0 else 0)

        if n == 1:
            V = np.eye(2, dtype=complex)
            labels = [(1, 1, 0), (1, -1, 0)]
            return V, labels

        V_prev, labels_prev = cls._schur_basis_qubits(n - 1)
        d_prev = V_prev.shape[0]
        d = 2 * d_prev
        vec_prev = {(j2, m2, a): V_prev[:, k] for k, (j2, m2, a) in enumerate(labels_prev)}

        mult_prev = {}
        for (j2, m2, a) in labels_prev:
            mult_prev[j2] = max(mult_prev.get(j2, -1), a)
        for j2 in mult_prev:
            mult_prev[j2] += 1

        ket_up = np.array([1.0, 0.0], dtype=complex)
        ket_dn = np.array([0.0, 1.0], dtype=complex)
        vec_new = {}
        alpha_counter = {J2: 0 for J2 in j2_list(n)}

        for j2 in j2_list(n - 1):
            for a in range(mult_prev.get(j2, 0)):
                for J2 in [j2 + 1, j2 - 1]:
                    if J2 not in alpha_counter:
                        continue
                    if J2 == j2 - 1 and j2 <= 0:
                        continue
                    alpha_new = alpha_counter[J2]
                    alpha_counter[J2] += 1
                    for M2 in range(J2, -J2 - 1, -2):
                        v = np.zeros(d, dtype=complex)
                        for ms2, ket in [(+1, ket_up), (-1, ket_dn)]:
                            m2 = M2 - ms2
                            if abs(m2) <= j2 and (j2, m2, a) in vec_prev:
                                c = cls._cg_coeff(j2, m2, ms2, J2)
                                v += c * np.kron(vec_prev[(j2, m2, a)], ket)
                        vec_new[(J2, M2, alpha_new)] = v

        cols, labels = [], []
        for J2 in j2_list(n):
            mJ = mult(n, J2)
            for a in range(mJ):
                for M2 in range(J2, -J2 - 1, -2):
                    cols.append(vec_new[(J2, M2, a)])
                    labels.append((J2, M2, a))

        V = np.column_stack(cols)
        V = V / np.linalg.norm(V, axis=0, keepdims=True)
        return V, labels

    @classmethod
    def _sector_basis_matrix(cls, n: int, j2: int):
        if n == 0:
            if j2 != 0:
                raise ValueError("For n=0 only j2=0 is allowed.")
            return np.ones((1, 1), dtype=complex)
        V, labels = cls._schur_basis_qubits(n)
        cols = [V[:, k] for k, (J2, _, _) in enumerate(labels) if J2 == j2]
        if not cols:
            raise ValueError(f"No Schur basis columns found for n={n}, j2={j2}.")
        return np.column_stack(cols) + 0.0j

    def sector_basis_input(self):
        if self._basis_in is None:
            self._basis_in = self._sector_basis_matrix(self.n_in, self.j2_in)
        return self._basis_in

    def sector_basis_output(self):
        if self._basis_out is None:
            self._basis_out = self._sector_basis_matrix(self.n_out, self.j2_out)
        return self._basis_out

    def sector_basis_nu(self):
        if self._basis_nu is None:
            self._basis_nu = self._sector_basis_matrix(self.n_anc, self.j2_nu)
        return self._basis_nu

    def sector_intertwiner(self):
        if self._sector_embedding is None:
            B_in = self.sector_basis_input()
            B_nu = self.sector_basis_nu()
            B_out = self.sector_basis_output()
            self._sector_embedding = B_out.conj().T @ np.kron(B_in, B_nu)
        return self._sector_embedding

    def sector_projector_in_lambda_tensor_nu(self):
        if self._sector_projector is None:
            C = self.sector_intertwiner()
            P = C.conj().T @ C
            self._sector_projector = (P + P.conj().T) / 2.0
        return self._sector_projector

    def kraus_operators(self):
        if self._kraus is None:
            C = self.sector_intertwiner()
            d_in = self.dim_sector_in
            d_nu = self.dim_sector_nu
            K_list = []
            for a in range(d_nu):
                e = np.zeros((d_nu, 1), dtype=complex)
                e[a, 0] = 1.0
                K = math.sqrt(self.prefactor) * (C @ np.kron(np.eye(d_in, dtype=complex), e))
                K_list.append(np.asarray(K, dtype=complex))
            self._kraus = K_list
        return self._kraus

    def apply(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=complex)
        if X.shape != (self.dim_sector_in, self.dim_sector_in):
            raise ValueError(f"Input X must have shape {(self.dim_sector_in, self.dim_sector_in)}, got {X.shape}.")
        out = np.zeros((self.dim_sector_out, self.dim_sector_out), dtype=complex)
        for K in self.kraus_operators():
            out += K @ X @ K.conj().T
        return (out + out.conj().T) / 2.0

    def sector_choi(self) -> np.ndarray:
        if self._sector_choi is None:
            d_out = self.dim_sector_out
            d_in = self.dim_sector_in
            J = np.zeros((d_out * d_in, d_out * d_in), dtype=complex)
            for K in self.kraus_operators():
                vecK = K.reshape(d_out * d_in, order="C")
                J += np.outer(vecK, vecK.conj())
            self._sector_choi = (J + J.conj().T) / 2.0
        return self._sector_choi

    def full_choi_computational(self) -> np.ndarray:
        if self._full_choi is None:
            B_out = self.sector_basis_output()
            B_in = self.sector_basis_input()
            W = np.kron(B_out, B_in)
            J_full = W @ self.sector_choi() @ W.conj().T
            self._full_choi = (J_full + J_full.conj().T) / 2.0
        return self._full_choi

    def effective_local_kraus_operators(self):
        if self._effective_local_kraus is None:
            dS_out, dV_out = self.dim_Sp_out, self.dim_V_out
            dS_in, dV_in = self.dim_Sp_in, self.dim_V_in
            scale = 1.0 / math.sqrt(dS_in)
            ops = []
            for K in self.kraus_operators():
                K4 = np.asarray(K, dtype=complex).reshape(dS_out, dV_out, dS_in, dV_in)
                for s_out in range(dS_out):
                    for s_in in range(dS_in):
                        L = scale * np.asarray(K4[s_out, :, s_in, :], dtype=complex)
                        if np.linalg.norm(L) > 1e-14:
                            ops.append(L.copy())
            self._effective_local_kraus = ops
        return self._effective_local_kraus

    def apply_effective_local(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=complex)
        if X.shape != (self.dim_V_in, self.dim_V_in):
            raise ValueError(f"Local input X must have shape {(self.dim_V_in, self.dim_V_in)}, got {X.shape}.")
        out = np.zeros((self.dim_V_out, self.dim_V_out), dtype=complex)
        for L in self.effective_local_kraus_operators():
            out += L @ X @ L.conj().T
        return (out + out.conj().T) / 2.0

    def effective_local_block_choi(self) -> np.ndarray:
        if self._effective_local_choi is None:
            d_out, d_in = self.dim_V_out, self.dim_V_in
            J = np.zeros((d_out * d_in, d_out * d_in), dtype=complex)
            for L in self.effective_local_kraus_operators():
                vecL = np.asarray(L, dtype=complex).reshape(d_out * d_in, order="C")
                J += np.outer(vecL, vecL.conj())
            self._effective_local_choi = (J + J.conj().T) / 2.0
        return self._effective_local_choi

    def verify_tp(self, tol=1e-8):
        J = self.sector_choi()
        d_out, d_in = self.dim_sector_out, self.dim_sector_in
        ptr = np.einsum("o i o j -> i j", J.reshape(d_out, d_in, d_out, d_in))
        err = np.linalg.norm(ptr - np.eye(d_in, dtype=complex))
        return err <= tol, float(err)

    def verify_effective_local_tp(self, tol=1e-8):
        J = self.effective_local_block_choi()
        d_out, d_in = self.dim_V_out, self.dim_V_in
        ptr = np.einsum("o i o j -> i j", J.reshape(d_out, d_in, d_out, d_in))
        err = np.linalg.norm(ptr - np.eye(d_in, dtype=complex))
        return err <= tol, float(err)

    def normalized_local_input_state(self, p: float) -> np.ndarray:
        rho_spin = rho_block_diag_in_spin_irrep(self.n_in, self.j2_in, float(p))
        z = float(np.trace(rho_spin).real)
        if z <= 0:
            return None
        rho_spin = rho_spin / z
        return np.kron(np.eye(self.dim_Sp_in, dtype=complex) / self.dim_Sp_in, rho_spin)

    def normalized_local_target_state(self, p: float) -> np.ndarray:
        rho_spin = rho_block_diag_in_spin_irrep(self.n_out, self.j2_out, float(p))
        z = float(np.trace(rho_spin).real)
        if z <= 0:
            return None
        rho_spin = rho_spin / z
        return np.kron(np.eye(self.dim_Sp_out, dtype=complex) / self.dim_Sp_out, rho_spin)

    def normalized_effective_local_input_state(self, p: float) -> np.ndarray:
        rho_spin = rho_block_diag_in_spin_irrep(self.n_in, self.j2_in, float(p))
        z = float(np.trace(rho_spin).real)
        if z <= 0:
            return None
        return rho_spin / z

    def normalized_effective_local_target_state(self, p: float) -> np.ndarray:
        rho_spin = rho_block_diag_in_spin_irrep(self.n_out, self.j2_out, float(p))
        z = float(np.trace(rho_spin).real)
        if z <= 0:
            return None
        return rho_spin / z

    def fidelity_curve(self, p_grid):
        p_grid = np.asarray(p_grid, dtype=float)
        roots = []
        kept = []
        for p in p_grid:
            rho_in = self.normalized_local_input_state(float(p))
            rho_out = self.normalized_local_target_state(float(p))
            if rho_in is None or rho_out is None:
                continue
            sigma = self.apply(rho_in)
            roots.append(fidelity_root_numpy(rho_out, sigma))
            kept.append(float(p))
        return np.asarray(kept, dtype=float), np.asarray(roots, dtype=float)

    def effective_local_fidelity_curve(self, p_grid):
        p_grid = np.asarray(p_grid, dtype=float)
        roots = []
        kept = []
        for p in p_grid:
            rho_in = self.normalized_effective_local_input_state(float(p))
            rho_out = self.normalized_effective_local_target_state(float(p))
            if rho_in is None or rho_out is None:
                continue
            sigma = self.apply_effective_local(rho_in)
            roots.append(fidelity_root_numpy(rho_out, sigma))
            kept.append(float(p))
        return np.asarray(kept, dtype=float), np.asarray(roots, dtype=float)

    def describe(self) -> str:
        ok, err = self.verify_effective_local_tp()
        lines = [
            f"Exact formula channel: lambda={self.partition_in} -> mu={self.partition_out}, nu={self.partition_nu}",
            "prefactor = dim(V_lambda) / (c_{lambda,nu}^mu dim(Sp_nu) dim(V_mu)) "
            f"= {self.dim_V_in} / ({self.lr_coeff} * {self.dim_Sp_nu} * {self.dim_V_out}) = {self.prefactor:.10f}",
            f"effective local dims: d_in={self.dim_V_in}, d_out={self.dim_V_out}; full sector dims: in={self.dim_sector_in}, nu={self.dim_sector_nu}, out={self.dim_sector_out}",
            f"Effective-local TP check: {ok} (error {err:.3e})",
        ]
        return "\n".join(lines)

    def get_result(self):
        ok, err = self.verify_effective_local_tp()
        result = {
            "n_in": int(self.n_in),
            "n_out": int(self.n_out),
            "n_anc": int(self.n_anc),
            "j2_in": int(self.j2_in),
            "j2_out": int(self.j2_out),
            "j2_nu": int(self.j2_nu),
            "partition_in": tuple(self.partition_in),
            "partition_out": tuple(self.partition_out),
            "partition_nu": tuple(self.partition_nu),
            "physical_partition_nu": tuple(self.partition_nu),
            "nu_matches_physical_ancilla": True,
            "dim_Sp_in": int(self.dim_Sp_in),
            "dim_Sp_out": int(self.dim_Sp_out),
            "dim_Sp_nu": int(self.dim_Sp_nu),
            "dim_V_in": int(self.dim_V_in),
            "dim_V_out": int(self.dim_V_out),
            "dim_V_nu": int(self.dim_V_nu),
            "dim_sector_in": int(self.dim_sector_in),
            "dim_sector_out": int(self.dim_sector_out),
            "dim_sector_nu": int(self.dim_sector_nu),
            "lr_coeff": int(self.lr_coeff),
            "prefactor": float(self.prefactor),
            "tp_ok": bool(ok),
            "tp_error": float(err),
        }
        result[block_key(self.j2_out, self.j2_in)] = np.asarray(self.effective_local_block_choi(), dtype=complex)
        return result

    def save_result(self, path: str):
        result = self.get_result()
        save_local_irrep_result(path, result, result[block_key(self.j2_out, self.j2_in)])
        return path


class ExactFormulaIrrepFamily:
    """Build ExactFormulaFixedNuChannel for every physical nu with c_{lambda,nu}^mu != 0."""

    def __init__(self, n_in, n_out, irrep_in, irrep_out, verbose=False):
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.n_anc = int(n_out - n_in)
        self.verbose = bool(verbose)
        self.j2_in, self.partition_in = parse_qubit_irrep_label(irrep_in, self.n_in)
        self.j2_out, self.partition_out = parse_qubit_irrep_label(irrep_out, self.n_out)
        self.nu_partitions = []
        for j2_nu in j2_list_for_n_qubits(self.n_anc):
            part_nu = tuple(j2_to_qubit_partition(self.n_anc, j2_nu))
            c = lr_coeff_qubit_irreps(self.partition_in, part_nu, self.partition_out)
            if c != 0:
                self.nu_partitions.append(part_nu)
        self._channels = None

    def build(self):
        if self._channels is None:
            self._channels = {}
            for part_nu in self.nu_partitions:
                self._channels[tuple(part_nu)] = ExactFormulaFixedNuChannel(
                    n_in=self.n_in, n_out=self.n_out,
                    irrep_in=self.partition_in, irrep_out=self.partition_out,
                    irrep_nu=part_nu, verbose=self.verbose,
                )
        return self._channels

    def get_channel(self, irrep_nu):
        channels = self.build()
        _, part_nu = ExactFormulaFixedNuChannel._parse_partition(irrep_nu, self.n_anc)
        return channels[tuple(part_nu)]

    def get_all_results(self):
        return {part_nu: ch.get_result() for part_nu, ch in self.build().items()}


class AbstractSpinOnlyExactFormulaFixedNuChannel:
    r"""
    Abstract exact spin-sector channel for arbitrary auxiliary SU(2) irrep label nu.

    This fallback is used when the input/output multiplicity spaces are both trivial
    (dim Sp_lambda = dim Sp_mu = 1), so that the exact formula reduces to the pure
    SU(2)-spin part. In that regime we can allow non-physical auxiliary labels nu,
    represented canonically by one-row partitions (j2_nu,).
    """
    def __init__(self, n_in, n_out, irrep_in, irrep_out, irrep_nu, verbose=False):
        if n_out < n_in:
            raise ValueError("Need n_out >= n_in.")
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.n_anc = int(n_out - n_in)
        self.verbose = bool(verbose)
        self.j2_in, self.partition_in = parse_qubit_irrep_label(irrep_in, self.n_in)
        self.j2_out, self.partition_out = parse_qubit_irrep_label(irrep_out, self.n_out)
        self.j2_nu, self.partition_nu = parse_auxiliary_qubit_irrep_label(irrep_nu)
        self.dim_Sp_in = mult_qubits(self.n_in, self.j2_in)
        self.dim_Sp_out = mult_qubits(self.n_out, self.j2_out)
        if self.dim_Sp_in != 1 or self.dim_Sp_out != 1:
            raise ValueError("AbstractSpinOnlyExactFormulaFixedNuChannel currently requires dim Sp_lambda = dim Sp_mu = 1.")
        self.dim_Sp_nu = 1
        self.dim_V_in = self.j2_in + 1
        self.dim_V_out = self.j2_out + 1
        self.dim_V_nu = self.j2_nu + 1
        self.dim_sector_in = self.dim_V_in
        self.dim_sector_out = self.dim_V_out
        self.dim_sector_nu = self.dim_V_nu
        if not (abs(self.j2_in - self.j2_nu) <= self.j2_out <= self.j2_in + self.j2_nu and (self.j2_in + self.j2_nu - self.j2_out) % 2 == 0):
            raise ValueError(f"No SU(2) coupling from j2_in={self.j2_in} and j2_nu={self.j2_nu} to j2_out={self.j2_out}.")
        self.lr_coeff = 1
        self.prefactor = float(self.dim_V_in) / float(self.dim_V_out)
        self.physical_partition_nu = None
        if self.j2_nu in j2_list_for_n_qubits(self.n_anc):
            self.physical_partition_nu = tuple(j2_to_qubit_partition(self.n_anc, self.j2_nu))
        self._coisometry = None
        self._kraus = None
        self._sector_choi = None

    @staticmethod
    def _spin_coupling_coisometry(j2_in: int, j2_nu: int, j2_out: int, tol: float = 1e-8):
        d_in = j2_in + 1
        d_nu = j2_nu + 1
        d_out = j2_out + 1
        I_in = np.eye(d_in, dtype=complex)
        I_nu = np.eye(d_nu, dtype=complex)
        Sx_i, Sy_i, Sz_i = spin_matrices_from_j2(j2_in)
        Sx_n, Sy_n, Sz_n = spin_matrices_from_j2(j2_nu)
        Jx = np.kron(Sx_i, I_nu) + np.kron(I_in, Sx_n)
        Jy = np.kron(Sy_i, I_nu) + np.kron(I_in, Sy_n)
        Jz = np.kron(Sz_i, I_nu) + np.kron(I_in, Sz_n)
        J2 = (Jx @ Jx + Jy @ Jy + Jz @ Jz)
        J2 = (J2 + J2.conj().T) / 2.0
        target = (j2_out / 2.0) * (j2_out / 2.0 + 1.0)
        evals, evecs = np.linalg.eigh(J2)
        idx = np.where(np.abs(evals - target) <= tol)[0]
        if len(idx) != d_out:
            raise RuntimeError(f"Unexpected SU(2) coupling multiplicity for (j2_in={j2_in}, j2_nu={j2_nu}, j2_out={j2_out}): found eigenspace dimension {len(idx)}, expected {d_out}.")
        Q = np.asarray(evecs[:, idx], dtype=complex)
        Jz_sub = (Q.conj().T @ Jz @ Q)
        Jz_sub = (Jz_sub + Jz_sub.conj().T) / 2.0
        mz_vals, W = np.linalg.eigh(Jz_sub)
        order = np.argsort(-mz_vals)
        U = Q @ W[:, order]
        desired_m = np.arange(j2_out / 2.0, -j2_out / 2.0 - 1, -1.0)
        if np.max(np.abs(mz_vals[order] - desired_m)) > 1e-6:
            raise RuntimeError(f"Failed to order Jz basis for (j2_in={j2_in}, j2_nu={j2_nu}, j2_out={j2_out}).")
        return U.conj().T

    def sector_intertwiner(self):
        if self._coisometry is None:
            self._coisometry = self._spin_coupling_coisometry(self.j2_in, self.j2_nu, self.j2_out)
        return self._coisometry

    def sector_projector_in_lambda_tensor_nu(self):
        C = self.sector_intertwiner()
        return (C.conj().T @ C + (C.conj().T @ C).conj().T) / 2.0

    def kraus_operators(self):
        if self._kraus is None:
            C = self.sector_intertwiner()
            d_in, d_nu = self.dim_sector_in, self.dim_sector_nu
            out = []
            for a in range(d_nu):
                e = np.zeros((d_nu, 1), dtype=complex)
                e[a, 0] = 1.0
                out.append(np.asarray(math.sqrt(self.prefactor) * (C @ np.kron(np.eye(d_in, dtype=complex), e)), dtype=complex))
            self._kraus = out
        return self._kraus

    def apply(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=complex)
        if X.shape != (self.dim_sector_in, self.dim_sector_in):
            raise ValueError(f"Input X must have shape {(self.dim_sector_in, self.dim_sector_in)}, got {X.shape}.")
        out = np.zeros((self.dim_sector_out, self.dim_sector_out), dtype=complex)
        for K in self.kraus_operators():
            out += K @ X @ K.conj().T
        return (out + out.conj().T) / 2.0

    def sector_choi(self) -> np.ndarray:
        if self._sector_choi is None:
            d_out, d_in = self.dim_sector_out, self.dim_sector_in
            J = np.zeros((d_out * d_in, d_out * d_in), dtype=complex)
            for K in self.kraus_operators():
                vecK = K.reshape(d_out * d_in, order="C")
                J += np.outer(vecK, vecK.conj())
            self._sector_choi = (J + J.conj().T) / 2.0
        return self._sector_choi

    # Unified effective-local aliases (already spin-only).
    def effective_local_kraus_operators(self):
        return self.kraus_operators()
    def apply_effective_local(self, X: np.ndarray) -> np.ndarray:
        return self.apply(X)
    def effective_local_block_choi(self) -> np.ndarray:
        return self.sector_choi()
    def verify_tp(self, tol=1e-8):
        J = self.sector_choi()
        d_out, d_in = self.dim_sector_out, self.dim_sector_in
        ptr = np.einsum("o i o j -> i j", J.reshape(d_out, d_in, d_out, d_in))
        err = np.linalg.norm(ptr - np.eye(d_in, dtype=complex))
        return err <= tol, float(err)
    def verify_effective_local_tp(self, tol=1e-8):
        return self.verify_tp(tol)
    def normalized_local_input_state(self, p: float) -> np.ndarray:
        rho_spin = rho_block_diag_in_spin_irrep(self.n_in, self.j2_in, float(p))
        z = float(np.trace(rho_spin).real)
        return None if z <= 0 else rho_spin / z
    def normalized_local_target_state(self, p: float) -> np.ndarray:
        rho_spin = rho_block_diag_in_spin_irrep(self.n_out, self.j2_out, float(p))
        z = float(np.trace(rho_spin).real)
        return None if z <= 0 else rho_spin / z
    def normalized_effective_local_input_state(self, p: float) -> np.ndarray:
        return self.normalized_local_input_state(p)
    def normalized_effective_local_target_state(self, p: float) -> np.ndarray:
        return self.normalized_local_target_state(p)
    def fidelity_curve(self, p_grid):
        p_grid = np.asarray(p_grid, dtype=float)
        roots, kept = [], []
        for p in p_grid:
            rho_in = self.normalized_local_input_state(float(p))
            rho_out = self.normalized_local_target_state(float(p))
            if rho_in is None or rho_out is None:
                continue
            roots.append(fidelity_root_numpy(rho_out, self.apply(rho_in)))
            kept.append(float(p))
        return np.asarray(kept, dtype=float), np.asarray(roots, dtype=float)
    def effective_local_fidelity_curve(self, p_grid):
        return self.fidelity_curve(p_grid)


class SolverExactIrrepFamily:
    """
    Build all exact maps for a fixed input/output irrep pair.

    Priority order:
    1) literal physical exact maps with nu \vdash (n_out-n_in) and LR != 0;
    2) if dim Sp_lambda = dim Sp_mu = 1, also allow abstract spin-only auxiliary
       sectors indexed by j2_nu, saved using canonical one-row partitions (j2_nu,).

    The block stored under block_key(j2_out, j2_in) is always the effective local
    block Choi on V_mu \otimes V_lambda, so it can be used directly by the same
    downstream analysis code as the other local/fixed-irrep solvers.
    """
    def __init__(self, n_in, n_out, dim=2, verbose=False, p_init_grid=5, p_fine_grid=101, n_rounds=1, irrep_in=None, irrep_out=None, cache_dir="data/exact_irrep"):
        if dim != 2:
            raise ValueError("SolverExactIrrepFamily currently supports only qubits (dim=2).")
        if irrep_in is None or irrep_out is None:
            raise ValueError("SolverExactIrrepFamily requires both irrep_in and irrep_out.")
        if n_out < n_in:
            raise ValueError("Need n_out >= n_in.")
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.n_anc = int(n_out - n_in)
        self.dim = int(dim)
        self.verbose = bool(verbose)
        self.p_init_grid = int(p_init_grid)
        self.p_fine_grid = int(p_fine_grid)
        self.n_rounds = int(n_rounds)
        self.cache_dir = cache_dir
        self.j2_in, self.partition_in = parse_qubit_irrep_label(irrep_in, self.n_in)
        self.j2_out, self.partition_out = parse_qubit_irrep_label(irrep_out, self.n_out)
        self.dim_Sp_in = mult_qubits(self.n_in, self.j2_in)
        self.dim_Sp_out = mult_qubits(self.n_out, self.j2_out)
        self.nu_entries = []
        for j2_nu in j2_list_for_n_qubits(self.n_anc):
            part_nu = tuple(j2_to_qubit_partition(self.n_anc, j2_nu))
            c = lr_coeff_qubit_irreps(self.partition_in, part_nu, self.partition_out)
            if c != 0:
                self.nu_entries.append({"mode": "physical", "partition_nu": part_nu, "j2_nu": int(j2_nu)})
        if self.dim_Sp_in == 1 and self.dim_Sp_out == 1:
            allowed_j2 = [int(L2) for (L2, _) in su2_commutant_projectors(self.j2_out, self.j2_in)]
            seen = {tuple(entry["partition_nu"]) for entry in self.nu_entries}
            for j2_nu in allowed_j2:
                part_nu = tuple(canonical_qubit_aux_partition(j2_nu))
                if part_nu in seen:
                    continue
                self.nu_entries.append({"mode": "abstract_spin_only", "partition_nu": part_nu, "j2_nu": int(j2_nu)})
                seen.add(part_nu)
        self._results_by_partition_nu = None
        self._blocks_by_partition_nu = None

    def admissible_nu_labels(self):
        return [{"j2_nu": int(e["j2_nu"]), "partition_nu": tuple(e["partition_nu"]), "mode": e["mode"]} for e in self.nu_entries]

    def _cache_path_for_nu(self, part_nu):
        part_in = str(tuple(self.partition_in)).replace(" ", "")
        part_out = str(tuple(self.partition_out)).replace(" ", "")
        part_nu_s = str(tuple(part_nu)).replace(" ", "")
        return os.path.join(self.cache_dir, f"{self.n_in}_to_{self.n_out}_{part_in}_to_{part_out}_nu_{part_nu_s}.npz")

    def _build_channel(self, entry):
        if entry["mode"] == "physical":
            return ExactFormulaFixedNuChannel(self.n_in, self.n_out, self.partition_in, self.partition_out, entry["partition_nu"], verbose=self.verbose)
        if entry["mode"] == "abstract_spin_only":
            return AbstractSpinOnlyExactFormulaFixedNuChannel(self.n_in, self.n_out, self.partition_in, self.partition_out, entry["partition_nu"], verbose=self.verbose)
        raise ValueError(f"Unknown exact-irrep nu mode: {entry['mode']}")

    def _result_dict_for_exact_nu(self, entry):
        ch = self._build_channel(entry)
        p_samples = sorted(set(np.linspace(0.5, 1.0, self.p_init_grid).tolist()))
        sampled_roots = []
        for p in p_samples:
            rho_in = ch.normalized_effective_local_input_state(float(p))
            rho_out = ch.normalized_effective_local_target_state(float(p))
            if rho_in is None or rho_out is None:
                continue
            sampled_roots.append(fidelity_root_numpy(rho_out, ch.apply_effective_local(rho_in)))
        sampled_root_lb = 0.0 if len(sampled_roots) == 0 else float(np.min(np.asarray(sampled_roots, dtype=float)))
        p_curve, root_curve = ch.effective_local_fidelity_curve(np.linspace(0.5, 1.0, self.p_fine_grid))
        if root_curve.size == 0:
            worst_p, worst_root = 1.0, 0.0
        else:
            idx = int(np.argmin(root_curve))
            worst_p = float(p_curve[idx])
            worst_root = float(root_curve[idx])
        ok, err = ch.verify_effective_local_tp()
        result = {
            "n_in": int(self.n_in), "n_out": int(self.n_out), "n_anc": int(self.n_anc),
            "j2_in": int(ch.j2_in), "j2_out": int(ch.j2_out), "j2_nu": int(ch.j2_nu),
            "partition_in": tuple(ch.partition_in), "partition_out": tuple(ch.partition_out),
            "partition_nu": tuple(ch.partition_nu),
            "physical_partition_nu": getattr(ch, "physical_partition_nu", None),
            "nu_matches_physical_ancilla": getattr(ch, "physical_partition_nu", None) is not None and tuple(ch.partition_nu) == tuple(getattr(ch, "physical_partition_nu", ()) or ()),
            "worst_p": float(worst_p), "worst_fidelity": float(worst_root ** 2),
            "sampled_fidelity": float(sampled_root_lb ** 2),
            "fidelity_curve_p": np.asarray(p_curve, dtype=float),
            "fidelity_curve_root": np.asarray(root_curve, dtype=float),
            "weights": {}, "local_choi_basis": {}, "exact_formula_mode": entry["mode"],
            "dim_Sp_in": int(ch.dim_Sp_in), "dim_Sp_out": int(ch.dim_Sp_out), "dim_Sp_nu": int(ch.dim_Sp_nu),
            "dim_V_in": int(ch.dim_V_in), "dim_V_out": int(ch.dim_V_out), "dim_V_nu": int(ch.dim_V_nu),
            "dim_sector_in": int(ch.dim_sector_in), "dim_sector_out": int(ch.dim_sector_out), "dim_sector_nu": int(ch.dim_sector_nu),
            "lr_coeff": int(getattr(ch, "lr_coeff", 1)), "prefactor": float(ch.prefactor),
            "tp_ok": bool(ok), "tp_error": float(err),
        }
        result[block_key(ch.j2_out, ch.j2_in)] = np.asarray(ch.effective_local_block_choi(), dtype=complex)
        return result

    def solve(self):
        if self._blocks_by_partition_nu is not None:
            return self._blocks_by_partition_nu
        results, blocks = {}, {}
        for entry in self.nu_entries:
            part_nu = tuple(entry["partition_nu"])
            result = self._result_dict_for_exact_nu(entry)
            results[part_nu] = result
            blocks[part_nu] = {(self.j2_out, self.j2_in): np.asarray(result[block_key(self.j2_out, self.j2_in)], dtype=complex)}
            if self.verbose:
                print(f"[exact nu] {self.partition_in} -> {self.partition_out}, nu={part_nu}, mode={entry['mode']}, worst F≈{result['worst_fidelity']:.8f}, tp_error={result['tp_error']:.3e}")
        self._results_by_partition_nu = results
        self._blocks_by_partition_nu = blocks
        return self._blocks_by_partition_nu

    def get_solution(self):
        return self.solve()
    def get_solution_blocks(self):
        return self.solve()
    def get_result(self, irrep_nu):
        self.solve()
        j2_nu, part_nu = parse_auxiliary_qubit_irrep_label(irrep_nu)
        if tuple(part_nu) in self._results_by_partition_nu:
            return self._results_by_partition_nu[tuple(part_nu)]
        return self._results_by_partition_nu[tuple(canonical_qubit_aux_partition(j2_nu))]
    def get_all_results(self):
        self.solve()
        return self._results_by_partition_nu
    def save_result(self, irrep_nu, path=None):
        result = self.get_result(irrep_nu)
        part_nu = tuple(result["partition_nu"])
        if path is None:
            path = self._cache_path_for_nu(part_nu)
        save_local_irrep_result(path, result, result[block_key(self.j2_out, self.j2_in)])
        return path
    def save_all_results(self):
        self.solve()
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
        saved = {}
        for entry in self.nu_entries:
            part_nu = tuple(entry["partition_nu"])
            path = self._cache_path_for_nu(part_nu)
            result = self._results_by_partition_nu[part_nu]
            save_local_irrep_result(path, result, result[block_key(self.j2_out, self.j2_in)])
            saved[part_nu] = path
        return saved
    def describe_results(self):
        self.solve()
        lines = [f"Exact irrep family for input {self.partition_in} (j2={self.j2_in}) -> output {self.partition_out} (j2={self.j2_out})"]
        for entry in self.nu_entries:
            part_nu = tuple(entry["partition_nu"])
            result = self._results_by_partition_nu[part_nu]
            lines.append(f"  nu={part_nu}, mode={entry['mode']}: worst_F≈{result['worst_fidelity']:.10f}, worst_p≈{result['worst_p']:.10f}, tp_error≈{result['tp_error']:.3e}")
        return "\n".join(lines)


def save_all_exact_irrep_families(n_in, n_out, dim=2, verbose=False, p_init_grid=5, p_fine_grid=101, n_rounds=1, cache_dir="data/exact_irrep", skip_empty=True):
    os.makedirs(cache_dir, exist_ok=True)
    input_parts = [tuple(j2_to_qubit_partition(n_in, j2)) for j2 in j2_list_for_n_qubits(n_in)]
    output_parts = [tuple(j2_to_qubit_partition(n_out, j2)) for j2 in j2_list_for_n_qubits(n_out)]
    saved_files, saved_pairs, skipped_pairs = [], [], []
    for part_in in input_parts:
        for part_out in output_parts:
            try:
                solver = SolverExactIrrepFamily(n_in=n_in, n_out=n_out, dim=dim, verbose=verbose, p_init_grid=p_init_grid, p_fine_grid=p_fine_grid, n_rounds=n_rounds, irrep_in=part_in, irrep_out=part_out, cache_dir=cache_dir)
                admissible = solver.admissible_nu_labels()
                if len(admissible) == 0:
                    msg = "no admissible nu"
                    skipped_pairs.append((part_in, part_out, msg))
                    if verbose:
                        print(f"[skip] {part_in} -> {part_out}: {msg}")
                    if skip_empty:
                        continue
                saved = solver.save_all_results()
                saved_paths = list(saved.values())
                saved_files.extend(saved_paths)
                saved_pairs.append((part_in, part_out))
                if verbose:
                    print(f"[saved exact] {part_in} -> {part_out}: {len(saved_paths)} file(s), nus={admissible}")
            except Exception as e:
                skipped_pairs.append((part_in, part_out, str(e)))
                print(f"[error] exact {part_in} -> {part_out}: {e}")
    return {"saved_files": saved_files, "saved_pairs": saved_pairs, "skipped_pairs": skipped_pairs}

