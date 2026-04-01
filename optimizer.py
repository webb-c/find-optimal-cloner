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

            # evaluate on fine grid
            p_fine = np.linspace(0.5, 1.0, self.p_fine_grid)
            fvals = []
            for p in p_fine:
                sigma_blocks = self._apply_channel_blocks_numpy(J_opt, float(p))
                fvals.append(self._root_fidelity_full_numpy(float(p), sigma_blocks))
            fvals = np.array(fvals)

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
        sigma_blocks = self._apply_channel_blocks_numpy(J_opt, float(self.spectrum))
        fvals = (self._root_fidelity_full_numpy(float(self.spectrum), sigma_blocks))
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
        
        result[str((j2_out, j2_in))] = solver.get_solution_blocks()
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
            sol = result[str((j2_out, j2_in))]
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
                J_loc = np.asarray(self.local_results[(j2_out, j2_in)][str((j2_out, j2_in))], dtype=complex)
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
                J_loc = np.asarray(self.local_results[(j2_out, j2_in)][str((j2_out, j2_in))], dtype=complex)
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
            fvals = []
            for p in p_fine:
                sigma_blocks = self._apply_channel_blocks_numpy(J_opt, float(p))
                fvals.append(self._root_fidelity_full_numpy(float(p), sigma_blocks))
            fvals = np.asarray(fvals, dtype=float)

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
            save_dict[str(k)] = J
            save_dict[f"c_{k}"] = np.array(self._coefficients[k], dtype=float)
        np.savez_compressed(path, **save_dict)
