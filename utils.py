import numpy as np
import argparse
from math import comb


def spin_matrices_from_j2(j2: int):
    j = j2 / 2.0
    d = j2 + 1
    m_vals = np.arange(j, -j - 1, -1)  # [j, j-1, ..., -j]

    Jp = np.zeros((d, d), dtype=complex)
    for col in range(1, d):
        m = m_vals[col]  # IMPORTANT: m of the COLUMN state (the ket being raised)
        Jp[col - 1, col] = np.sqrt((j - m) * (j + m + 1.0))

    Jm = Jp.conj().T
    Jx = (Jp + Jm) / 2.0
    Jy = (Jp - Jm) / (2.0 * 1j)
    Jz = np.diag(m_vals.astype(float))
    return Jx, Jy, Jz


def su2_commutant_projectors(j2_out: int, j2_in: int, tol: float = 1e-8):
    d_out = j2_out + 1
    d_in = j2_in + 1
    Iout = np.eye(d_out, dtype=complex)
    Iin  = np.eye(d_in, dtype=complex)

    Sx_o, Sy_o, Sz_o = spin_matrices_from_j2(j2_out)
    Sx_i, Sy_i, Sz_i = spin_matrices_from_j2(j2_in)

    Gx = np.kron(Sx_o, Iin) - np.kron(Iout, Sx_i.T)
    Gy = np.kron(Sy_o, Iin) - np.kron(Iout, Sy_i.T)
    Gz = np.kron(Sz_o, Iin) - np.kron(Iout, Sz_i.T)

    G2 = Gx @ Gx + Gy @ Gy + Gz @ Gz
    G2 = (G2 + G2.conj().T) / 2.0

    evals, evecs = np.linalg.eigh(G2)

    L2_min = abs(j2_out - j2_in)
    L2_max = j2_out + j2_in
    projectors = []
    for L2 in range(L2_min, L2_max + 1, 2):
        L = L2 / 2.0
        target = L * (L + 1.0)

        idx = np.where(np.abs(evals - target) <= tol)[0]
        if len(idx) == 0:
            # 디버깅 도움: 실제 고유값/타겟 확인
            raise RuntimeError(
                f"Projector build failed for (j2_out={j2_out}, j2_in={j2_in}, L2={L2}). "
                f"target={target}, eigs(min..max)=({evals.min()}..{evals.max()})"
            )

        P = evecs[:, idx] @ evecs[:, idx].conj().T
        P = (P + P.conj().T) / 2.0
        projectors.append((L2, P))

    return projectors


def j2_list_for_n_qubits(n: int):
    return list(range(n, n % 2 - 1, -2))

def mult_qubits(n: int, j2: int) -> int:
    k = (n - j2) // 2
    if k < 0 or k > n:
        return 0
    return comb(n, k) - (comb(n, k - 1) if k - 1 >= 0 else 0)

def rho_block_diag_in_spin_irrep(n: int, j2: int, p: float):
    q = 1.0 - p
    d = j2 + 1
    m2_vals = np.arange(j2, -j2 - 1, -2)  # j2, j2-2, ..., -j2

    expp = (n + m2_vals) // 2
    expq = (n - m2_vals) // 2
    vals = (p ** expp) * (q ** expq)
    return np.diag(vals.astype(float)).astype(complex)

def partial_trace_numpy(A: np.ndarray, d_out: int, d_in: int, axis: int):
    A4 = A.reshape(d_out, d_in, d_out, d_in)
    if axis == 0:
        # Tr_out: sum_o A[o, i, o, j]
        return np.einsum("o i o j -> i j", A4)
    elif axis == 1:
        # Tr_in: sum_i A[o, i, p, i]
        return np.einsum("o i p i -> o p", A4)
    else:
        raise ValueError("axis must be 0 or 1")

def fidelity_root_numpy(rho: np.ndarray, sigma: np.ndarray) -> float:
    rho = (rho + rho.conj().T) / 2
    sigma = (sigma + sigma.conj().T) / 2
    w, V = np.linalg.eigh(rho)
    w = np.clip(w, 0.0, None)
    sqrt_rho = (V * np.sqrt(w)) @ V.conj().T
    C = sqrt_rho @ sigma @ sqrt_rho
    C = (C + C.conj().T) / 2
    wc, _ = np.linalg.eigh(C)
    wc = np.clip(wc, 0.0, None)
    return float(np.sum(np.sqrt(wc)).real)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def kron_all(mats):
    out = np.array([[1.0 + 0.0j]])
    for A in mats:
        out = np.kron(out, A)
    return out

def qubit_paulis():
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return I, X, Y, Z

def op_on_n_qubits(single_op, n_qubits, which):
    I = np.eye(2, dtype=complex)
    mats = [I] * n_qubits
    mats[which] = single_op
    return kron_all(mats)

def collective_spin(pauli, n_qubits):
    dim = 2**n_qubits
    S = np.zeros((dim, dim), dtype=complex)
    for i in range(n_qubits):
        S += op_on_n_qubits(pauli, n_qubits, i) / 2.0
    return S

def rho_diag(p):
    return np.array([[p, 0], [0, 1 - p]], dtype=complex)

def rho_tensor_power(rho, n):
    out = np.array([[1.0 + 0.0j]])
    for _ in range(n):
        out = np.kron(out, rho)
    return out

def fidelity_root(rho, sigma, eps=1e-12):
    rho = (rho + rho.conj().T) / 2
    sigma = (sigma + sigma.conj().T) / 2

    w, v = np.linalg.eigh(rho)
    w = np.clip(w, 0, None)
    sqrt_rho = (v * np.sqrt(w)) @ v.conj().T

    A = sqrt_rho @ sigma @ sqrt_rho
    A = (A + A.conj().T) / 2

    w2, _ = np.linalg.eigh(A)
    w2 = np.clip(w2, 0, None)
    return float(np.sum(np.sqrt(w2)).real)

def apply_choi_numpy(J, rho, d_out, d_in):
    J4 = J.reshape(d_out, d_in, d_out, d_in)
    # sigma = np.einsum("aibj,ji->ab", J4, rho)
    sigma = np.einsum("aibj,ij->ab", J4, rho)
    sigma = (sigma + sigma.conj().T) / 2
    tr = np.trace(sigma)
    if abs(tr) > 1e-15:
        sigma = sigma / tr
    return sigma

def save_choi_blocks(filename: str, j_blocks: dict):
    save_dict = {str(k): v for k, v in j_blocks.items()}
    np.savez_compressed(filename, **save_dict)