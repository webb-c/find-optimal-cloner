import os
import re
import argparse
import numpy as np

from math import comb
from typing import List, Tuple, Union


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


LabelLike = Union[str, int, Tuple[int, ...], List[int]]
def normalize_partition(parts, n: int):
    parts = [int(x) for x in parts]
    if len(parts) == 0:
        raise ValueError("Empty irrep label.")
    if any(x < 0 for x in parts):
        raise ValueError(f"Partition entries must be nonnegative: {parts}")
    parts = [x for x in parts if x > 0]
    if len(parts) == 0:
        raise ValueError("Partition cannot be all zeros.")
    if parts != sorted(parts, reverse=True):
        raise ValueError(f"Partition must be weakly decreasing: {parts}")
    if sum(parts) != n:
        raise ValueError(f"Partition {tuple(parts)} does not sum to n={n}.")
    if len(parts) > 2:
        raise ValueError("Only qubit (d=2) irreps are supported, so partitions may have at most 2 rows.")
    return tuple(parts)


def j2_to_qubit_partition(n: int, j2: int):
    if j2 not in j2_list_for_n_qubits(n):
        raise ValueError(f"j2={j2} is not a valid qubit irrep label for n={n}.")
    lam1 = (n + j2) // 2
    lam2 = (n - j2) // 2
    return (lam1,) if lam2 == 0 else (lam1, lam2)


def qubit_partition_to_j2(partition, n: int):
    partition = normalize_partition(list(partition), n)
    if len(partition) == 1:
        lam1, lam2 = partition[0], 0
    else:
        lam1, lam2 = partition
    j2 = lam1 - lam2
    if j2 not in j2_list_for_n_qubits(n):
        raise ValueError(f"Partition {partition} is not a valid qubit irrep for n={n}.")
    return j2


def parse_qubit_irrep_label(label, n: int):
    if isinstance(label, int):
        j2 = int(label)
        return j2, j2_to_qubit_partition(n, j2)

    if isinstance(label, (tuple, list)):
        parts = tuple(int(x) for x in label)
        j2 = qubit_partition_to_j2(parts, n)
        return j2, normalize_partition(list(parts), n)

    s = str(label).strip().replace(" ", "")
    if len(s) == 0:
        raise ValueError("Irrep label cannot be empty.")

    if s.startswith("j2="):
        j2 = int(s.split("=", 1)[1])
        return j2, j2_to_qubit_partition(n, j2)

    if re.fullmatch(r"\d+", s):
        x = int(s)
        if x in j2_list_for_n_qubits(n):
            return x, j2_to_qubit_partition(n, x)
        return qubit_partition_to_j2((x,), n), normalize_partition([x], n)

    nums = [int(x) for x in re.findall(r"\d+", s)]
    if len(nums) == 0:
        raise ValueError(f"Could not parse irrep label: {label!r}")
    parts = tuple(nums)
    j2 = qubit_partition_to_j2(parts, n)
    return j2, normalize_partition(list(parts), n)


def normalized_spin_irrep_state(n: int, j2: int, p: float, tol: float = 1e-12):
    p = float(p)
    block = rho_block_diag_in_spin_irrep(n, j2, p)
    z = float(np.trace(block).real)
    if z > tol:
        return block / z
    if p >= 1.0 - tol:
        out = np.zeros((j2 + 1, j2 + 1), dtype=complex)
        out[0, 0] = 1.0
        return out
    
    raise ValueError(f"Projected state on irrep j2={j2} has zero trace for n={n}, p={p:.12g}.")


def apply_local_choi_numpy(J: np.ndarray, rho: np.ndarray) -> np.ndarray:
    rho = np.asarray(rho, dtype=complex)
    J = np.asarray(J, dtype=complex)
    d_in = rho.shape[0]
    d_out = J.shape[0] // d_in
    if J.shape != (d_out * d_in, d_out * d_in):
        raise ValueError("Incompatible shapes for local Choi matrix and local input state.")
    M = np.kron(np.eye(d_out, dtype=complex), rho.T) @ J
    sigma = partial_trace_numpy(M, d_out, d_in, axis=1)
    sigma = (sigma + sigma.conj().T) / 2.0
    tr = float(np.trace(sigma).real)
    if tr > 1e-14:
        sigma = sigma / tr
    return sigma



# ===================================
# Helpers for irrep optimizer
# ===================================
def local_irrep_result_to_dict(solver):
    result = {
        "n_in": int(solver.n_in),
        "n_out": int(solver.n_out),
        "j2_in": int(solver.j2_in),
        "j2_out": int(solver.j2_out),
        "worst_p": float(solver._worst_p),
        "worst_fidelity": float(solver._worst_root_fidelity ** 2),
        "sampled_fidelity": float(solver._sampled_root_lb ** 2),
        "weights": {int(L2): float(val) for L2, val in solver._weights.items()},
        "local_choi_basis": {int(L2): np.asarray(J, dtype=complex) for L2, J in solver.local_basis.items()},
        "fidelity_curve_p": np.asarray(solver._fidelity_curve_p, dtype=float),
    }
    return result

def save_local_irrep_result(path: str, result: dict, J: None):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    save_dict = {
        "n_in": np.array(result["n_in"], dtype=int),
        "n_out": np.array(result["n_out"], dtype=int),
        "j2_in": np.array(result["j2_in"], dtype=int),
        "j2_out": np.array(result["j2_out"], dtype=int),
        "worst_p": np.array(result["worst_p"], dtype=float),
        "worst_fidelity": np.array(result["worst_fidelity"], dtype=float),
        "sampled_fidelity": np.array(result.get("sampled_fidelity", 0.0), dtype=float),
        "fidelity_curve_p": np.asarray(result["fidelity_curve_p"], dtype=float),
    }
    
    save_dict[str((result["j2_out"], result["j2_in"]))] = J
    

    # Description about the final choi map represented with basis matrix and weights.
    for L2, val in result.get("weights", {}).items():
        save_dict[f"weight_L2_{int(L2)}"] = np.array(float(val), dtype=float)
    for L2, J in result.get("local_choi_basis", {}).items():
        save_dict[f"basis_L2_{int(L2)}"] = np.asarray(J, dtype=complex)
    np.savez_compressed(path, **save_dict)


def load_local_irrep_result(path: str):
    raw = np.load(path, allow_pickle=True)
    data = {k: raw[k] for k in raw.files}
    
    weights = {}
    basis = {}
    for k, v in data.items():
        if k.startswith("weight_L2_"):
            weights[int(k.split("_")[-1])] = float(np.asarray(v).item())
        elif k.startswith("basis_choi_L2_"):
            basis[int(k.split("_")[-1])] = np.asarray(v)
    return {
        "n_in": int(np.asarray(data["n_in"]).item()),
        "n_out": int(np.asarray(data["n_out"]).item()),
        "j2_in": int(np.asarray(data["j2_in"]).item()),
        "j2_out": int(np.asarray(data["j2_out"]).item()),
        "p_samples": np.asarray(data.get("p_samples", np.array([], dtype=float)), dtype=float),
        "worst_p": float(np.asarray(data.get("worst_p", 1.0)).item()),
        "worst_fidelity": float(np.asarray(data.get("worst_fidelity", 0.0)).item()),
        "weights": weights,
        "local_choi_basis": basis,
        "sampled_fidelity": float(np.asarray(data.get("sampled_fidelity", 0.0)).item()),
        "fidelity_curve_p": np.asarray(data.get("fidelity_curve_p", np.array([], dtype=float)), dtype=float),
        f"({data['j2_out']}, {data['j2_in']})": data[str((data['j2_out'], data['j2_in']))]
    }