import numpy as np
import argparse

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
