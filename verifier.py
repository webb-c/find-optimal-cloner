from utils import *
import numpy as np

class Verifier:
    def __init__(self, n_pure_samples, n_mixed_samples, dim, n_in, n_out, choi_matrix):
        self.n_pure_samples = n_pure_samples
        self.n_mixed_samples = n_mixed_samples
        self.dim = dim
        self.n_in = n_in
        self.n_out = n_out
        self.d_in = dim ** n_in
        self.d_out = dim ** n_out
        self.choi_matrix = choi_matrix

    def verify(self):
        # 1. Static Verification (Mixed p=0.5, Pure p=1.0)
        rho_mix_static = rho_diag(0.5)
        ideal_out_mix = rho_tensor_power(rho_mix_static, self.n_out)
        result_mix = apply_choi_numpy(self.choi_matrix, rho_tensor_power(rho_mix_static, self.n_in), self.d_out, self.d_in)
        print(f"Fidelity at p = 0.5 (Mixed): {fidelity_root(ideal_out_mix, result_mix)**2:.8f}")

        rho_pure_static = rho_diag(1.0)
        ideal_out_pure = rho_tensor_power(rho_pure_static, self.n_out)
        result_pure = apply_choi_numpy(self.choi_matrix, rho_tensor_power(rho_pure_static, self.n_in), self.d_out, self.d_in)
        print(f"Fidelity at p = 1.0 (Pure) : {fidelity_root(ideal_out_pure, result_pure)**2:.8f}")
        
        rng = np.random.default_rng()

        # 2. Random Pure State Verification
        pure_fidelities = []
        for i in range(self.n_pure_samples):
            # Haar-measure random ket vector
            psi = self.random_pure_ket(rng)
            rho = np.outer(psi, psi.conj())
            
            rho_in = rho_tensor_power(rho, self.n_in)
            rho_ideal = rho_tensor_power(rho, self.n_out)
            rho_actual = apply_choi_numpy(self.choi_matrix, rho_in, self.d_out, self.d_in)
            
            f = fidelity_root(rho_ideal, rho_actual)**2
            print(f"Random pure state fidelity [{i+1}/{self.n_pure_samples}]: {f:.8f}")
            pure_fidelities.append(f)
        
        if self.n_pure_samples > 0:
            print(f"Minimum Pure Fidelity ({self.n_pure_samples} samples): {np.min(pure_fidelities):.8f}")

        # 3. Random Mixed State Verification (Hilbert-Schmidt)
        mixed_fidelities = []
        for i in range(self.n_mixed_samples):
            rho = self.random_mixed_rho_hs(rng)
            
            rho_in = rho_tensor_power(rho, self.n_in)
            rho_ideal = rho_tensor_power(rho, self.n_out)
            rho_actual = apply_choi_numpy(self.choi_matrix, rho_in, self.d_out, self.d_in)
            
            f = fidelity_root(rho_ideal, rho_actual)**2
            print(f"Random mixed state fidelity [{i+1}/{self.n_mixed_samples}]: {f:.8f}")
            mixed_fidelities.append(f)
            
        if self.n_mixed_samples > 0:
            print(f"Minimum Mixed Fidelity ({self.n_mixed_samples} samples): {np.min(mixed_fidelities):.8f}")

    @staticmethod
    def random_pure_ket(rng):
        v = rng.normal(size=2) + 1j * rng.normal(size=2)
        v = v / np.linalg.norm(v)
        return v

    @staticmethod
    def random_mixed_rho_hs(rng):
        x = rng.normal(size=3)
        norm = np.linalg.norm(x)
        if norm > 0:
            x = x / norm
        
        # Radius sampling for HS distribution in 3D (r^2 dr)
        u = rng.random()
        r = u ** (1.0 / 3.0)

        I, X, Y, Z = qubit_paulis()
        nx, ny, nz = x
        bloch = nx * X + ny * Y + nz * Z
        rho = (I + r * bloch) / 2.0
        rho = (rho + rho.conj().T) / 2

        # Numerical stability: enforce positive semi-definiteness
        w, v = np.linalg.eigh(rho)
        w = np.clip(w, 0.0, None)
        rho = (v * w) @ v.conj().T
        rho = rho / np.trace(rho)
        return rho