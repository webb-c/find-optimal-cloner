# Quantum Cloner Optimizer

Python implementation for optimizing quantum cloning machines using Semidefinite Programming (SDP). 

The code seeks to find the optimal Quantum Channel (represented by its Choi matrix) that transforms $N$ input copies of a qubit state into $M$ ($M > N$) output copies ($N \to M$ cloner), maximizing the fidelity.

$$ \mathcal{T}_{\text{opt}} = \argmax_{\mathcal{T} \text{ : CPTP}} \inf_{\rho \text{ : mixed}} F(\mathcal{T}(\rho^{\otimes M}, \rho^{\otimes N}))$$

## Project Structure

- `main.py`: The entry point of the program. Parses arguments and runs the optimization.
- `optimizer.py`: Contains the SDP solver logic using `cvxpy`.
  - `Solver`: Abstract class for solvers.
  - `SolverSDP`: Implements the iterative cutting-plane method to optimize over the range of purities.
  - `SolverSDPTwoPoint` : SolverSDP with using just two points (pure state, maximally mixed state).
- `verifier.py`: Tools to verify the solution against random pure and mixed states.
- `utils.py`: Utility functions for quantum information operations (Pauli matrices, tensor products, fidelity, etc.).

## Dependencies

- Python 3+
- NumPy
- CVXPY
- A convex solver (e.g., **MOSEK**, SCS, Clarabel).
  - *Note: MOSEK is recommended for performance and stability with complex SDPs. You have to get license keys from mosek.com* 

```bash
pip install cvxpy numpy mosek
```

## Usage

Run the optimization via the command line:

```bash
python main.py --n_in <N> --n_out <M> [options]
```

### Examples

**1. Optimize a 2 to 3 cloner and verify the result:**

```bash
python main.py --n_in 2 --n_out 3 --method sdp --verify True
```

**2. Optimize a 1 to 2 cloner with specific grid settings:**

```bash
python main.py --n_in 1 --n_out 2 --p_init_grid 20 --n_rounds 5
```

**3. Optimize a 1 to 5 cloner by using fixed point optimizer:**

```bash
python main.py --n_in 1 --n_out 5 --method sdp_fix
```

### Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--n_in` | `4` | Number of input copies ($N$). |
| `--n_out` | `4` | Number of output copies ($M$). |
| `--dim` | `2` | Dimension of the quantum system (currently fixed to 2 for qubits). |
| `--method` | `sdp` | Optimization method: `sdp` (iterative) or `sdp_fix` (endpoints (i.e., pure and maximally mixed) only). |
| `--p_init_grid` | `21` | Number of p grid points for sdp method. 
| `--p_fine_grid` | `301` | Number of p grid points for refinement stage.
| `--n_rounds` | `3` | Number of refinement rounds for sdp method. 
| `--verify` | `False` | Run verification with random states after optimization. |
| `--n_samples`| `10` | Number of random samples for verification. |
| `--verbose` | `True` | Print detailed logs. |
| `--save_data`| `True` | Save the resulting Choi matrix to `data/`. |

## Output

The resulting Choi matrix $J$ is saved as a NumPy binary file in the `data/` directory, named `{n_in}_to_{n_out}.npy`.