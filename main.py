import os
import argparse
from utils import *
from optimizer import *
from verifier import *

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--n_in", type=int, default=4, help="Number of input clones.")
    parser.add_argument("--n_out", type=int, default=4, help="Number of output clones.")
    parser.add_argument("--dim", type=int, default=2, help="Dimension of each qubit.")
    parser.add_argument("--method", type=str, default="sdp", choices=["sdp_fix", "sdp", "sdp_perm_fix", "sdp_perm"], help="Optimization method.")
    parser.add_argument("--p_init_grid", type=int, default=21, help="Number of p grid points for sdp method. (20 - 50 recommended)")
    parser.add_argument("--p_fine_grid", type=int, default=301, help="Number of p grid points for refinement. (300 - 500 recommended)")
    parser.add_argument("--n_rounds", type=int, default=3, help="Number of refinement rounds for sdp method. (3 - 5 recommended)")
    
    parser.add_argument("--verify", type=str2bool, default=False, help="Whether to run verification after optimization.")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of input sample states. (each pure, mix)")
    
    parser.add_argument("--verbose", type=str2bool, default=True, help="Whether to print verbose output.")
    parser.add_argument("--save_data", type=str2bool, default=True, help="Whether to save data.")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.dim != 2:
        raise NotImplementedError("Only qubit (dim=2) case is implemented in this code.")
    
    if args.method == "sdp_fix":
        solver = SolverSDPTwoPoint(n_in=args.n_in, n_out=args.n_out, dim=args.dim, verbose=args.verbose)
    elif args.method == "sdp":
        solver = SolverSDP(n_in=args.n_in, n_out=args.n_out, dim=args.dim, verbose=args.verbose,p_init_grid=args.p_init_grid, p_fine_grid=args.p_fine_grid, n_rounds=args.n_rounds)
    elif args.method == "sdp_perm_fix":
        solver = SolverSDPPermTwoPoint(n_in=args.n_in, n_out=args.n_out, dim=args.dim, verbose=args.verbose)
    elif args.method == "sdp_perm":
        solver = SolverSDPPerm(n_in=args.n_in, n_out=args.n_out, dim=args.dim, verbose=args.verbose,p_init_grid=args.p_init_grid, p_fine_grid=args.p_fine_grid, n_rounds=args.n_rounds)
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    J = solver.get_solution()
    if args.verify:
        verifier = Verifier(n_pure_samples=args.n_samples, n_mixed_samples=args.n_samples, dim=args.dim, n_in=args.n_in, n_out=args.n_out, choi_matrix=J)
        verifier.verify()
        
    if args.save_data:
        os.makedirs(f"data/{args.method}", exist_ok=True)
        filepath = f"data/{args.method}/{args.n_in}_to_{args.n_out}.npy"
        np.save(filepath, J)
        print(f"Saved J_opt to {filepath}")