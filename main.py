import os
import argparse
from utils import *
from optimizer import *
from verifier import *

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--n_in", type=int, default=1, help="Number of input clones.")
    parser.add_argument("--n_out", type=int, default=4, help="Number of output clones.")
    parser.add_argument("--dim", type=int, default=2, help="Dimension of each qubit.")
    parser.add_argument("--method", type=str, default="sdp_perm", choices=["sdp_fix", "sdp", "sdp_perm_fix", "sdp_perm", "sdp_irrep", "sdp_irrep_global", "fixed_irrep", "fixed_irrep_all"], help="Optimization method.")
    parser.add_argument("--p_init_grid", type=int, default=21, help="Number of p grid points for sdp method. (20 - 50 recommended)")
    parser.add_argument("--p_fine_grid", type=int, default=301, help="Number of p grid points for refinement. (300 - 500 recommended)")
    parser.add_argument("--n_rounds", type=int, default=3, help="Number of refinement rounds for sdp method. (3 - 5 recommended)")
    
    parser.add_argument("--irrep_in", type=str, help='Input irrep label. Examples: "(2)", "(1,1)", "j2=2".')
    parser.add_argument("--irrep_out", type=str, help='Output irrep label. Examples: "(4)", "(3,1)", "j2=2".')
    parser.add_argument("--spectrum", type=str, default=None, help="If you want to find the optimal cloner for some fixed spectrum of input state.")
    
    parser.add_argument("--irrep_local_dir", type=str, default="data/sdp_irrep", help="Directory used to save/reuse local irrep solutions.")
    parser.add_argument("--reuse_irrep_local", type=str2bool, default=True, help="Whether sdp_irrep_global reuses saved local-irrep solutions.")
    
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
        if args.spectrum is None:
            solver = SolverSDPPerm(n_in=args.n_in, n_out=args.n_out, dim=args.dim, verbose=args.verbose, p_init_grid=args.p_init_grid, p_fine_grid=args.p_fine_grid, n_rounds=args.n_rounds)
        else:
            solver = SolverSDPPermSpectrum(n_in=args.n_in, n_out=args.n_out, dim=args.dim, verbose=args.verbose, spectrum=args.spectrum)
    elif args.method == "sdp_irrep":
        if args.irrep_in is None or args.irrep_out is None:
            raise ValueError("sdp_irrep requires both --irrep_in and --irrep_out.")
        solver = SolverLocalIrrepSDP(n_in=args.n_in, n_out=args.n_out, dim=args.dim, verbose=args.verbose, p_init_grid=args.p_init_grid, p_fine_grid=args.p_fine_grid, n_rounds=args.n_rounds, irrep_in=args.irrep_in, irrep_out=args.irrep_out)
    elif args.method == "sdp_irrep_global":
        solver = SolverGlobalIrrepSDP(n_in=args.n_in, n_out=args.n_out, dim=args.dim, verbose=args.verbose, p_init_grid=args.p_init_grid, p_fine_grid=args.p_fine_grid, n_rounds=args.n_rounds, local_cache_dir=args.irrep_local_dir, reuse_local_irrep=args.reuse_irrep_local)
    elif args.method == "fixed_irrep":
        if args.irrep_in is None or args.irrep_out is None:
            raise ValueError("fixed_irrep requires both --irrep_in and --irrep_out.")
        solver = SolverFixedIrrepFamily(n_in=args.n_in, n_out=args.n_out, dim=args.dim, verbose=args.verbose, p_init_grid=args.p_init_grid, p_fine_grid=args.p_fine_grid, n_rounds=args.n_rounds, irrep_in=args.irrep_in, irrep_out=args.irrep_out)
    elif args.method == "fixed_irrep_all":
        summary = save_all_fixed_irrep_families(
            n_in=args.n_in,
            n_out=args.n_out,
            dim=args.dim,
            verbose=args.verbose,
            p_init_grid=args.p_init_grid,
            p_fine_grid=args.p_fine_grid,
            n_rounds=args.n_rounds,
            cache_dir="data/fixed_irrep",
        )
        print("\n[saved files]")
        for path in summary["saved_files"]:
            print(path)
        print("\n[skipped pairs]")
        for part_in, part_out, reason in summary["skipped_pairs"]:
            print(f"{part_in} -> {part_out}: {reason}")
            
        exit(0)
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    if args.method in ["sdp", "sdp_fix"]:
        J = solver.get_solution()
    elif args.method == "sdp_irrep":
        J = solver.get_solution()
        print(solver.describe_result())
    elif args.method == "fixed_irrep":
        J = solver.get_solution_blocks()
    else:
        J = solver.get_solution_blocks()
        
    if args.verify:
        if args.method in ["sdp", "sdp_fix"]:
            J_choi = solver.get_solution()
        else:
            J_choi = solver.blocks_to_full_choi(J)
        
        verifier = Verifier(n_pure_samples=args.n_samples, n_mixed_samples=args.n_samples, dim=args.dim, n_in=args.n_in, n_out=args.n_out, choi_matrix=J_choi)
        verifier.verify()
        
    if args.save_data:
        os.makedirs(f"data/{args.method}", exist_ok=True)
        
        if args.method in ["sdp", "sdp_fix"]:
            filepath = f"data/{args.method}/{args.n_in}_to_{args.n_out}.npy"
            np.save(filepath, J)
        elif args.method == "sdp_irrep":
            result = solver.get_result()
            safe_in = str(result['partition_in']).replace(" ", "")
            safe_out = str(result['partition_out']).replace(" ", "")
            filepath = f"data/{args.method}/{args.n_in}_to_{args.n_out}_{safe_in}_to_{safe_out}.npz"
            solver.save_result(filepath)
        elif args.method == "sdp_irrep_global":
            filepath = f"data/{args.method}/{args.n_in}_to_{args.n_out}.npz"
            solver.save_result(filepath)
        elif args.method == "fixed_irrep":
            saved = solver.save_all_results()
            print(saved)
            filepath = ", ".join(saved.values()) if len(saved) > 0 else "(no files saved)"
        else:
            if args.spectrum is not None:
                filepath = f"data/{args.method}/{args.n_in}_to_{args.n_out}_p={args.spectrum}.npz"
            else:
                filepath = f"data/{args.method}/{args.n_in}_to_{args.n_out}.npz"
            save_choi_blocks(filepath, J)
        
        print(f"Saved J_opt to {filepath}")