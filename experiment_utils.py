import wandb
import argparse
import numpy as np
PDEBENCH_STATES = [
        'Vx', 'Vy', 'Vz', 'density', 'pressure'
    ]
def initialize_wandb_catgel(args,timestamp,tags:list[str],method:str):
    run_config = wandb.config = {
        "epsilon": args.epsilon,
        "batch_size": args.batch_size,
        "seed_idx": args.seed_idx,
        "simulation_type": args.type,
        "normalization": args.normalization,
    }
    wandb.init(
        project="HierarchicalTucker_experiments",
        name=f"CatGel_{method}_eps_"+"".join(f"{args.epsilon:0.2f}_".split("."))+f"{args.normalization}"+f"_type_{args.type}_batchsize_{args.batch_size:03d}"+"_date_"+timestamp,
        config=run_config,
        tags=tags,
    )

def get_args_catgel():
    parser = argparse.ArgumentParser(description='This script reads the PDEBench simulation snapshots and compresses them using the HT format.')
    parser.add_argument('-s', '--seed', dest='seed_idx', type=int , help='Variable to pass seed index', default=None)
    parser.add_argument('-e', '--epsilon', dest='epsilon', type=float ,help='epsilon value', default=0.1)
    parser.add_argument('-d', '--data_location', dest='data_location', help='path to data', default=None)
    # parser.add_argument('-r', '--reshaping', dest='reshaping', nargs='+', type=int, help='Determines the reshaping for the tensor stream', default=[])
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, help='Batch size', default=1)
    parser.add_argument('-w', '--wandb', dest='wandb', action='store_true', help='Use wandb for logging', default=False)
    parser.add_argument('-t', '--type', dest='type', type=int, help='Type of simulation data', default=1)
    parser.add_argument('-n', '--normalization', dest='normalization', type=str, help='Method of normalization that is used', default='none')
    return parser.parse_args()

def get_args_pdebench():
    parser = argparse.ArgumentParser(description='This script reads the PDEBench simulation snapshots and compresses them using the HT format.')
    parser.add_argument('-s', '--seed', dest='seed_idx', type=int , help='Variable to pass seed index', default=None)
    parser.add_argument('-e', '--epsilon', dest='epsilon', type=float, help='epsilon value', default=0.1)
    parser.add_argument('-N', '--normalization', dest='normalization', type=str, help='Normalization method used', default='none')
    parser.add_argument('-t', '--type', dest='type', type=str, help='Type of simulation data', default="Rand")
    parser.add_argument('-M', '--mach_number', dest='M', help='Mach number for the simulations', default=None)
    parser.add_argument('-r', '--reshaping', dest='reshaping', nargs='+', type=int, help='Determines the reshaping for the tensor stream. Note that the transpose operation happens before reshaping!', default=[])
    parser.add_argument('-T', '--transpose', dest='transpose', nargs='+', type=int, help='Swaps the axes for the tensor stream', default=None)
    parser.add_argument('-d', '--data_location', dest='data_location', help='path to data', default=None)
    parser.add_argument('--states', dest='states', nargs='+', type=int, help='Determines the states that will be compressed', default=list(range(len(PDEBENCH_STATES))))
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, help='Batch size', default=1)
    parser.add_argument('-c', '--combine', dest='combine', action='store_true', help='Combine timesteps of the simulation', default=False)
    parser.add_argument('-n', '--numpy', dest='numpy', action='store_true', help='Use extracted numpy files to read data' , default=False)
    parser.add_argument('-S', '--save', dest='save', action='store_true', help='Enables saving the compression' , default=False)
    parser.add_argument('-w', '--wandb', dest='wandb', action='store_true', help='Use wandb for logging', default=False)
    return parser.parse_args()

def normalize(arr:np.array,method:str):
    implemented_methods = [
        'minmax',
        'zscore',
        'maxabs',
        'unitvector',
        'none',
    ]
    if method not in implemented_methods:
        raise NotImplementedError(f"method should be one of  "+" ".join(implemented_methods)+"!")
    if method == 'none':
        return arr, 0, 1
    else:
        if method == 'minmax':
            norm_constant1 = np.min(arr)
            norm_constant2 = np.max(arr) - norm_constant1
            norm_arr = (arr - norm_constant1) / norm_constant2

        elif method == 'zscore':
            norm_constant1 = np.mean(arr)
            norm_constant2 = np.std(arr)
            norm_arr = (arr - norm_constant1) / norm_constant2

        elif method == 'maxabs':
            norm_constant1 = 0
            norm_constant2 = np.max(np.abs(arr))
            norm_arr = arr / norm_constant2
        elif method == 'unitvector':
            norm_constant1 = 0
            norm_constant2 = np.linalg.norm(arr)
            norm_arr = arr / norm_constant2
    return norm_arr, norm_constant1, norm_constant2

def denormalize(norm_arr, method, norm_constant1, norm_constant2):
    implemented_methods = [
        'minmax',
        'zscore',
        'maxabs',
        'unitvector',
        'none',
    ]
    if method not in implemented_methods:
        raise NotImplementedError(f"method should be one of  "+" ".join(implemented_methods)+"!")
    if method == 'none':
        return norm_arr
    elif method == 'minmax':
        arr = norm_arr * norm_constant2 + norm_constant1
    elif method == 'zscore':
        arr = norm_arr * norm_constant2 + norm_constant1
    elif method == 'maxabs':
        arr = norm_arr * norm_constant2
    elif method == 'unitvector':
        arr = norm_arr * norm_constant2  + norm_constant1
    return arr
