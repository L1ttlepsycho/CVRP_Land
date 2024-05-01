import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
base_dir='../../'

def run_lkh(dataset_dir='cvrp_land/Dataset/scale-1000', save_dir='results/heuristics',cpus=32):
    """
    run lkh on a defaut setting
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    dataset_dir=base_dir+dataset_dir
    save_dir=base_dir+save_dir

    path_list = [os.path.join(dataset_dir, f) for f in sorted(os.listdir(dataset_dir))] \
        if os.path.isdir(dataset_dir) else [dataset_dir]
    for path in path_list:
        if path.endswith('.pkl'):
            #print('running lkh')
            os.system(f'python -u CVRP_baseline.py --method "lkh" --cpus {cpus} --disable_cache --dataset "{path}" --results_dir {save_dir}' )

def run_hgs(dataset_dir='cvrp_land/Dataset/scale-1', save_dir='results/heuristics',cpus=32): 
    """
    run hgs on a defaut setting
    """

    dataset_dir=base_dir+dataset_dir
    save_dir=base_dir+save_dir

    path_list = [os.path.join(dataset_dir, f) for f in sorted(os.listdir(dataset_dir))] \
        if os.path.isdir(dataset_dir) else [dataset_dir]
    for path in path_list:
        if path.endswith('.pkl'):
            #print('running hgs')
            os.system(f'python -u CVRP_baseline.py --method "hgs" --cpus {cpus} --disable_cache --dataset "{path}" --results_dir {save_dir}' )   

def run_ortools(dataset_dir='cvrp_land/Dataset', save_dir='results/heuristics/cvrp_ortools'):
    """
    run or-tools on a defaut setting
    """

    dataset_dir=base_dir+dataset_dir
    save_dir=base_dir+save_dir

    path_list = [os.path.join(dataset_dir, f) for f in sorted(os.listdir(dataset_dir))] \
        if os.path.isdir(dataset_dir) else [dataset_dir]
    #print(path_list)
    for path in path_list:
        res=os.system(f'python or_tools/or-tools.py --dataset_dir {path} --results_dir {save_dir}')