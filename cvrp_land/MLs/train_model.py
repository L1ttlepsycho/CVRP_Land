import os

#os.chdir(os.path.dirname(os.path.abspath(__file__)))
base_dir='../../'

def train_pomo():
    """
    No params, train on default settings
    """

    res=os.system(f'python POMO/CVRP/POMO/train.py ')
    if res:
        print('Train Done')
    return res

def train_am(data_dir='data/vrp/vrp100_validation_seed4321.pkl',graph_size=100,device_str='0,2,3'):
    """
    Params: training data dir, training data size, cuda device string
    """

    res=os.system(f'CUDA_VISIBLE_DEVICES={device_str} python AM/run.py  --output_dir AM/outputs --log_dir AM/logs --problem cvrp --graph_size {graph_size} --baseline rollout --run_name "cvrp_{graph_size}rollout" --val_dataset {data_dir}')
    if res:
        print('Train Done')
    return res

def train_amdkd():
    """
    No params, train on default settings
    """

    res=os.system('python AMDKD/CVRP/POMO/distill.py')
    if res:
        print('Train Done')
    return res

def train_neuopt(data_dir='datasets/cvrp_100.pkl',graph_size=100,device_str='0,2,3',run_name='training_CVRP100'):
    res=os.system(f'CUDA_VISIBLE_DEVICES={device_str} python NeuOpt/run.py --log_dir NeuOpt/logs --problem cvrp --val_dataset {data_dir} --dummy_rate 0.2 --graph {graph_size} --warm_up 0.25 --val_m 1 --T_train 250 --n_step 5 --batch_size 600 --epoch_size 12000 --max_grad_norm 0.05 --val_size 1000 --val_batch_size 1000 --T_max 1000 --stall 0 --k 4 --init_val_met random --run_name "{run_name}"')
    if res:
        print('Train Done')
    return res

def train_omni():
    """
    No params, train on default settings
    """

    res=os.system(f'python Omni/POMO/CVRP/train.py ')
    if res:
        print('Train Done')
    return res
    