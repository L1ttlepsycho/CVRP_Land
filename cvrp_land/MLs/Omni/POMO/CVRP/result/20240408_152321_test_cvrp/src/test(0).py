import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for utils

import argparse
import numpy
import torch
import logging
from utils.utils import create_logger, copy_all_src
from utils.functions import seed_everything, check_null_hypothesis
from CVRPTester import CVRPTester as Tester

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE and torch.cuda.is_available()
CUDA_DEVICE_NUM = 0

##########################################################################################
# parameters

env_params = {
    'problem_size': 200,
    'pomo_size': 200,
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
    'norm': 'batch_no_track'
}


tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'seed': 2024,
    'model_load': {
        'path': 'result/maml_250000checkpts',  # directory path of pre-trained model and log files saved.
        'epoch': 250000,  # epoch version of pre-trained model to load.
    },
    'test_episodes': 100,
    'test_batch_size': 1,
    'augmentation_enable': True,
    'test_robustness': False,
    'aug_factor': 8, #8
    'aug_batch_size': 1,#100
    'test_set_path': '../../data/CVRP/A_33',#'../../data/CVRP/Size_Distribution/cvrp200_rotation.pkl' '../../data/CVRP/test_i'
    'test_set_opt_sol_path': '../../data/CVRP/A_33_sol',#'../../data/CVRP/Size_Distribution/hgs/cvrp200_rotationoffset0n1000-hgs.pkl' '../../data/CVRP/test_s'
    'test_set_name': 'A-n33'
}

fine_tune_params = {
    'enable': False,  # evaluate few-shot generalization
    'fine_tune_episodes': 1000,  # how many data used to fine-tune the pretrained model
    'k': 10,  # fine-tune steps/epochs
    'fine_tune_batch_size': 10,  # the batch size of the inner-loop optimization
    'augmentation_enable': True,
    'optimizer': {
        'lr': 1e-4 * 0.1,
        'weight_decay': 1e-6
    }
}

if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']

logger_params = {
    'log_file': {
        'desc': 'test_cvrp',
        'filename': 'log.txt'
    }
}


def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    seed_everything(tester_params['seed'])

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params,
                    fine_tune_params=fine_tune_params)

    copy_all_src(tester.result_folder)

    tester.run()


def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 1


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


def t_test(path1, path2):
    """
    Conduct T-test to check the null hypothesis. If p < 0.05, the null hypothesis is rejected.
    """
    import pickle
    with open(path1, 'rb') as f1:
        results1 = pickle.load(f1)
    with open(path2, 'rb') as f2:
        results2 = pickle.load(f2)
    check_null_hypothesis(results1["score_list"], results2["score_list"])
    check_null_hypothesis(results1["aug_score_list"], results2["aug_score_list"])


if __name__ == "__main__":
    # Ningjun Xu Modified due to test needs
    parser=argparse.ArgumentParser()
    parser.add_argument('--episode',type=int,default=1000)
    parser.add_argument('--batch',type=int,default=1)
    parser.add_argument('--augbatch',type=int,default=1)
    parser.add_argument('--testpaths',default='../../data/CVRP/VRP_Datasets/Set_A/A-n33')
    parser.add_argument('--solpaths',default='../../data/CVRP/VRP_Datasets/Set_A/A-n33-sol')
    parser.add_argument('--setname',default='A-n33')

    args=parser.parse_args()
    tester_params['test_episodes']=args.episode
    tester_params['test_batch_size']=args.batch
    tester_params['aug_batch_size']=args.augbatch
    tester_params['test_set_path']= args.testpaths
    tester_params['test_set_opt_sol_path']= args.solpaths
    tester_params['test_set_name']=args.setname
    main()
