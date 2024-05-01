##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
import argparse
from utils.utils import create_logger, copy_all_src

from CVRPTester import CVRPTester as Tester


##########################################################################################
# parameters

env_params = {
    'problem_size': 100,
    'pomo_size': 100,
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
}


tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': './result/saved_CVRP100_model',  # directory path of pre-trained model and log files saved.
        'epoch': 30500,  # epoch version of pre-trained model to laod.
    },
    'test_episodes': 10*1000,
    'test_batch_size': 1,
    'augmentation_enable': True,
    'aug_factor': 8,
    'aug_batch_size': 1,
    'test_data_load': {
        'enable': False,
        'filename': '../vrp100_test_seed1234.pt'
    },
    'test_set_path': '',# temp
    'test_set_opt_sol_path': '',# temp
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']


logger_params = {
    'log_file': {
        'desc': 'test_cvrp100',
        'filename': 'log.txt'
    }
}


##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    tester = Tester(env_params=env_params,
                      model_params=model_params,
                      tester_params=tester_params)

    copy_all_src(tester.result_folder)

    tester.run()


def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 10


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":
    # Ningjun Xu Modified due to test needs
    parser=argparse.ArgumentParser()
    parser.add_argument('--episode',type=int,default=1000)
    parser.add_argument('--batch',type=int,default=1)
    parser.add_argument('--augbatch',type=int,default=1)
    parser.add_argument('--testpaths',default='../../../../VRP-Omni-modified/Vrp-Set-A/A-n33')
    parser.add_argument('--solpaths',default='../../../../VRP-Omni-modified/Vrp-Set-A/A-n33-sol')
    parser.add_argument('--setname',default='A-n33')

    args=parser.parse_args()
    tester_params['test_episodes']=args.episode
    tester_params['test_batch_size']=args.batch
    tester_params['aug_batch_size']=args.augbatch
    tester_params['test_set_path']= args.testpaths
    tester_params['test_set_opt_sol_path']= args.solpaths
    tester_params['test_set_name']=args.setname
    main()
