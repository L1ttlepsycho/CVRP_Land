
import torch
import numpy as np
import os
import pickle
from logging import getLogger

from CVRPEnv import CVRPEnv as Env
from CVRPModel import CVRPModel as Model

from utils.utils import *


class CVRPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        
        # dataset init 
        self.path_list = None
        self.sol_path_list = None
        self.score_list, self.aug_score_list=[],[]

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # load dataset
        if tester_params['test_set_path'].endswith(".pkl"):
            self.test_data = load_dataset(tester_params['test_set_path'])[: self.tester_params['test_episodes']]
        else:
            # for solving instances with CVRPLIB format
            """
            Todo: load dataset from cvrp datasets
            """
            self.path_list = [os.path.join(tester_params['test_set_path'], f) for f in sorted(os.listdir(tester_params['test_set_path']))] \
                if os.path.isdir(tester_params['test_set_path']) else [tester_params['test_set_path']]
            print(self.path_list)
            assert self.path_list[-1].endswith(".vrp")

            self.test_data = self.pack_cvrplib(self.path_list)[: self.tester_params['test_episodes']]

        self.env_params['problem_size']= len(self.test_data[0][-2])
        self.env_params['pomo_size'] = len(self.test_data[0][-2])

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # Restore
        model_load = tester_params['model_load']
        if '.pt' in model_load['path']:
            checkpoint_fullname = model_load['path']
        else:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        print(checkpoint_fullname)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()

        

    def run(self):

        ##########################################################################################
        self.time_estimator.reset()

        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()

        if self.tester_params['test_data_load']['enable']:
            print(self.tester_params['test_data_load']['filename'])
            self.env.use_saved_problems(self.tester_params['test_data_load']['filename'], self.device)


        test_num_episode = len(self.test_data) #1000
        episode = 0

        inferTime = []
        while episode < test_num_episode:

            
            remaining = test_num_episode - episode
            #print('remain: ',remaining,' , test_batch size',self.tester_params['test_batch_size'])
            batch_size = min(self.tester_params['test_batch_size'], remaining)

             # load data
            data = self.test_data[episode: episode + batch_size]
            depot_xy, node_xy, node_demand, capacity = [i[0] for i in data], [i[1] for i in data], [i[2] for i in data], [i[3] for i in data]
            depot_xy, node_xy, node_demand, capacity = torch.Tensor(depot_xy), torch.Tensor(node_xy), torch.Tensor(node_demand), torch.Tensor(capacity)
            node_demand = node_demand / capacity.view(-1, 1)
            data = (depot_xy, node_xy, node_demand)
            #print(data)

            import time
            tik = time.time()
            #print('batch size: ',batch_size)
            score, aug_score,all_score,all_aug_score = self._test_one_batch(batch_size, episode,data=data)

            torch.cuda.synchronize()
            tok =time.time()
            inferTime.append(tok-tik)
            # print(np.mean(inferTime))

            score_AM.update(score)
            aug_score_AM.update(aug_score)

            self.score_list += all_score.tolist()
            self.aug_score_list += all_aug_score.tolist()

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score, aug_score))

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" NO-AUG SCORE: {} ".format(score_AM.avg))
                self.logger.info(" AUGMENTATION SCORE: {} ".format(aug_score_AM.avg))
            
        print(">> Evaluation on {} finished within {:.2f}s".format(self.tester_params['test_set_path'], tok-tik))

        # save results to file
        if not os.path.isdir('../../../../../results/mls/amdkd/'):
            os.mkdir('../../../../../results/mls/amdkd/')
        with open('../../../../../results/mls/amdkd/'+os.path.split(self.tester_params['test_set_path'])[-1], 'wb') as f:
            result = {"score_list": self.score_list, "aug_score_list": self.aug_score_list}
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
            print(">> Save final results to {}".format('res/'+os.path.split(self.tester_params['test_set_path'])[-1]))


        return score_AM.avg, aug_score_AM.avg, np.mean(inferTime)

    def _test_one_batch(self, batch_size,episode=None, data=None):

        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(batch_size, aug_factor,load_path=self.env_params['load_path'], episode=0,copy=data)
            # self.env.load_problems(batch_size, aug_factor)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state, attn_type='qk_scaled')

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, _ = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)

        # Return
        ###############################################
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)

        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_score = -max_pomo_reward[0, :].float()  # negative sign to make positive value
        no_aug_score_mean=no_aug_score.mean()

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward.float()
        aug_score_mean =aug_score.mean()  # negative sign to make positive value

        return no_aug_score_mean.item(),aug_score_mean.item(),no_aug_score, aug_score

    def pack_cvrplib(self,path_list):
        data_list=[]
        for path in path_list:
            #print('Solving .vrp file: ',path)
            file = open(path, "r")
            lines = [ll.strip() for ll in file]
            #print('lines: ',lines)
            i = 0
            while i < len(lines):
                line = lines[i]
                if line.startswith("DIMENSION"):
                    dimension = int(line.split(':')[1])
                elif line.startswith("CAPACITY"):
                    capacity = int(line.split(':')[1])
                elif line.startswith('NODE_COORD_SECTION'):
                    locations = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=int)
                    #print('locations: ',locations)
                    i = i + dimension
                elif line.startswith('DEMAND_SECTION'):
                    demand = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=int)
                    i = i + dimension
                i += 1
            original_locations = locations[:, 1:]
            original_locations = np.expand_dims(original_locations, axis=0)  # [1, n+1, 2]
            loc_scaler = 1000
            locations = original_locations / loc_scaler  # [1, n+1, 2]: Scale location coordinates to [0, 1]
            depot_xy, node_xy = locations[:, :1, :], locations[:, 1:, :]
            node_demand = demand[1:, 1:].reshape((1, -1))
            data = (depot_xy[0].tolist(), node_xy[0].tolist(), node_demand[0].tolist(),capacity)
            data_list.append(data)
            #print('dimension: ',dimension)
            #print('capacity: ',capacity)
            #print('depot_xy: ',depot_xy)
            #print('node_xy: ',node_xy)
            #print('demand: ',node_demand.shape)

            #print(path)
        #s=pickle.dumps(data_list)
        #with open('test.pkl','wb+') as f:
            #f.write(s)
        return data_list
    
    def pack_cvrp_sol(self,path_list):
        data_list=[]
        for path in path_list:
            #Todo
            #print('Solving .vrp file: ',path)
            file = open(path, "r")
            lines = [ll.strip() for ll in file]
            #print('lines: ',lines)
            i = 0
            while i < len(lines):
                line = lines[i]
                if line.startswith("Cost"):
                    cost = float(line.split(' ')[1])
                i += 1
            loc_scaler = 1000
            cost=cost/loc_scaler
            data_list.append(cost)
        #s=pickle.dumps(data_list)
        #with open('test_sol.pkl','wb+') as f:
            #f.write(s) 
        return data_list

def validate(model, env, batch_size, augment = True, load_path = None):

    # Augmentation
    ###############################################
    if augment:
        aug_factor = 8
    else:
        aug_factor = 1
    # Ready
    ###############################################
    model.eval()

    episode = 0
    test_num_episode = 1000
    no_aug_score1=[]
    aug_score1=[]
    while episode < test_num_episode:
        remaining = test_num_episode - episode
        batch_size = min(batch_size, remaining)
        with torch.no_grad():
            env.load_problems(batch_size, aug_factor, load_path=load_path, episode=episode)
            reset_state, _, _ = env.reset()
            model.pre_forward(reset_state)
        # POMO Rollout
        ###############################################
        state, reward, done = env.pre_step()
        while not done:
            selected, _ = model(state)
            # shape: (batch, pomo)
            state, reward, done = env.step(selected)
        ###############################################
        aug_reward = reward.reshape(aug_factor, batch_size, env.pomo_size)
        # shape: (augmentation, batch, pomo)
        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value
        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value
        no_aug_score1.append(no_aug_score.item())
        aug_score1.append(aug_score.item())
        episode += batch_size
    import numpy as np
    return np.mean(no_aug_score1),np.mean(aug_score1)