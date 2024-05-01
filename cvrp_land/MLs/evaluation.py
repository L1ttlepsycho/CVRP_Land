import os
import pickle
#os.chdir(os.path.dirname(os.path.abspath(__file__)))
base='../../'

def eval_pomo(test_path='../Dataset'):
    """
    path base: 'cvrp_land/MLs'
    Params:  test_path: where datasets.pkl stores

    """

    path_list=[]

    for dir in sorted(os.listdir(test_path)):
        dir=os.path.join(test_path,dir)
        #print(dir)
        list=[os.path.join(dir, f) for f in sorted(os.listdir(dir))] \
            if os.path.isdir(dir) else [dir]
        path_list.append(list)
    #print(path_list)
    for sub_list in path_list:
        res=0
        for path in sub_list:
            if not path.endswith('sol') and not path.endswith('.pkl') and not path.endswith('.ipynb'):
                res=os.system(f'python -u POMO/CVRP/POMO/test.py  --testpaths ../../../{path}')
                if res==0:
                    print('Test on "',path,'" Success')
                    print('-----------------------------------------------------------------------------------------------------------------')
                    del path
                else: 
                    print('Check out issue on set: ',path)


def eval_am(test_path='AM/data/vrp/scale-1000',model_dir='AM/outputs/cvrp_100/cvrp100_rollout'):
    
    """
    path base: 'cvrp_land/MLs'
    Params:  test_path: where datasets.pkl stores

    """

    path_list=[os.path.join(test_path, f) for f in sorted(os.listdir(test_path))] \
        if os.path.isdir(test_path) else [test_path]

    for path in path_list:
        res=os.system(f'python AM/eval.py --results_dir ../../results/mls/am --datasets {path} --model AM/outputs/cvrp_100/cvrp100_rollout --decode_strategy greedy -f')
        if res==0:
            print('baseline on "',path,'" Success')
            del path
        else: 
            print('Check out issue')


def eval_amdkd(test_path='../Dataset'):
    """
    path base: 'cvrp_land/MLs'
    Params:  test_path: where datasets.pkl stores

    """

    path_list=[]

    for dir in sorted(os.listdir(test_path)):
        dir=os.path.join(test_path,dir)
        #print(dir)
        list=[os.path.join(dir, f) for f in sorted(os.listdir(dir))] \
            if os.path.isdir(dir) else [dir]
        path_list.append(list)
    #print(path_list)
    for sub_list in path_list:
        res=0
        for path in sub_list:
            if not path.endswith('sol') and not path.endswith('.pkl') and not path.endswith('.ipynb'):
                res=os.system(f'python -u AMDKD/CVRP/POMO/test.py  --testpaths ../../../{path}')
                if res==0:
                    print('Test on "',path,'" Success')
                    print('-----------------------------------------------------------------------------------------------------------------')
                    del path
                else: 
                    print('Check out issue on set: ',path)



def eval_neuopt(test_path='../Dataset/scale-1000',device_str='0'):
    """
    path base: 'cvrp_land/MLs'
    Params:  test_path: where datasets.pkl stores
             device_str: CUDA devices string

    """
    path_list = [os.path.join(test_path, f) for f in sorted(os.listdir(test_path))] \
        if os.path.isdir(test_path) else [test_path]
    #print(path_list)

    for path in path_list:
        if path.endswith('.pkl'):
            n=int(path.split('-')[-1][1:-4])-1
            data=pickle.load(open(path,'rb+'))
            ep=len(data)

            res=os.system(f'CUDA_VISIBLE_DEVICES={device_str} python NeuOpt/run.py --eval_only --no_saving --no_tb --init_val_met random --val_size {ep} --val_batch_size 1 --k 4 --problem cvrp --val_dataset {path} --graph_size {n} --dummy_rate 0.2 --val_m 5 --stall 10 --T_max 1000 --load_path NeuOpt/pre-trained/cvrp200.pt')
            if res==0:
                print('Test on "',path,'" Success')
                print('-----------------------------------------------------------------------------------------------------------------')
                del path
            else: 
                print('Check out issue on set: ',path)


def eval_omni(test_path='../Dataset'):
    """
    path base: 'cvrp_land/MLs'
    Params:  test_path: where datasets.pkl stores

    """

    path_list=[]

    for dir in sorted(os.listdir(test_path)):
        dir=os.path.join(test_path,dir)
        #print(dir)
        list=[os.path.join(dir, f) for f in sorted(os.listdir(dir))] \
            if os.path.isdir(dir) else [dir]
        path_list.append(list)
    #print(path_list)
    for sub_list in path_list:
        res=0
        for path in sub_list:
            if not path.endswith('sol') and not path.endswith('.pkl') and not path.endswith('.ipynb'):
                ep=len(os.listdir(path))
                batch_size=1
                res=os.system(f'python -u Omni/POMO/CVRP/test.py  --episode {ep} --testpaths ../../../{path} --solpaths ../../../{path}-sol')
                if res==0:
                    print('Test on "',path,'" Success')
                    print('-----------------------------------------------------------------------------------------------------------------')
                    del path
                else: 
                    print('Check out issue on set: ',path)