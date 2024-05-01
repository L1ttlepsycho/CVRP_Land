import time
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import os
import argparse
import vrplib
import numpy as np
import pandas as pd
import pickle

root_dir=''
save_dir=''
path_list=[]
k_dict={}



def cal_euc_2d(src,dst):
    return ((src[0]-dst[0])**2+(src[1]-dst[1])**2)**0.5

# Todo: resolving vrp files
def resolve_vrp(instance_path):
    instance = vrplib.read_instance(instance_path)
    data={}
    distance_type = instance['edge_weight_type']
    coord_matrix=instance['node_coord']
    #print(instance)
    data['distance_matrix']=coord_to_distance(coord_matrix=coord_matrix,distance_type=distance_type)
    data["num_vehicles"] = resolve_vehicles(path=instance_path[0:-4])
    data["demands"] = instance['demand'].tolist()
    data["vehicle_capacities"] = [instance['capacity']]*data['num_vehicles']
    data["depot"] = 0

    print('Path: ',instance_path)
    print('Num of Vehicles: ',data['num_vehicles'])
    return data

# Todo   
def resolve_vehicles(path='',comment=''):
    num=0
    if path!='':
        index=path.find('-k')
        if index>-1:
            num=int(path[index+2:])
    return num

def coord_to_distance(coord_matrix,distance_type):
    cm_size=coord_matrix.shape[0]
    distance_matrix=np.zeros((cm_size,cm_size)).tolist()
    #print(cm_size)

    if distance_type=='EUC_2D': 
        # Todo 
        for i in range(cm_size):
            for j in range(cm_size):
                distance_matrix[i][j]=round(cal_euc_2d(coord_matrix[i],coord_matrix[j]))
    return distance_matrix

def print_solution(data, manager, routing, solution,time,path,save_path):
    """Prints solution on console."""
    #print(f"Objective: {solution.ObjectiveValue()}")
    total_distance = 0
    total_load = 0
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data["demands"][node_index]

            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
        total_distance += route_distance
        total_load += route_load
    print(f"Total distance of all routes: {total_distance:.2f}m")
    print(f"Total load of all routes: {total_load}")
    print(f"Elapsed Time: {time:.3f}")
    print('--------------------------------------------------------------------------------------------')
    sol_data=(total_distance,time)
    data_list=[]
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    path,fname=os.path.split(path)
    fname=save_path+'/'+fname[0:-4]+'-sol.pkl'
    if os.path.isdir(fname):
        data_list=pickle.load(fname)
    data_list.append(sol_data)
    s=pickle.dumps(data_list)
    with open(fname,'wb+') as f:
        f.write(s) 

def run_or_tools(instance_path,save_path):
    """Entry point of the program."""
    # Instantiate the data problem.
    # data=resolve_vrp('tai75a')
    
    data=resolve_vrp(instance_path)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data["demands"][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data["vehicle_capacities"],  # vehicle maximum capacities
        True,  # start cumul to zero
        "Capacity",
    )
    
    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint.
    dimension_name = "Distance"
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        30000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name,
    )
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    
    #search_parameters.local_search_metaheuristic = (
        #routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    #)
    #search_parameters.log_search = True

    # Setting maximum searching time
    search_parameters.time_limit.seconds = 1000

    start_time=time.time()
    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)
    sol_time=time.time()-start_time

    # Print solution on console.
    if solution:
        print_solution(data, manager, routing, solution,sol_time,instance_path,save_path)
    else:
        print("No solution found !")
        print('--------------------------------------------------------------------------------------------')

def main():
    for sub_path in path_list:
        for path in sub_path:
            if path.endswith('.vrp'):
                run_or_tools(path,save_dir)

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset_dir',default='../dataset/')
    parser.add_argument('--results_dir',default='../results/cvrp_ortools')

    args=parser.parse_args()

    root_dir=args.dataset_dir
    save_dir=args.results_dir

    for dir in sorted(os.listdir(root_dir)):
        dir=os.path.join(root_dir,dir)
        #print(dir)
        t_list=[os.path.join(dir, f) for f in sorted(os.listdir(dir))] \
            if os.path.isdir(dir) else [dir]
        path_list.append(t_list)
    main()