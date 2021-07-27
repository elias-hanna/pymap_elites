# CVT map elite imports
# from map_elites import compute
import map_elites.cvt as cvt_map_elites
import map_elites.common as cm_map_elites

# gym_fastsim imports
import gym
import gym_fastsim
real_env = gym.make('FastsimSimpleNavigationPos-v0') # Create the target environment
simplified_env = gym.make('FastsimEmptyMapNavigationPos-v0') # Create the simplified environment

# Model learning imports
from model_learning import ProbabilisticNeuralNetwork
import tensorflow as tf
from tensorflow import keras

# Robot controller imports
from model_learning import SimpleNeuralNetwork
from diversity_algorithms.controllers.fixed_structure_nn_numpy import SimpleNeuralControllerNumpy as SimpleNeuralController

# Utils imports
import numpy as np
import time
import multiprocessing
from multiprocessing import set_start_method

import datetime
import os

## Utils methods
def normalize_standard(vector, mean_vector, std_vector):
  return [(vector[i] - mean_vector[i])/std_vector[i] for i in range(len(vector))]

def rescale_standard(vector, mean_vector, std_vector):
  return [vector[i]*std_vector[i] + mean_vector[i] for i in range(len(vector))]

# Model global learning params
means_in = None
stds_in = None
means_out = None
stds_out = None

num_epochs = 30 # number of epochs when learning on gathered data

hidden_units = [500,500,500]
batch_size = 256
train_prop = 0.85 # ratio in data of training / testing
learning_rate = 0.001
input_dim = real_env.action_space.shape[0] + real_env.observation_space.shape[0] + 1 # +1 to make up for the angular state that is divided in two
output_dim = real_env.observation_space.shape[0] + 1 # +1 to make up for the angular state that is divided in two

pnn = ProbabilisticNeuralNetwork(input_dim=input_dim, output_dim=output_dim, train_prop=train_prop,
                                 batch_size=batch_size, learning_rate=learning_rate, hidden_units=hidden_units)

# Robot controller params
controller_input_dim = real_env.observation_space.shape[0]
controller_output_dim = real_env.action_space.shape[0]
controller_nnparams={"n_hidden_layers": 2, "n_neurons_per_hidden": 10} # same params as ran NS experiment 
n_weights = SimpleNeuralController(controller_input_dim, controller_output_dim, params=controller_nnparams).n_weights

# cvt map elites global params
eval_batch_size = 1000

# env run params
init_random_trajs = 100

max_vel = 4

# some other global params
init_angle = np.pi/4
angular_state = np.array([np.sin(init_angle), np.cos(init_angle)])
init_state = np.concatenate((np.array([60., 450.]), angular_state))
horizon = 2000 # time steps on env (real and learned one)

def pnn_eval(to_evaluate): 
    inputs, pnn_weights = to_evaluate

    pnn_loc = ProbabilisticNeuralNetwork(input_dim=input_dim, output_dim=output_dim, train_prop=train_prop,
                                batch_size=batch_size, learning_rate=learning_rate, hidden_units=hidden_units)
    pnn_loc.create_model() # can put a dummy val because model isn't trained here
    pnn_loc.model.set_weights(pnn_weights)
    return np.transpose(pnn_loc.model(inputs))

# takes genotype of all population (controller parameters)
# returns list of fitness , behavior descriptor for each individual
def fastsim_eval(xx):
    ##### 0 ####
    ## Initialize the controllers with population genotypes ##
    controllers = []
    for i in range(len(xx)):
        # print(i,"th controller\n",xx[i])
        controllers.append(SimpleNeuralController(controller_input_dim,
                                                  controller_output_dim,
                                                  params=controller_nnparams))
        controllers[-1].set_parameters(xx[i])

    ## Initialize data fields ##
    trajs = np.zeros((horizon, output_dim, len(xx)))
    trajs_stddev = np.zeros((horizon, output_dim, len(xx)))
    for i in range(len(xx)):
        trajs[0,:,i] = init_state

    ## Create a local version of pnn ##
    pnn_weights = pnn.model.get_weights()

    pnn_loc = ProbabilisticNeuralNetwork(input_dim=input_dim, output_dim=output_dim,
                                         train_prop=train_prop, batch_size=batch_size,
                                         learning_rate=learning_rate, hidden_units=hidden_units)
    pnn_loc.create_model()
    pnn_loc.model.set_weights(pnn_weights)
    #### 1 ####
    ## sequential evaluation
    ts2 = datetime.datetime.now()
    for t in range(horizon-1):
        to_input = np.zeros((len(xx), input_dim))
        for i in range(len(xx)):
            prev_step = np.transpose(trajs[t,:,i])
            # Compute action given last observed state/predicted state
            to_input_controller = np.concatenate((prev_step[:2], 
                                                  [np.arctan2(prev_step[2], prev_step[3])]))
            action = controllers[i](to_input_controller) # compute next action
            action[action>max_vel] = max_vel
            action[action<-max_vel] = -max_vel
            # Create input vector to give to transition model PNN
            to_input[i,:2] = action
            to_input[i,2:] = np.transpose(trajs[t,:,i])
            to_input[i,:] = normalize_standard(to_input[i,:], means_in, stds_in) 
        # Predict using model
        output_distribution = pnn_loc.model(to_input)
        output_mean = output_distribution.mean().numpy()
        trajs[t+1,:,:] = np.transpose(output_mean)
        output_stdv = output_distribution.stddev().numpy()
        trajs_stddev[t+1,:,:] = np.transpose(output_stdv[:,:])
        for i in range(len(xx)): # rescale the transition model output
            trajs[t+1,:,i] = rescale_standard(trajs_stddev[t+1,:,i], means_out, stds_out)
            trajs[t+1,:,i] = rescale_standard(trajs[t+1,:,i], means_out, stds_out)
            trajs[t+1,:,i] += trajs[t,:,i] # add previous state as we predict st+1 - st
    te2 = datetime.datetime.now()
    print("Time for sequential: ", te2-ts2)
    
    ## Compute fitness and bd for each indivs
    fitness = [0.]*len(xx)
    bd = [[0.,0.]]*len(xx)
    ## iterate over all individuals
    for i in range(len(xx)):
        ## Compute BD (last ball position of closest observed trajectory to mean trajectory
        bd[i] = trajs[-1,:2,i]
        print(bd[i])
        ## Compute fitness (sum of mean range over ball pose [x,y])
        mean_range = np.mean(trajs_stddev[:,:,i], axis=0)
        fitness[i] = -(mean_range[0] + mean_range[1])

    return fitness, bd

# takes genotype of all population (controller parameters)
# returns list of fitness , behavior descriptor for each individual
def real_env_eval(xx):

    ## 1 ##
    ## 1st step on len(xx) different models (to have a first different step)

    # M = 10 # number of iterations on model
    
    trajs = np.zeros((horizon, output_dim, len(xx)))
    for i in range(len(xx)):
        trajs[0,:, i] = init_state
    
        data_in, data_out = run_on_gym_env(real_env, xx[i], horizon)

        trajs[:-1,:,i] = data_in[:,2:]
        trajs[-1,:,i] = data_out[-1,:] + data_in[-1,2:]
    ## Compute fitness and bd for each indivs
    fitness = [0.]*len(xx)
    bd = [[0.,0.]]*len(xx)
    ## iterate over all individuals
    for i in range(len(xx)):
        loc_traj = trajs[:,:,i]
        tmp_loc_traj = loc_traj[~np.all(loc_traj == 0, axis=1)]
        
        ## Compute BD (last ball position of closest observed trajectory to mean trajectory
        bd[i] = tmp_loc_traj[-1,:2]
        ## Compute fitness (sum of mean range over ball pose [x,y])
        fitness[i] = np.random.rand()

    return fitness, bd

# genotype shape for FastsimSimpleNavigationPos env: (number of weights of nn controller,)
def run_on_gym_env(env, genotype, horizon, display=False):
    # Initialize data containers that will retain whole trajectory
    data_in = np.zeros((horizon, input_dim))
    data_out = np.zeros((horizon, output_dim))
    to_input_controller = np.zeros((1, controller_input_dim))

    obs = env.reset()
    
    prev_action = None
    prev_obs = None

    controller_nn = SimpleNeuralController(controller_input_dim, controller_output_dim, params=controller_nnparams)

    controller_nn.set_parameters(genotype) # set the controller with the individual weights

    tab_cpt = 0
    # T time steps, but "done" can be attained before T is reached
    for t in range(horizon):
        if(display):
            env.render()

        action = controller_nn(obs) # compute next action

        action[action>max_vel] = max_vel
        action[action<-max_vel] = -max_vel

        prev_action = action
        prev_obs = obs
    
        obs, reward, done, info = env.step(action)

        if(t==0):
            continue
        
        ## transform angular state so its continuous
        prev_angular_state = np.array([np.sin(prev_obs[-1]), np.cos(prev_obs[-1])])
        angular_state = np.array([np.sin(obs[-1]), np.cos(obs[-1])])

        ## Output of dynamic model is the difference between next and current state st+1-st
        data_out[tab_cpt] = np.concatenate((np.subtract(obs[:-1], prev_obs[:-1]), np.subtract(angular_state, prev_angular_state)))
        data_in[tab_cpt] = np.concatenate((prev_action, prev_obs[:-1], prev_angular_state), axis=None)
        tab_cpt += 1

        if done:
            break
    
        if(display):
            time.sleep(0.0001)

    # format data
    data_in_no_0s = data_in[~np.all(data_in == 0, axis=1)]
    data_out_no_0s = data_out[~np.all(data_in == 0, axis=1)] # filter out the same lines as for data_in
    return data_in_no_0s, data_out_no_0s
    
if __name__=='__main__':
    ### run params ###
    # M = 10 # number of iterations on model
    # pool = multiprocessing.Pool(processes=M,maxtasksperchild=1) # create M parallel evaluation pools

    # cvt map elites params
    params = \
    {
        "cvt_samples": 25000,
        "batch_size": eval_batch_size,
        "random_init": 0.01,
        "random_init_batch": eval_batch_size,
        "iso_sigma": 0.01,
        "line_sigma": 0.2,
        "dump_period": 2,
        "parallel": True,
        "cvt_use_cache": True,
        "min": -5,
        "max": 5,
        "bd_min": [0, 0],
        "bd_max": [600, 600]
    }

    dim_map = 2
    dim_gen = n_weights
    n_niches = 10000
    n_gen = 10
    
    # data containers
    data_in = np.zeros((horizon*init_random_trajs, input_dim))
    data_out = np.zeros((horizon*init_random_trajs, output_dim))

    #-1#
    # Do init_evals trajectories on empty_env with NS algorithm and save the traj data for model learning. Idea is to first learn robot model.
    pop_size = 100
    nb_gen = 50
    # Population initialization
    population = []
    for i in range(pop_size):
        rand_genotype = np.random.uniform(low=params["min"], high=params["max"], size=(n_weights,))
        population.append(rand_genotype)

    for i in range(nb_gen):
        # -1)A) Evaluate the population
        for j in range(pop_size):
            data_in_to_add, data_out_to_add = run_on_gym_env(simplified_env,
                                                             population[j], horizon)
        
        # -1)B) Update novelty score and archive

        # -1)C) Selection and variation
    

    #0#
    
    # Do init_random_trajs trajectories on real_env with random genotype and save the traj data for model learning
    # input: genotype, horizon
    # output: trajs
    # Note: this type of init works for envs that are run with a unique action at beginning (not action trajs)
    tab_cpt = 0
    for i in range(init_random_trajs):
        # Create a random genotype
        genotype = np.random.uniform(low=params["min"], high=params["max"], size=(n_weights,))

        data_in_to_add, data_out_to_add = run_on_gym_env(real_env, genotype, horizon)

        data_in[tab_cpt:tab_cpt+len(data_in_to_add),:] = data_in_to_add
        data_out[tab_cpt:tab_cpt+len(data_out_to_add),:] = data_out_to_add
               
        print("{:.1f}".format(i/init_random_trajs*100),"% done", end="\r")
        tab_cpt += len(data_in_to_add)

    # filter out 0 lines that were left
    data_in_no_0s = data_in[~np.all(data_in == 0, axis=1)] 
    data_out_no_0s = data_out[~np.all(data_in == 0, axis=1)]

    # for i in range(len(data_in)):
    #     print(data_in[i], " - ", data_out[i])
    
    # unique_data_in, index = np.unique(data_in, axis=0, return_index=True)
    # unique_data_out = np.take(data_out, index, axis=0)
    # for i in range(init_random_trajs):
    #     print("example")
    #     print(unique_data_in[i])
    #     print("corresponding label")
    #     print(unique_data_out[i])

    max_iter = 10
    convergence_thresh = 0.1
    has_converged = False # If uncertainty of less certain trajectory is below threshold ?
    while (max_iter > 0 or not has_converged):

        #1#
        # Normalize training data
        normalized_data_in = np.zeros(data_in_no_0s.shape)
        normalized_data_out = np.zeros(data_out_no_0s.shape)

        #### Standard normalization ####
        means_in = [np.mean(data_in_no_0s[:,dim]) for dim in range(input_dim)]
        stds_in = [np.std(data_in_no_0s[:,dim]) for dim in range(input_dim)]
        means_out = [np.mean(data_out_no_0s[:,dim]) for dim in range(output_dim)]
        stds_out = [np.std(data_out_no_0s[:,dim]) for dim in range(output_dim)]

        for dim in range(input_dim):
            normalized_data_in[:, dim] = (data_in_no_0s[:,dim] - means_in[dim])/stds_in[dim]

        for dim in range(output_dim):
            normalized_data_out[:, dim] = (data_out_no_0s[:,dim] - means_out[dim])/stds_out[dim]
        
        # Learn a model from the gathered data
        train_dataset, test_dataset = pnn.get_train_and_test_splits(normalized_data_in, normalized_data_out)

        # simple_nn = SimpleNeuralNetwork(input_dim=input_dim, output_dim=output_dim, train_prop=train_prop,
                                        # batch_size=batch_size, learning_rate=learning_rate, hidden_units=hidden_units,
                                        # loss=loss)
        # simple_nn.create_model()
        # simple_nn.run_experiment(train_dataset, test_dataset, num_epochs)
        pnn.create_model()
        pnn.run_experiment(train_dataset, test_dataset, num_epochs)

        # Perform CVT map elites computation on learned model
        # archive = compute(dim_map, dim_gen, fastsim_eval, n_niches=n_niches, n_gen=n_gen, params=params)
        # archive is made of a collection of species
        # archive = compute(dim_map, dim_gen, fastsim_test_eval, n_niches=n_niches, n_gen=n_gen, params=params)
        archive = compute_step(dim_map, dim_gen, fastsim_eval, n_niches=n_niches, n_gen=n_gen, params=params)
        # archive = compute_step(dim_map, dim_gen, real_env_eval, n_niches=n_niches, n_gen=n_gen, params=params) # just to test cvt archive

        #3#

        # Get the N most uncertain individuals and test them on real setup to gather data
        N = 3 # Number of individuals that we'll try on real_env

        sorted_archive = sorted( archive.items(), key=lambda pair: pair[1].fitness, reverse=True )[-N:]

        print("N most uncertain archive individuals fitnesses:")
        for i in sorted_archive:
            print(i[1].fitness)

        if (sorted_archive[0][1].fitness > -convergence_thresh):
            print("Algorithm has converged")
            break
        
        #4#

        # reset data containers while maintaining previous data
        tmp_data_in = np.zeros((len(data_in_no_0s)+horizon*N, input_dim))
        tmp_data_out = np.zeros((len(data_out_no_0s)+horizon*N, output_dim))

        tmp_data_in[:tab_cpt,:] = data_in_no_0s
        tmp_data_out[:tab_cpt,:] = data_out_no_0s
        data_in = tmp_data_in
        data_out = tmp_data_out

        # Test the N most uncertain individuals on real setup to gather new data
        for i in range(min(N, len(sorted_archive))):
            data_in_to_add, data_out_to_add = run_on_gym_env(real_env,
                                                             sorted_archive[i][1].x, horizon)
        
            data_in[tab_cpt:tab_cpt+len(data_in_to_add),:] = data_in_to_add
            data_out[tab_cpt:tab_cpt+len(data_in_to_add),:] = data_out_to_add

            tab_cpt += len(data_in_to_add)

        # filter out 0 lines that were left
        data_in_no_0s = data_in[~np.all(data_in == 0, axis=1)] 
        data_out_no_0s = data_out[~np.all(data_in == 0, axis=1)]

        # Reduce iteration number
        max_iter -= 1

    print("Finished learning.")