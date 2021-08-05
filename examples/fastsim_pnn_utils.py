# gym_fastsim imports
import gym
import gym_fastsim
real_env = gym.make('FastsimSimpleNavigationPos-v0') # Create the target environment
simplified_env = gym.make('FastsimEmptyMapNavigationPos-v0') # Create the simplified environment

# Model learning imports
from model_learning import ProbabilisticNeuralNetwork
from tensorflow import keras

# Robot controller imports
from model_learning import SimpleNeuralNetwork
from diversity_algorithms.controllers.fixed_structure_nn_numpy import SimpleNeuralControllerNumpy as SimpleNeuralController

# Utils imports
import numpy as np
import time

import datetime

## Utils methods
def normalize_standard(vector, mean_vector, std_vector):
  return [(vector[i] - mean_vector[i])/std_vector[i] for i in range(len(vector))]

def rescale_standard(vector, mean_vector, std_vector):
  return [vector[i]*std_vector[i] + mean_vector[i] for i in range(len(vector))]

def normalize_controller_input(vector):
  min_vector = [0, 0, -np.pi]
  max_vector = [600, 600, np.pi]
  return [(vector[i] - min_vector[i])/(max_vector[i]- min_vector[i]) for i in range(len(vector))]

# Model global learning params
means_in = None
stds_in = None
means_out = None
stds_out = None

num_epochs = 10 # number of epochs when learning on gathered data

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
eval_batch_size = 100

# env run params
init_random_trajs = 10

max_vel = 4

# some other global params
init_angle = np.pi/4
angular_state = np.array([np.sin(init_angle), np.cos(init_angle)])
init_state = np.concatenate((np.array([60., 450.]), angular_state))
horizon = 2000 # time steps on env (real and learned one)

#### Standard normalization ####
def normalize_data(data_in, data_out):
  global means_in, stds_in, means_out, stds_out
  normalized_data_in = np.zeros(data_in.shape)
  normalized_data_out = np.zeros(data_out.shape)
  
  means_in = [np.mean(data_in[:,dim]) for dim in range(input_dim)]
  stds_in = [np.std(data_in[:,dim]) for dim in range(input_dim)]
  means_out = [np.mean(data_out[:,dim]) for dim in range(output_dim)]
  stds_out = [np.std(data_out[:,dim]) for dim in range(output_dim)]

  for dim in range(input_dim):
    normalized_data_in[:, dim] = (data_in[:,dim] - means_in[dim])/stds_in[dim]
    
  for dim in range(output_dim):
    normalized_data_out[:, dim] = (data_out[:,dim] - means_out[dim])/stds_out[dim]

  return normalized_data_in, normalized_data_out

    
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
            to_input_controller = normalize_controller_input(to_input_controller)
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
        ## Compute fitness (sum of mean range over ball pose [x,y])
        mean_range = np.mean(trajs_stddev[:,:,i], axis=0)
        # fitness[i] = -(mean_range[0] + mean_range[1]) # negative fitness will push evolution towwards "certain" individuals
        fitness[i] = (mean_range[0] + mean_range[1]) # positive fitness will push evolution towwards "uncertain" individuals

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

    if(display):
        env.enable_display()
        
    prev_action = None
    prev_obs = None

    controller_nn = SimpleNeuralController(controller_input_dim, controller_output_dim, params=controller_nnparams)

    controller_nn.set_parameters(genotype) # set the controller with the individual weights

    tab_cpt = 0
    # T time steps, but "done" can be attained before T is reached
    for t in range(horizon):
        if(display):
            env.render()

        # action = controller_nn(obs) # compute next action
        action = controller_nn(normalize_controller_input(obs)) # compute next action

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
            time.sleep(0.001)

    # format data
    data_in_no_0s = data_in[~np.all(data_in == 0, axis=1)]
    data_out_no_0s = data_out[~np.all(data_in == 0, axis=1)] # filter out the same lines as for data_in
    return data_in_no_0s, data_out_no_0s, obs
