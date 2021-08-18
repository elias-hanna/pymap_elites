# CVT map elite imports
# from map_elites import compute
import map_elites.cvt as cvt_map_elites
import map_elites.common as cm_map_elites

# from scipy.spatial import cKDTree : TODO -- faster?
from sklearn.neighbors import KDTree

# Utils import
import fastsim_pnn_utils as fpu
import numpy as np

# Env imports
import gym
import gym_learned_model

# MCTS imports
from mcts.DPW import DPW

if __name__=='__main__':
    ### run params ###
    # M = 10 # number of iterations on model
    # pool = multiprocessing.Pool(processes=M,maxtasksperchild=1) # create M parallel evaluation pools

    # cvt map elites params
    params = \
    {
        "cvt_samples": 25000,
        "batch_size": fpu.eval_batch_size,
        "random_init": 0.01,
        "random_init_batch": fpu.eval_batch_size,
        "iso_sigma": 0.01,
        "line_sigma": 0.2,
        "dump_period": -1,
        "parallel": True,
        "cvt_use_cache": True,
        "min": -5,
        "max": 5,
        "bd_min": [-10, -10, -np.pi],
        "bd_max": [10, 10, np.pi]
    }

    dim_map = 3
    dim_gen = fpu.n_weights
    n_niches = 1000
    n_gen = 50
    max_evals = n_gen*fpu.eval_batch_size

    real_env_evals = 0
    learned_env_evals = 0

    # data containers
    data_in = np.zeros((fpu.horizon*fpu.init_random_trajs, fpu.input_dim))
    data_out = np.zeros((fpu.horizon*fpu.init_random_trajs, fpu.output_dim))

    #-1#
    # Do init_evals trajectories on empty_env with NS algorithm and save the traj data for model learning. Idea is to first learn robot model.
    pop_size = 100
    nb_gen = 50
    # Population initialization
    population = []
    # for i in range(pop_size):
    #     rand_genotype = np.random.uniform(low=params["min"], high=params["max"], size=(fpu.n_weights,))
    #     population.append(rand_genotype)

    # for i in range(nb_gen):
    #     # -1)A) Evaluate the population
    #     for j in range(pop_size):
    #         data_in_to_add, data_out_to_add = fpu.run_on_gym_env(fpu.simplified_env,
    #                                                          population[j], fpu.horizon)
        
        # -1)B) Update novelty score and archive

        # -1)C) Selection and variation
    

    #-1#
    # Do init_evals trajectories on empty_env with MAP-Elites algorithm and save the traj data for model learning.
    init_evals = 50*fpu.eval_batch_size
    # surrogate_archive, n_evals = cvt_map_elites.compute(dim_map, dim_gen, fpu.real_env_eval,
    surrogate_archive, n_evals = cvt_map_elites.compute(dim_map, dim_gen, fpu.simplified_env_eval,
                                                        n_niches=n_niches, max_evals=init_evals,
                                                        params=params, all_pop_at_once=True,
                                                        iter_number=-1)
    # learned_env_evals += n_evals
    # real_env_evals += n_evals

    cm_map_elites.__save_archive(surrogate_archive, max_evals, -1, total_evals=0)

    # data containers
    data_in = np.zeros((fpu.horizon*len(surrogate_archive), fpu.input_dim))
    data_out = np.zeros((fpu.horizon*len(surrogate_archive), fpu.output_dim))

    # 0 bis#
    print("Archive len: ", len(surrogate_archive))
    tab_cpt = 0
    for i, niche in zip(range(len(surrogate_archive)), surrogate_archive):
        # Create a random genotype
        genotype = surrogate_archive[niche].x

        data_in_to_add, data_out_to_add, last_obs = fpu.run_on_gym_env(fpu.real_env,
                                                                       genotype,
                                                                       fpu.horizon
        )

        data_in[tab_cpt:tab_cpt+len(data_in_to_add),:] = data_in_to_add
        data_out[tab_cpt:tab_cpt+len(data_out_to_add),:] = data_out_to_add
               
        print("{:.1f}".format(i/len(surrogate_archive)*100),"% done", end="\r")
        tab_cpt += len(data_in_to_add)
        real_env_evals += 1

    # filter out 0 lines that were left
    data_in_no_0s = data_in[~np.all(data_in == 0, axis=1)] 
    data_out_no_0s = data_out[~np.all(data_in == 0, axis=1)]
    
    #0#
    # Do init_random_trajs trajectories on real_env with random genotype and save the traj data for model learning
    # input: genotype, horizon
    # output: trajs
    # Note: this type of init works for envs that are run with a unique action at beginning (not action trajs)
    # tab_cpt = 0
    # for i in range(fpu.init_random_trajs):
    #     # Create a random genotype
    #     genotype = np.random.uniform(low=params["min"], high=params["max"], size=(fpu.n_weights,))

    #     data_in_to_add, data_out_to_add, last_obs = fpu.run_on_gym_env(fpu.real_env,
    #                                                                    genotype,
    #                                                                    fpu.horizon
    #     )

    #     data_in[tab_cpt:tab_cpt+len(data_in_to_add),:] = data_in_to_add
    #     data_out[tab_cpt:tab_cpt+len(data_out_to_add),:] = data_out_to_add
               
    #     print("{:.1f}".format(i/fpu.init_random_trajs*100),"% done", end="\r")
    #     tab_cpt += len(data_in_to_add)
    #     real_env_evals += 1

    # # filter out 0 lines that were left
    # data_in_no_0s = data_in[~np.all(data_in == 0, axis=1)] 
    # data_out_no_0s = data_out[~np.all(data_in == 0, axis=1)]

    max_iter = 2
    itr = 0
    convergence_thresh = 0.1
    has_converged = False # While uncertainty of less certain trajectory is below threshold ?
    surrogate_archive = {}
    real_archive = {}
    while (itr < max_iter and not has_converged):

        #1#
        # Normalize training data
        normalized_data_in, normalized_data_out = fpu.normalize_data(data_in_no_0s, data_out_no_0s)
        
        # Learn a model from the gathered data
        train_dataset, test_dataset = fpu.pnn.get_train_and_test_splits(normalized_data_in, normalized_data_out)

        # simple_nn = SimpleNeuralNetwork(input_dim=fpu.input_dim, output_dim=fpu.output_dim, train_prop=train_prop,
                                        # batch_size=batch_size, learning_rate=learning_rate, hidden_units=hidden_units,
                                        # loss=loss)
        # simple_nn.create_model()
        # simple_nn.run_experiment(train_dataset, test_dataset, fpu.num_epochs)
        fpu.pnn.create_model()
        fpu.pnn.run_experiment(train_dataset, test_dataset, fpu.num_epochs)

        # Perform CVT map elites computation on learned model
        # archive is made of a collection of species
        surrogate_archive, n_evals = cvt_map_elites.compute(dim_map, dim_gen, fpu.fastsim_eval,
                                                            prev_archive=real_archive.copy(),
                                                            n_niches=n_niches, max_evals=max_evals,
                                                            params=params, all_pop_at_once=True,
                                                            iter_number=itr)
        learned_env_evals += n_evals

        #3#
        # Get the N most uncertain individuals and test them on real setup to gather data

        sorted_archive = sorted(surrogate_archive.items(), key=lambda pair: pair[1].fitness, reverse=False)

        print("Archive len: ", len(sorted_archive))
        
        if (sorted_archive[0][1].fitness > -convergence_thresh): # When fitness is negative
        # if (sorted_archive[-1][1].fitness < convergence_thresh): # When fitness is positive
            print("Algorithm has converged")
            break
        
        #4#
        # N = round(1.0*fpu.eval_batch_size) # Number of individuals that we'll try on real_env
        N = len(sorted_archive) # Like in Antoine Cully paper, test all imagined archive
        print(N, " most uncertain archive individuals fitnesses:")
        for ind in sorted_archive[:N]:
            print(ind[1].fitness)

        # reset data containers while maintaining previous data
        tmp_data_in = np.zeros((len(data_in_no_0s)+fpu.horizon*N, fpu.input_dim))
        tmp_data_out = np.zeros((len(data_out_no_0s)+fpu.horizon*N, fpu.output_dim))

        tmp_data_in[:tab_cpt,:] = data_in_no_0s
        tmp_data_out[:tab_cpt,:] = data_out_no_0s
        data_in = tmp_data_in
        data_out = tmp_data_out

        # Test the N most uncertain individuals on real setup to gather new data
        # and update the archive at the same time
        # create the cvt
        c = cm_map_elites.cvt(n_niches, dim_map, params)
        kdt = KDTree(c, leaf_size=30, metric='euclidean')
        for i in range(min(N, len(sorted_archive))):
            ind = sorted_archive[i][1] # (centroid, Species) tuple
            data_in_to_add, data_out_to_add, last_obs = fpu.run_on_gym_env(fpu.real_env,
                                                                           sorted_archive[i][1].x,
                                                                           fpu.horizon,
                                                                           display=False,
                                                                           test_model=False)
        
            # Create a new Species indiv
            first_state = np.concatenate((data_in_to_add[0,2:4], # position
                                          np.array([np.arctan2(data_in_to_add[0,4],
                                                               data_in_to_add[0,5])])))
            last_data = np.add(data_in_to_add[-1,2:], data_out_to_add[-1,:])
            last_state = np.concatenate((last_data[:2], # position
                                          np.array([np.arctan2(last_data[2],
                                                               last_data[3])])))
            
            desc = last_state - first_state
            s = cm_map_elites.Species(sorted_archive[i][1].x, desc, sorted_archive[i][1].fitness)
            # Add to archive
            added = cvt_map_elites.__add_to_archive(s, s.desc, real_archive, kdt)
            # we could change fitness on a another base also?

            # added = 1 if individual was added to archive, 0 otherwise
            # Only add the data if its new
            if(added):
                data_in[tab_cpt:tab_cpt+len(data_in_to_add),:] = data_in_to_add
                data_out[tab_cpt:tab_cpt+len(data_in_to_add),:] = data_out_to_add
            
                tab_cpt += len(data_in_to_add)

            real_env_evals += 1
            
        cm_map_elites.__save_archive(real_archive, max_evals, itr, total_evals=real_env_evals)
        # filter out 0 lines that were left
        data_in_no_0s = data_in[~np.all(data_in == 0, axis=1)] 
        data_out_no_0s = data_out[~np.all(data_in == 0, axis=1)]

        # Reduce iteration number
        itr += 1

    print("Finished learning.")

    # Once skill repertoire is learnt

    # Plan using MCTS to attain final goal

    # Problem -> need to refine model on real system before

    learned_env_params = \
    {
        "hidden_units": fpu.hidden_units,
        "input_dim": fpu.input_dim, # needs to be defined
        "output_dim": fpu.output_dim,  # needs to be defined
        "controller_input_dim": fpu.controller_input_dim,
        "controller_output_dim": fpu.controller_output_dim,
        "controller_nn_params": fpu.controller_nnparams,
        "action_min": [-4, -4],
        "action_max": [4, 4],
        "observation_min": [0, 0, -np.pi],
        "observation_max": [600, 600, np.pi],
        "horizon": fpu.horizon, # time steps
        "init_state": [60., 450.],
        "goal_state": [60., 60.]
    }
    
    # fpu.real_env = 0 # env on which real robot is moving
    # env wrapping the learnt model + archive of behaviours
    learned_env = gym.make('PnnEnv-v0',
                           model_weights=fpu.pnn.model.get_weights(),
                           archive=real_archive,
                           params=learned_env_params)
    
    obs = fpu.real_env.reset()
    model = DPW(alpha=0.3, beta=0.2, initial_obs=obs, env=learned_env, K=3**0.5)
    done = False
    
    N = 10000
    H = 20
    while not done: # until completion of real_env

        ### Reset data containers ###
        # reset data containers while maintaining previous data
        tmp_data_in = np.zeros((len(data_in_no_0s)+fpu.horizon, fpu.input_dim))
        tmp_data_out = np.zeros((len(data_out_no_0s)+fpu.horizon, fpu.output_dim))

        tmp_data_in[:tab_cpt,:] = data_in_no_0s
        tmp_data_out[:tab_cpt,:] = data_out_no_0s
        data_in = tmp_data_in
        data_out = tmp_data_out

        ### Expand the tree and get best action ###
        model.learn(N, progress_bar=True)
        bh_index = model.best_action() # /!\ Outputs index of a behaviour in archive

        ### Run selected action on real_env for horizon timesteps ###
        data_in_to_add, data_out_to_add, last_obs = fpu.run_on_gym_env(fpu.real_env,
                                                                       real_archive[bh_index].x,
                                                                       fpu.horizon,
                                                                       reset_env=False,
                                                                       display=True
        )

        ### Train the model using the foraged data ###
        # Refine our model
        data_in[tab_cpt:tab_cpt+len(data_in_to_add),:] = data_in_to_add
        data_out[tab_cpt:tab_cpt+len(data_in_to_add),:] = data_out_to_add

        # filter out 0 lines that were left
        data_in_no_0s = data_in[~np.all(data_in == 0, axis=1)] 
        data_out_no_0s = data_out[~np.all(data_in == 0, axis=1)]

        # Normalize training data
        normalized_data_in, normalized_data_out = fpu.normalize_data(data_in_no_0s, data_out_no_0s)
        
        # Learn a model from the gathered data
        train_dataset, test_dataset = fpu.pnn.get_train_and_test_splits(normalized_data_in, normalized_data_out)

        fpu.pnn.create_model()
        fpu.pnn.run_experiment(train_dataset, test_dataset, fpu.num_epochs)

        tab_cpt += len(data_in_to_add)

        real_env_evals += 1

        ### Advance the tree to the action taken ###
        model.forward(bh_index, last_obs)
        
        if done:
            break
