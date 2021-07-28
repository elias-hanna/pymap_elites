# CVT map elite imports
# from map_elites import compute
import map_elites.cvt as cvt_map_elites
import map_elites.common as cm_map_elites

# Utils import
import fastsim_pnn_utils as fpu
import numpy as np


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
        "dump_period": 2,
        "parallel": True,
        "cvt_use_cache": True,
        "min": -5,
        "max": 5,
        "bd_min": [0, 0],
        "bd_max": [600, 600]
    }

    dim_map = 2
    dim_gen = fpu.n_weights
    n_niches = 1000
    n_gen = 20
    max_evals = n_gen*fpu.eval_batch_size
    
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
    

    #0#
    
    # Do init_random_trajs trajectories on real_env with random genotype and save the traj data for model learning
    # input: genotype, horizon
    # output: trajs
    # Note: this type of init works for envs that are run with a unique action at beginning (not action trajs)
    tab_cpt = 0
    for i in range(fpu.init_random_trajs):
        # Create a random genotype
        genotype = np.random.uniform(low=params["min"], high=params["max"], size=(fpu.n_weights,))

        data_in_to_add, data_out_to_add = fpu.run_on_gym_env(fpu.real_env, genotype, fpu.horizon)

        data_in[tab_cpt:tab_cpt+len(data_in_to_add),:] = data_in_to_add
        data_out[tab_cpt:tab_cpt+len(data_out_to_add),:] = data_out_to_add
               
        print("{:.1f}".format(i/fpu.init_random_trajs*100),"% done", end="\r")
        tab_cpt += len(data_in_to_add)

    # filter out 0 lines that were left
    data_in_no_0s = data_in[~np.all(data_in == 0, axis=1)] 
    data_out_no_0s = data_out[~np.all(data_in == 0, axis=1)]

    max_iter = 10
    itr = 0
    convergence_thresh = 0.1
    has_converged = False # If uncertainty of less certain trajectory is below threshold ?
    archive = {}
    
    while (itr < max_iter or not has_converged):

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
        # archive = compute(dim_map, dim_gen, fastsim_eval, n_niches=n_niches, n_gen=n_gen, params=params)
        # archive is made of a collection of species
        # archive = compute(dim_map, dim_gen, fastsim_test_eval, n_niches=n_niches, n_gen=n_gen, params=params)
        archive = cvt_map_elites.compute(dim_map, dim_gen, fpu.fastsim_eval, prev_archive=archive,
                                         n_niches=n_niches, max_evals=max_evals, params=params,
                                         all_pop_at_once=True, iter_number=itr)
        # archive = compute_step(dim_map, dim_gen, fpu.real_env_eval, n_niches=n_niches, n_gen=n_gen, params=params) # just to test cvt archive

        #3#
        # Get the N most uncertain individuals and test them on real setup to gather data
        N = round(0.5*fpu.eval_batch_size) # Number of individuals that we'll try on real_env

        sorted_archive = sorted(archive.items(), key=lambda pair: pair[1].fitness, reverse=False)

        print("Archive len: ", len(sorted_archive))
        print(N, " most uncertain archive individuals fitnesses:")
        for i in sorted_archive:
            print(i[1].fitness)

        if (sorted_archive[0][1].fitness > -convergence_thresh):
            print("Algorithm has converged")
            break
        
        #4#

        # reset data containers while maintaining previous data
        tmp_data_in = np.zeros((len(data_in_no_0s)+fpu.horizon*N, fpu.input_dim))
        tmp_data_out = np.zeros((len(data_out_no_0s)+fpu.horizon*N, fpu.output_dim))

        tmp_data_in[:tab_cpt,:] = data_in_no_0s
        tmp_data_out[:tab_cpt,:] = data_out_no_0s
        data_in = tmp_data_in
        data_out = tmp_data_out

        # Test the N most uncertain individuals on real setup to gather new data
        for i in range(min(N, len(sorted_archive))):
            data_in_to_add, data_out_to_add = fpu.run_on_gym_env(fpu.real_env,
                                                             sorted_archive[i][1].x, fpu.horizon)
        
            data_in[tab_cpt:tab_cpt+len(data_in_to_add),:] = data_in_to_add
            data_out[tab_cpt:tab_cpt+len(data_in_to_add),:] = data_out_to_add

            tab_cpt += len(data_in_to_add)

        # filter out 0 lines that were left
        data_in_no_0s = data_in[~np.all(data_in == 0, axis=1)] 
        data_out_no_0s = data_out[~np.all(data_in == 0, axis=1)]

        # Reduce iteration number
        itr += 1

    print("Finished learning.")
