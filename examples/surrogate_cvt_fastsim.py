# CVT map elite imports
# from map_elites import compute
import map_elites.cvt as cvt_map_elites
import map_elites.common as cm_map_elites

# from scipy.spatial import cKDTree : TODO -- faster?
from sklearn.neighbors import KDTree

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
        "dump_period": 100000,
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
    surrogate_archive = {}
    real_archive = {}
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
        # archive is made of a collection of species
        surrogate_archive = cvt_map_elites.compute(dim_map, dim_gen, fpu.fastsim_eval, prev_archive=real_archive,
                                         n_niches=n_niches, max_evals=max_evals, params=params,
                                         all_pop_at_once=True, iter_number=itr)
        # surrogate_archive = cvt_map_elites.compute(dim_map, dim_gen, fpu.real_env_eval, prev_archive=real_archive,
        #                                  n_niches=n_niches, max_evals=max_evlas, params=params,
        #                                  all_pop_at_once=True, iter_number=itr) # to test cvt alg

        #3#
        # Get the N most uncertain individuals and test them on real setup to gather data

        sorted_archive = sorted(surrogate_archive.items(), key=lambda pair: pair[1].fitness, reverse=False)

        print("Archive len: ", len(sorted_archive))
        
        if (sorted_archive[0][1].fitness > -convergence_thresh):
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
            data_in_to_add, data_out_to_add = fpu.run_on_gym_env(fpu.real_env,
                                                             sorted_archive[i][1].x, fpu.horizon)
        
            data_in[tab_cpt:tab_cpt+len(data_in_to_add),:] = data_in_to_add
            data_out[tab_cpt:tab_cpt+len(data_in_to_add),:] = data_out_to_add
            
            s = sorted_archive[i][1] # Species type
            s.centroid = None
            s.desc = np.add(fpu.rescale_standard(data_in_to_add[-1,2:4], fpu.means_in, fpu.stds_in), fpu.rescale_standard(data_out_to_add[-1,:2], fpu.means_out, fpu.stds_out)) # replace imagined desc by real desc
            cvt_map_elites.__add_to_archive(s, s.desc, real_archive, kdt)
            # we could change fitness on a another base also?
            
            tab_cpt += len(data_in_to_add)

        cm_map_elites.__save_archive(real_archive, max_evals, itr)
        # filter out 0 lines that were left
        data_in_no_0s = data_in[~np.all(data_in == 0, axis=1)] 
        data_out_no_0s = data_out[~np.all(data_in == 0, axis=1)]

        # Reduce iteration number
        itr += 1

    print("Finished learning.")
