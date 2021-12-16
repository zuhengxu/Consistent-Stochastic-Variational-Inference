import autograd.numpy as np
import argparse
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from VI.MAP import *
from VI.svi import *
from VI.csvi import *
from VI.laplace import *
from examples.common.sparse_reg_model import *
from examples.common.elbo import *
from examples.common.results import *




###########################
###########################
### VI initialization
###########################
###########################

def get_init(arguments):
    if not os.path.exists("results/"):
        os.mkdir('results/')
    #check if intials already exists
    if check_exists(arguments.dataset+ "_" + arguments.mu_scheme + arguments.L_scheme, arguments.init_folder):
        print('Initialization already exists for' + arguments.dataset+ "_" + arguments.mu_scheme + arguments.L_scheme)
        print('Quitting')
        quit()

    ###################
    ## Step 0: Setup
    ###################
    np.random.seed(int(int(arguments.mu_scheme + arguments.L_scheme, base = 32)/1e15 + arguments.trial))

    posterior_dict = {'SYN': syn_lpdf,
                    'REAL': real_lpdf }
    dim_dict = {
        'SYN': '5,0.1, 10',
        'REAL': '8, 0.01, 5'
    }
    D,tau1, tau2 = eval(dim_dict[arguments.dataset])
    lpdf = posterior_dict[arguments.dataset]
    alpha = arguments.alpha
    map_lrt = eval(arguments.map_stepsched)
    num_iters = 40000
    mcS = 100 # mc sample size for smooth gradient
    # define map/smap functions with same arguments
    SMAP = lambda x0: smooth_MAP(x0, lpdf, map_lrt, alpha, mcS, num_iters)
    SMAP_AD = lambda x0: SMAP_adam(x0, lpdf, map_lrt, alpha, mcS, num_iters)


    #######################################
    ## Step 1: Gets initialization
    #######################################
    init_list = []
    # 100 repititions for each setting
    for i in range(20):
        # intialization for mean
        mu_random = prior_sample(1, D, tau1, tau2)[0]
        mu_init_dict = {'Prior': 'mu_random',
                        'SMAP': 'SMAP(mu_random)',
                        'SMAP_adam': 'SMAP_AD(mu_random)'}
        L_init_dict= {'Random': np.diag(np.exp(np.random.random(D)*np.log(100)+ np.log(0.1)) ),
                    'Ind': np.identity(D)}
        mu0, L0 = eval(mu_init_dict[arguments.mu_scheme]), L_init_dict[arguments.L_scheme]
        init, _ = flatten((mu0, L0))
        # append to the list
        init_list.append(init)
        print('intialism = ', init)

    #######################################
    ## Step 2: save results
    #######################################
    init_matrix =  np.array(init_list)
    save(init_matrix, arguments.dataset+ "_" + arguments.mu_scheme + arguments.L_scheme, arguments.init_folder)




#######################################################
#######################################################
### run GVB inference 100 rep
#######################################################
#######################################################

def run_vi(arguments):
    if not os.path.exists("results/"):
        os.mkdir('results/')
    # check if results already exists
    if check_exists(arguments.dataset + '_' + arguments.vi_alg + '_'+ arguments.mu_scheme + arguments.L_scheme, arguments.vi_folder):
        print('VI results already exists for' + arguments.dataset + '_' + arguments.vi_alg + '_'+ arguments.mu_scheme + arguments.L_scheme)
        print('Quitting')
        quit()

    #######################
    ## Step 0: Setup
    #######################
    np.random.seed(int(int(arguments.vi_alg, base = 32)/1e5 + arguments.trial))

    #mixture lpdf
    lpdf_dict = {'SYN': syn_lpdf,
                'REAL': real_lpdf}
    lpdf = lpdf_dict[arguments.dataset]
    # gets sample size N
    Data_size = {
        'SYN': 10,
        'REAL': 30
    }
    N = Data_size[arguments.dataset]
    # choose the alg to run
    alg_dict = {'SVI': svi,
                'SVI_adam': svi_adam,
                'CSVI': csvi,
                'CSVI_adam': csvi_adam, 
                'CSL': cs_laplace}

    vi_alg = alg_dict[arguments.vi_alg]
    # set step schedule for vi optimization
    vi_lrt = eval(arguments.vi_stepsched)

    #######################################
    ## Step 1: Read initialization results
    #######################################
    df_init = pd.read_csv(os.path.join(arguments.init_folder, arguments.dataset+ "_" + arguments.mu_scheme + arguments.L_scheme + '.csv' ))

    ##############################
    ## Step 2: run vi alg
    ##############################
    print('Running', arguments.vi_alg)

    VI_result_list = []
    elbo_list = []
    #break 100 interation into 2 trials
    size = int(df_init.shape[0]/5)
    for k in range(size):
        i = size*(arguments.trial- 1) + k
        init_val = np.array(df_init.iloc[i])
        # get vi results (mean, sd)
        if arguments.vi_alg == 'CSL':
            if arguments.mu_scheme == 'SMAP_adam':
                num_iter = 10000
            else: 
                num_iter = 40000
            x = vi_alg(init_val, lpdf, vi_lrt, num_iter)
        else:
            x = vi_alg(init_val, N , lpdf, vi_lrt, 200000)
        # compute elbo
        elbo = multi_ELBO(lpdf, x)

        # append to list
        VI_result_list.append(x)
        elbo_list.append(elbo)

    #############################
    ## step 3: save results
    #############################
    results_matrix = np.hstack((np.array(elbo_list)[:,None], np.array(VI_result_list)))
    save(results_matrix, arguments.dataset + '_' + arguments.vi_alg + '_'+ arguments.mu_scheme + arguments.L_scheme, arguments.vi_folder)




#######################################################
#######################################################
### run RSVI 100 rep
#######################################################
#######################################################

def run_rsvi(arguments):
    if not os.path.exists("results/"):
        os.mkdir('results/')
    # check if results already exists
    # if check_exists(arguments.dataset + '_' + arguments.vi_alg + '_'+ arguments.mu_scheme + arguments.L_scheme, arguments.vi_folder):
    #     print('VI results already exists for' + arguments.dataset + '_' + arguments.vi_alg + '_'+ arguments.mu_scheme + arguments.L_scheme)
    #     print('Quitting')
    #     quit()

    #######################
    ## Step 0: Setup
    #######################
    np.random.seed(int(int(arguments.vi_alg, base = 32)/1e5 + arguments.trial))

    #mixture lpdf
    lpdf_dict = {'SYN': syn_lpdf,
                'REAL': real_lpdf}
    lpdf = lpdf_dict[arguments.dataset]
    # gets sample size N
    Data_size = {
        'SYN': 10,
        'REAL': 30
    }
    N = Data_size[arguments.dataset]
    # choose the alg to run
    vi_alg = rsvi_adam
    # set step schedule for vi optimization
    vi_lrt = eval(arguments.vi_stepsched)

    #######################################
    ## Step 1: Read initialization results
    #######################################
    df_init = pd.read_csv(os.path.join(arguments.init_folder, arguments.dataset+ "_" + arguments.mu_scheme + arguments.L_scheme + '.csv' ))

    ##############################
    ## Step 2: run vi alg
    ##############################
    print('Running', arguments.vi_alg)

    VI_result_list = []
    elbo_list = []
    lmd_list = []
    #break 100 interation into 2 trials
    size = int(df_init.shape[0]/5)
    for k in range(size):
        i = size*(arguments.trial- 1) + k
        init_val = np.array(df_init.iloc[i])
        # get vi results (mean, sd)
        x = vi_alg(init_val, N , lpdf, vi_lrt, 200000, arguments.regularizer)
        # compute elbo
        elbo = multi_ELBO(lpdf, x)

        # append to list
        VI_result_list.append(x)
        elbo_list.append(elbo)
        lmd_list.append(arguments.regularizer)
    #############################
    ## step 3: save results
    #############################
    results_matrix = np.hstack((np.array(lmd_list)[:,None], np.array(elbo_list)[:,None], np.array(VI_result_list)))
    save(results_matrix, arguments.dataset + '_' + arguments.vi_alg + '_'+ arguments.mu_scheme + arguments.L_scheme, arguments.vi_folder)




###########################
###########################
### Parse arguments
###########################
###########################

parser = argparse.ArgumentParser(description = " Runs synthetic mixture example")
subparsers = parser.add_subparsers(help='sub-command help')
get_init_subparser = subparsers.add_parser('get_init', help = 'Get initialization for GVB')
get_init_subparser.set_defaults(func = get_init)
run_vi_subparser = subparsers.add_parser('run_vi', help = 'Run GVB inference')
run_vi_subparser.set_defaults(func = run_vi)
run_rsvi_subparser = subparsers.add_parser('run_rsvi', help='Run RSVI inference')
run_rsvi_subparser.set_defaults(func=run_rsvi)

parser.add_argument('--dataset', type  = str, default= "SYN", help = "The choice of dataset")
parser.add_argument('--trial', type = int, default = 1 ,help = "Index of the trial")
parser.add_argument('--alpha', type = int, default= 2 , help = 'Smoothing variance')
parser.add_argument('--init_folder', type = str, default = 'results/initials/', help = 'Folder of saving initialization results')
# parser.add_argument('--init_title', type = str, default = 'syn_', help = 'Prefix of the intialization file')
parser.add_argument('--vi_folder', type = str, default = 'results/VI_results/', help = 'Folder of saving vi results')
# parser.add_argument('--vi_title', type = str, default = 'syn_', help = 'Prefix of the vi result file')
parser.add_argument('--mu_scheme', type = str, default= 'Prior', help = 'GVI mean initialization')
parser.add_argument('--L_scheme', type = str, default= 'Ind', help = 'GVI covariance intialization')

get_init_subparser.add_argument('--map_stepsched', type = str, default= 'lambda itr: 0.01', help = 'MAP initialization step schedule')

run_vi_subparser.add_argument('--vi_stepsched', type= str, default  = 'lambda itr: 0.001', help = 'VI optimization step schedule')
run_vi_subparser.add_argument('--vi_alg', type= str, default  = 'CSVI_adam', help = 'VI optimization algorithm')

run_rsvi_subparser.add_argument('--vi_stepsched', type= str, default  = 'lambda itr: 0.01', help = 'VI optimization step schedule')
run_rsvi_subparser.add_argument('--regularizer', type= float, default= 0.0, help='RSVI regularization constant')
run_rsvi_subparser.add_argument('--vi_alg', type= str, default  = 'CSVI_adam', help = 'VI optimization algorithm')

arguments = parser.parse_args()
arguments.func(arguments)




