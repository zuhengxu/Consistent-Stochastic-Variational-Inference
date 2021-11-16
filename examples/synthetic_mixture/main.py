import autograd.numpy as np
import argparse
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from examples.common.results import *
from examples.common.elbo import *
from examples.common.synthetic_model import *
from VI.csvi import *
from VI.svi import *
from VI.MAP import *


###########################
###########################
# Smooth MAP estimaton
###########################
###########################

def get_init(arguments):
    if not os.path.exists("results/"):
        os.mkdir('results/')
    # check if intials already exists
    if check_exists(arguments.init_title + arguments.alg, arguments.init_folder):
        print('Initialization already exists for' + arguments.alg)
        print('Quitting')
        quit()

    ###################
    # Step 0: Setup
    ###################
    np.random.seed(int(int(arguments.alg, base=32)/1000) + arguments.alpha)

    def lpdf(x): return log_gs_mix(x)  # mixture lpdf
    alpha = arguments.alpha

    map_lrt = eval(arguments.map_stepsched)
    num_iters = 20000
    mcS = 100  # mc sample size for smooth gradient
    # define map/smap functions with same arguments
    def SMAP(mu0): return smooth_1d_MAP( mu0, lpdf, map_lrt, alpha, mcS, num_iters)

    #######################################
    # Step 1: Gets initialization
    #######################################

    # 100 repititions for each setting
    mean_list = []
    sd_list = []
    for i in range(100):
        # random
        mu0 = np.array([(np.random.random() - 0.5)*100])
        logsd = np.array([np.random.random()*np.log(10) + np.log(0.5)])
        sd0 = np.exp(logsd)
        # intialization strategies for different settings
        init_dict = {'SVI': [mu0, sd0],
                     'CSVI': [SMAP(mu0), np.array([1.])],
                     'CSVI_RSD': [SMAP(mu0), sd0],
                     'SVI_Ind': [mu0, np.array([1.])],
                     'SVI_SMAP': [SMAP(mu0), sd0],
                     'SVI_OPT': [SMAP(mu0), np.array([1.])]}

        mean_init, sd_init = init_dict[arguments.alg]
        # append to the list
        mean_list.append(mean_init[0])
        sd_list.append(sd_init[0])
        print('intialism = ', mean_init, sd_init)

    #######################################
    # Step 2: save results
    #######################################

    results_dict = {'method': arguments.alg,
                    'mean_initial': mean_list,
                    'sd_initial': sd_list,
                    'alpha': arguments.alpha
                    }

    save(results_dict, arguments.init_title +
         arguments.alg, arguments.init_folder)


#######################################################
#######################################################
# run GVB inference: alg(step_sched, init_folder, init_title) 100 rep
#######################################################
#######################################################

def run_vi(arguments):
    if not os.path.exists("results/"):
        os.mkdir('results/')
    # check if results already exists
    if check_exists(arguments.vi_title + arguments.alg, arguments.vi_folder):
        print('VI results already exists for' + arguments.alg)
        print('Quitting')
        quit()

    #######################
    # Step 0: Setup
    #######################
    np.random.seed(int(int(arguments.alg, base=32)/1000))

    # mixture lpdf
    def lpdf(x): return log_gs_mix(x)

    # alg type
    alg_class = {
        'CSVI': 'CSVI',
        'CSVI_RSD': 'CSVI',
        'CSVI_MAP': 'CSVI',
        'SVI': 'SVI',
        'SVI_Ind': 'SVI',
        'SVI_SMAP': 'SVI',
        'SVI_OPT': 'SVI', 
        'CSL': 'LAPLACE'
    }

    # choose the alg to run
    alg_dict = {'SVI': svi, 'CSVI': csvi, 'LAPLACE': cs_laplace_1d}
    vi_alg = alg_dict[alg_class[arguments.alg]]

    # set step schedule for vi optimization
    if arguments.alg == 'CSL':
        def vi_lrt(i): return 0.001
    else:
        def vi_lrt(i): return arguments.vi_stepsched/(1 + i)

    #######################################
    # Step 1: Read initialization results
    #######################################
    if arguments.alg == 'CSL':
        df_init = pd.read_csv(os.path.join(
            arguments.init_folder, arguments.init_title + 'CSVI.csv'))
    else:
        df_init = pd.read_csv(os.path.join(
                arguments.init_folder, arguments.init_title + arguments.alg + '.csv'))

    ##############################
    # Step 2: run vi alg
    ##############################
    print('Running', arguments.alg)

    mean_list = []
    sd_list = []
    elbo_list = []

    for i in range(df_init.shape[0]):
        mean_init, sd_init = np.array([df_init.loc[i, 'mean_initial']]), np.array([df_init.loc[i, 'sd_initial']])
        init_val, _ = flatten([mean_init, sd_init])

        # get vi results (mean, sd)
        if arguments.alg == 'CSL':
            x = vi_alg(init_val, lpdf, vi_lrt, 5000)
        else:
            x = vi_alg(init_val, 1, lpdf, vi_lrt, 100000)
        
        mean_vi, sd_vi = np.array([x[0]]), np.array([x[1]])
        # compute elbo
        elbo = GVB_ELBO(lpdf, mean_vi, sd_vi, 1000)

        # append to list
        mean_list.append(mean_vi[0])
        sd_list.append(sd_vi[0])
        elbo_list.append(elbo[0])

    #############################
    # step 3: save results
    #############################
    df_results = pd.DataFrame({
        'mean_vi': mean_list,
        'sd_vi': sd_list,
        'elbo': elbo_list,
        'vi_stepsched': arguments.vi_stepsched
    })
    df = df_init.join(df_results)

    save(df, arguments.vi_title + arguments.alg, arguments.vi_folder)


###########################
###########################
# Parse arguments
###########################
###########################
parser = argparse.ArgumentParser(description=" Runs synthetic mixture example")
subparsers = parser.add_subparsers(help='sub-command help')
get_init_subparser = subparsers.add_parser(
    'get_init', help='Get initialization for GVB')
get_init_subparser.set_defaults(func=get_init)
run_vi_subparser = subparsers.add_parser('run_vi', help='Run GVB inference')
run_vi_subparser.set_defaults(func=run_vi)

parser.add_argument('--alg', type=str, default="CSVI",
                    help="The name of the GVB algorithm to use")
parser.add_argument('--trial', type=int, default=5,
                    help="Break 100 repitition into several trials")
parser.add_argument('--alpha', type=int, default=100,
                    help='Smoothing variance')
parser.add_argument('--init_folder', type=str, default='results/initials/',
                    help='Folder of saving initialization results')
parser.add_argument('--init_title', type=str, default='initial_',
                    help='Prefix of the intialization file')
parser.add_argument('--vi_folder', type=str,
                    default='results/VI_results/', help='Folder of saving vi results')
parser.add_argument('--vi_title', type=str, default='results_',
                    help='Prefix of the vi result file')

get_init_subparser.add_argument(
    '--map_stepsched', type=str, default='lambda itr: 200./(1+ itr)', help='MAP initialization step schedule')

run_vi_subparser.add_argument(
    '--vi_stepsched', type=int, default=10, help='VI optimization step schedule')

arguments = parser.parse_args()
arguments.func(arguments)
