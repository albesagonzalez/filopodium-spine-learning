import itertools
import multiprocessing

from collections import defaultdict, OrderedDict
import pickle
from brian2 import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from aux import c_timed_array, get_zero_current, get_vm_corr, get_dynamical_terms
from run_network_functions import run_FS_network

from scipy.stats import pearsonr

from aux import make_data_dir
make_data_dir()

def save_results(results_list, filename):
    results = {}
    results["RF"], results["corr"], results["DI"] = [], [], []

    for RF, corr, DI in results_list:
        results["RF"].append(RF)
        results["corr"].append(corr)
        results["DI"].append(DI)

    with open('Data/{}'.format(filename), 'wb') as handle:
        pickle.dump(dict(results), handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_RF_DI(neuron_params, plasticity_params, simulation_params, w, c_0):
    
    def DI_from_spike_train(recording_T, y_0, spike_times):
      frs = np.zeros(int(recording_T/second))
      for spike_time in spike_times.t:
          t_index = int(spike_time//second)
          frs[t_index] += 1
      frs = frs/second
      DI = np.zeros(simulation_params["N_pre"])
      for theta_index, y1 in enumerate(frs):
         DI[theta_index] = (y_0 - y1)/(y_0 + y1)
      return frs, np.mean(DI), y_0
    

    w = w/np.max(w)

    plasticity_params["lmbda"] = 0.
    simulation_params["w"] = w

    patterns = []
    pattern_duration = 100*second
    current_time = 0*second
    pattern = {}
    pattern["start_time"] = current_time
    pattern["duration"] = pattern_duration
    pattern["c"] = c_0
    patterns.append(pattern)
    current_time += pattern["duration"]
    simulation_params["total_time"] = current_time
    simulation_params["c"] = c_timed_array(patterns, simulation_params)
    simulation_params["I_ext"] = get_zero_current(simulation_params, 0)
    spike_ref_mon, spike_pre_mon, spike_post_mon_0, w_trajs, post_mon, mu_trajs = run_FS_network(neuron_params, plasticity_params, simulation_params)

    y_0 = spike_post_mon_0.count[0]/pattern_duration

    patterns = []
    pattern_duration = 1*second  
    current_time = 0*second
    for theta_index in range(simulation_params["N_pre"]):
      pattern = {}
      pattern["start_time"] = current_time
      pattern["duration"] = pattern_duration
      pattern["c"] = np.roll(c_0, theta_index)
      patterns.append(pattern)
      current_time += pattern["duration"]

    simulation_params["total_time"] = current_time
    simulation_params["c"] = c_timed_array(patterns, simulation_params)
    simulation_params["I_ext"] = get_zero_current(simulation_params, 0)

    spike_ref_mon, spike_pre_mon, spike_post_mon, w_trajs, post_mon, mu_trajs = run_FS_network(neuron_params, plasticity_params, simulation_params)

    return DI_from_spike_train(current_time, y_0, spike_post_mon)
   


def get_RF_stats(params):

    neuron_params, plasticity_params, simulation_params, bias, alpha, c_tot, num_seeds = params

    for seed in range(num_seeds):
        simulation_params["seed"] = seed

        plasticity_params["alpha"] = alpha
        kappa = 8

        patterns = []
        current_time = 0*second
        max_time = 200*second
        pattern = {}
        pattern["start_time"] = current_time
        pattern["duration"] = max_time
        pattern["c"] = get_vm_corr(0, kappa, c_tot, bias=bias)
        patterns.append(pattern)
        current_time += pattern["duration"]

        simulation_params["total_time"] = current_time
        simulation_params["c"] = c_timed_array(patterns, simulation_params)
        simulation_params["I_ext"] = get_zero_current(simulation_params, 0)

        spike_ref_mon, spike_pre_mon, spike_post_mon, w_trajs, post_mon, mu_trajs = run_FS_network(neuron_params, plasticity_params, simulation_params)
        del simulation_params["c"], simulation_params["I_ext"]

        c = patterns[0]["c"]
        w = np.mean(w_trajs[:, -10:], axis=1)

        filo_index = np.where(w < plasticity_params["w0_minus"])[0]
        spine_index = np.where(w >= plasticity_params["w0_minus"])[0]

        w_filo = w[filo_index]
        w_spine = w[spine_index]

        filo_num, filo_mu, filo_sgm = len(w_filo), np.mean(w_filo), np.std(w_filo)
        spine_num, spine_mu, spine_sgm = len(w_spine), np.mean(w_spine), np.std(w_spine)

        try:
            corr = pearsonr(c[spine_index], w_spine).statistic
        except:
            corr = 0

        r_DI = 15*Hz
        simulation_params["r_pre"] = r_DI
        frs, DI, y_0 = get_RF_DI(neuron_params, plasticity_params, simulation_params, w, c)

        try:
            corr_av += corr
            DI_av += DI

        except:
            corr_av = corr
            DI_av = DI

    return  w, corr_av/num_seeds, DI_av/num_seeds

def get_square_RFs(params):

    neuron_params, plasticity_params, simulation_params, bias, alpha, c_tot, num_seeds = params


    for seed in range(1):
        simulation_params["seed"] = seed

        plasticity_params["alpha"] = alpha

        patterns = []
        current_time = 0*second
        max_time = 200*second
        pattern = {}
        pattern["start_time"] = current_time
        pattern["duration"] = max_time
        pattern["c"] = np.zeros((1000))
        pattern["c"][np.arange(400, 600)] = c_tot/200
        patterns.append(pattern)
        current_time += pattern["duration"]

        simulation_params["total_time"] = current_time
        simulation_params["c"] = c_timed_array(patterns, simulation_params)
        simulation_params["I_ext"] = get_zero_current(simulation_params, 0)

        spike_ref_mon, spike_pre_mon, spike_post_mon, w_trajs, post_mon, mu_trajs = run_FS_network(neuron_params, plasticity_params, simulation_params)
        del simulation_params["c"], simulation_params["I_ext"]

        w = np.mean(w_trajs[:, -10:], axis=1)

    return  w, 0, 0



if __name__ == '__main__':

    #define neuron physiological parameters
    neuron_params = {}
    neuron_params["C_m"] = 200*pfarad
    neuron_params["R_m"] = 100*Mohm
    neuron_params["tau_m"] = neuron_params["C_m"]*neuron_params["R_m"]
    neuron_params["v_thres"] = -54*mV
    neuron_params["v_rest"] = -70*mV
    neuron_params["E_exc"] = 0*mV
    neuron_params["E_inh"] = -70*mV
    neuron_params["g_exc_hat"] = 0.15*nsiemens
    neuron_params["g_inh_hat"] = 0.25*nsiemens
    neuron_params["tau_exc"] = 5*ms
    neuron_params["tau_inh"] = 5*ms

    #define learning parameters
    plasticity_params = {}
    plasticity_params["add"] = 0
    plasticity_params["mlt"] = 0
    plasticity_params["nlta"] = 0
    plasticity_params["FS"] = 0
    plasticity_params["a"] = 0
    plasticity_params["q"] = 8
    plasticity_params["mu_plus"] = 0
    plasticity_params["mu_minus"] = 0
    plasticity_params["tau_mu"] = 20*second
    plasticity_params["mu_3"] = 1
    plasticity_params["tau_plus"] = 20*ms
    plasticity_params["tau_minus"] = 20*ms
    plasticity_params["w0_plus"] = 1
    plasticity_params["w0_minus"] = 0.5
    plasticity_params["lmbda"] = 0.006
    plasticity_params["alpha"] = 1.35

    #define network architecture and simulation specs
    simulation_params = {}
    simulation_params["total_time"] = 150*second
    simulation_params["integration_dt"] = 0.5*ms
    simulation_params["input_dt"] = 1*second
    simulation_params["w_recording_dt"] = 1*second
    simulation_params["N_pre"] = 1000
    simulation_params["r_pre"] = 30*Hz
    simulation_params["N_post"] = 1
    simulation_params["class_pools"] = False
    simulation_params["w"] = 0.3
    simulation_params["seed"] = 0


    #########################################################
    #simulate RF formation with von Mises shaped correlations
    #########################################################
    
    c_tot = 60
    kappa = 8

    current_time = 0*second
    A_duration = 200*second

    patterns = []
    pattern = {}
    pattern["start_time"] = current_time
    pattern["duration"] = A_duration
    pattern["c"] = get_vm_corr(0, kappa, c_tot)
    c = pattern["c"]
    patterns.append(pattern)
    current_time += pattern["duration"]

    simulation_params["total_time"] = current_time
    simulation_params["c"] = c_timed_array(patterns, simulation_params)
    simulation_params["I_ext"] = get_zero_current(simulation_params, 0)

    '''
    plasticity_params["add"] = 0
    plasticity_params["mlt"] = 0
    plasticity_params["nlta"] = 0
    plasticity_params["FS"] = 1
    spike_ref_mon, spike_pre_mon, spike_post_mon, w_trajs_FS, mu_trajs_FS, post_mon = run_FS_network(neuron_params, plasticity_params, simulation_params)
    fp_FS, fm_FS, factor_FS, competition_FS, cooperation_FS = get_dynamical_terms(w_trajs_FS, mu_trajs_FS, patterns, neuron_params, plasticity_params, simulation_params)
    w_FS = np.mean(w_trajs_FS[:, -10:], axis=1)
    filo_index_FS = np.where(w_FS < plasticity_params["w0_minus"])[0]
    spine_index_FS = np.where(w_FS >= plasticity_params["w0_minus"])[0]


    #########################################################
    # correlation vs final weight curve
    #########################################################

    results = {}
    results["c"] = c
    results["w_FS"] = w_FS
    results["filo_index"] = filo_index_FS
    results["spine_index"] = spine_index_FS

    with open("Data/figure_2B.pickle", 'wb') as handle:
        pickle.dump(dict(results), handle, protocol=pickle.HIGHEST_PROTOCOL)


    #########################################################
    # discrimination index
    #########################################################

    simulation_params["r_pre"] = 15*Hz
    FR_FS, DI_FS, y_0 = get_RF_DI(neuron_params, plasticity_params, simulation_params, w_FS, c)

    results = {}
    results["FR_FS"] = FR_FS
    results["y_0"] = y_0

    with open("Data/figure_2C.pickle", 'wb') as handle:
        pickle.dump(dict(results), handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    simulation_params["total_time"] =  A_duration
    simulation_params["r_pre"] = 30*Hz
    simulation_params["w"] = 0.3
    simulation_params["c"] = c_timed_array(patterns, simulation_params)
    simulation_params["I_ext"] = get_zero_current(simulation_params, 0)
    plasticity_params["lmbda"] = 0.006


    ###########################################
    # repeat simulations for add- and mlt-STDP
    ###########################################


    plasticity_params["add"] = 1
    plasticity_params["mlt"] = 0
    plasticity_params["nlta"] = 0
    plasticity_params["FS"] = 0
    spike_ref_mon, spike_pre_mon, spike_post_mon, w_trajs_add, mu_trajs_add, post_mon, = run_FS_network(neuron_params, plasticity_params, simulation_params)
    w_add = np.mean(w_trajs_add[:, -10:], axis=1)
    filo_index_add = np.where(w_add < plasticity_params["w0_minus"])[0]
    spine_index_add = np.where(w_add >= plasticity_params["w0_minus"])[0]
    fp_add, fm_add, factor_add, competition_add, cooperation_add = get_dynamical_terms(w_trajs_add, mu_trajs_add, patterns, neuron_params, plasticity_params, simulation_params)


    plasticity_params["mu_plus"] = 0.025
    plasticity_params["mu_minus"] = 0.025
    plasticity_params["add"] = 0
    plasticity_params["mlt"] = 0
    plasticity_params["nlta"] = 1
    plasticity_params["FS"] = 0
    plasticity_params["w0_minus"] = 0.
    spike_ref_mon, spike_pre_mon, spike_post_mon, w_trajs_nlta_25, mu_trajs_nlta_25, post_mon = run_FS_network(neuron_params, plasticity_params, simulation_params)
    w_nlta_25 = np.mean(w_trajs_nlta_25[:, -10:], axis=1)
    filo_index_nlta_25 = np.where(w_nlta_25 < plasticity_params["w0_minus"])[0]
    spine_index_nlta_25 = np.where(w_nlta_25 >= plasticity_params["w0_minus"])[0]

    plasticity_params["mu_plus"] = 0.05
    plasticity_params["mu_minus"] = 0.05
    plasticity_params["add"] = 0
    plasticity_params["mlt"] = 0
    plasticity_params["nlta"] = 1
    plasticity_params["FS"] = 0
    spike_ref_mon, spike_pre_mon, spike_post_mon, w_trajs_nlta_50, mu_trajs_nlta_50, post_mon = run_FS_network(neuron_params, plasticity_params, simulation_params)
    w_nlta_50 = np.mean(w_trajs_nlta_50[:, -10:], axis=1)
    filo_index_nlta_50 = np.where(w_nlta_50 < plasticity_params["w0_minus"])[0]
    spine_index_nlta_50 = np.where(w_nlta_50 >= plasticity_params["w0_minus"])[0]

    plasticity_params["mu_plus"] = 0.075
    plasticity_params["mu_minus"] = 0.075
    plasticity_params["add"] = 0
    plasticity_params["mlt"] = 0
    plasticity_params["nlta"] = 1
    plasticity_params["FS"] = 0
    spike_ref_mon, spike_pre_mon, spike_post_mon, w_trajs_nlta_75, mu_trajs_nlta_75, post_mon = run_FS_network(neuron_params, plasticity_params, simulation_params)
    w_nlta_75 = np.mean(w_trajs_nlta_75[:, -10:], axis=1)
    filo_index_nlta_75 = np.where(w_nlta_75 < plasticity_params["w0_minus"])[0]
    spine_index_nlta_75 = np.where(w_nlta_75 >= plasticity_params["w0_minus"])[0]

    plasticity_params["mu_plus"] = 0.1
    plasticity_params["mu_minus"] = 0.1
    plasticity_params["add"] = 0
    plasticity_params["mlt"] = 0
    plasticity_params["nlta"] = 1
    plasticity_params["FS"] = 0
    spike_ref_mon, spike_pre_mon, spike_post_mon, w_trajs_nlta_100, mu_trajs_nlta_100, post_mon = run_FS_network(neuron_params, plasticity_params, simulation_params)
    w_nlta_100 = np.mean(w_trajs_nlta_100[:, -10:], axis=1)
    filo_index_nlta_100 = np.where(w_nlta_100 < plasticity_params["w0_minus"])[0]
    spine_index_nlta_100 = np.where(w_nlta_100 >= plasticity_params["w0_minus"])[0]

    results_mu_sweep = {}
    results_mu_sweep["w_trajs_add"] = w_trajs_add
    results_mu_sweep["w_trajs_nlta_25"] = w_trajs_nlta_25
    results_mu_sweep["w_trajs_nlta_50"] = w_trajs_nlta_50
    results_mu_sweep["w_trajs_nlta_75"] = w_trajs_nlta_75
    results_mu_sweep["w_trajs_nlta_100"] = w_trajs_nlta_100

    with open("Data/figure_mu_sweep.pickle", 'wb') as handle:
        pickle.dump(dict(results_mu_sweep), handle, protocol=pickle.HIGHEST_PROTOCOL)


    current_time = 0*second
    A_duration = 400*second
    simulation_params["N_pre"] = len(spine_index_FS)
    patterns = []
    pattern = {}
    pattern["start_time"] = current_time
    pattern["duration"] = A_duration
    pattern["c"] = get_vm_corr(0, kappa, c_tot)[spine_index_FS]
    patterns.append(pattern)
    current_time += pattern["duration"]
    simulation_params["total_time"] = current_time
    simulation_params["c"] = c_timed_array(patterns, simulation_params)
    simulation_params["I_ext"] = get_zero_current(simulation_params, 0)
    plasticity_params["w0_minus"] = 0.5

    plasticity_params["add"] = 0
    plasticity_params["mlt"] = 1
    plasticity_params["nlta"] = 0
    plasticity_params["FS"] = 0
    spike_ref_mon, spike_pre_mon, spike_post_mon, w_trajs_mlt, mu_trajs_mlt,  post_mon = run_FS_network(neuron_params, plasticity_params, simulation_params)
    w_mlt = np.mean(w_trajs_mlt[:, -10:], axis=1)
    filo_index_mlt = np.where(w_mlt < plasticity_params["w0_minus"])[0]
    spine_index_mlt = np.where(w_mlt >= plasticity_params["w0_minus"])[0]
    fp_mlt, fm_mlt, factor_mlt, competition_mlt, cooperation_mlt = get_dynamical_terms(w_trajs_mlt, mu_trajs_mlt, patterns, neuron_params, plasticity_params, simulation_params)

    plasticity_params["mu_plus"] = 0.1
    plasticity_params["mu_minus"] = 0.1
    plasticity_params["add"] = 0
    plasticity_params["mlt"] = 0
    plasticity_params["nlta"] = 1
    plasticity_params["FS"] = 0
    spike_ref_mon, spike_pre_mon, spike_post_mon, w_trajs_nlta, mu_trajs_nlta, post_mon = run_FS_network(neuron_params, plasticity_params, simulation_params)
    w_nlta = np.mean(w_trajs_nlta[:, -10:], axis=1)
    filo_index_nlta = np.where(w_nlta < plasticity_params["w0_minus"])[0]
    spine_index_nlta= np.where(w_nlta >= plasticity_params["w0_minus"])[0]
    fp_nlta, fm_nlta, factor_nlta, competition_nlta, cooperation_nlta = get_dynamical_terms(w_trajs_nlta, mu_trajs_nlta, patterns, neuron_params, plasticity_params, simulation_params)




    results = {}
    results["w_trajs_FS"] = w_trajs_FS
    results["mu_trajs_FS"] = mu_trajs_FS
    results["filo_index_FS"] = filo_index_FS
    results["spine_index_FS"] = spine_index_FS
    results["fp_FS"] = fp_FS
    results["fm_FS"] = fm_FS
    results["factor_FS"] = factor_FS
    results["competition_FS"] = competition_FS
    results["cooperation_FS"] = cooperation_FS
    results["cooperation_FS"] = cooperation_FS
    results["w_trajs_add"] = w_trajs_add
    results["mu_trajs_add"] = mu_trajs_add
    results["filo_index_add"] = filo_index_add
    results["spine_index_add"] = spine_index_add
    results["competition_add"] = competition_add
    results["cooperation_add"] = cooperation_add
    results["fp_add"] = fp_add
    results["fm_add"] = fm_add
    results["factor_add"] = factor_add
    results["w_trajs_mlt"] = w_trajs_mlt
    results["mu_trajs_mlt"] = mu_trajs_mlt
    results["competition_mlt"] = competition_mlt
    results["cooperation_mlt"] = cooperation_mlt
    results["fp_mlt"] = fp_mlt
    results["fm_mlt"] = fm_mlt
    results["factor_mlt"] = factor_mlt
    results["w_trajs_nlta"] = w_trajs_nlta
    results["mu_trajs_nlta"] = mu_trajs_nlta
    results["competition_nlta"] = competition_nlta
    results["cooperation_nlta"] = cooperation_nlta
    results["fp_nlta"] = fp_nlta
    results["fm_nlta"] = fm_nlta
    results["factor_nlta"] = factor_nlta

    with open("Data/figure_2DE.pickle", 'wb') as handle:
        pickle.dump(dict(results), handle, protocol=pickle.HIGHEST_PROTOCOL)
    '''

    ##############################################################################
    #parameter sweep across potentation/depression imbalance and total correlation
    ##############################################################################

    num_seeds = 1

    simulation_params = {}
    simulation_params["total_time"] = 150*second
    simulation_params["integration_dt"] = 0.5*ms
    simulation_params["input_dt"] = 1*second
    simulation_params["w_recording_dt"] = 5*second
    simulation_params["N_pre"] = 1000
    simulation_params["r_pre"] = 30*Hz
    simulation_params["N_post"] = 1
    simulation_params["class_pools"] = False
    simulation_params["w"] = 0.3
    simulation_params["seed"] = 0
    plasticity_params["lmbda"] = 0.006


    spans = OrderedDict()
    spans["bias"] =  {"min": 0., "max": 0.05 , "num_values": 2}
    spans["alpha"] =  {"min": 1., "max": 1.75, "num_values": 10}
    spans["c_tot"] =  {"min": 0., "max": 120 , "num_values": 10}
    for key, value in spans.items():
        spans[key]["range"] = np.linspace(value["min"], value["max"], value["num_values"])
    mesh = list(itertools.product(*[span["range"] for span in spans.values()]))

    plasticity_params["add"] = 0
    plasticity_params["mlt"] = 0
    plasticity_params["nlta"] = 0
    plasticity_params["mu_plus"] = 0
    plasticity_params["FS"] = 1
    plasticity_params["mu_minus"] = 0
    experiment_params = [(neuron_params, plasticity_params, simulation_params, bias, alpha, c_tot, num_seeds) for bias, alpha, c_tot in mesh]
    pool = multiprocessing.Pool(processes=128)
    results_list = pool.map(get_RF_stats, experiment_params)
    save_results(results_list, filename='figure_2FG_FS.pickle')


    plasticity_params["add"] = 0
    plasticity_params["mlt"] = 0
    plasticity_params["nlta"] = 0
    plasticity_params["FS"] = 1
    experiment_params = [(neuron_params, plasticity_params, simulation_params, bias, alpha, c_tot, num_seeds) for bias, alpha, c_tot in mesh]
    pool = multiprocessing.Pool(processes=128)
    results_list = pool.map(get_square_RFs, experiment_params)
    save_results(results_list, filename='figure_2FG_FS_square.pickle')


    plasticity_params["add"] = 0
    plasticity_params["mlt"] = 0
    plasticity_params["nlta"] = 1
    plasticity_params["FS"] = 0
    plasticity_params["mu_plus"] = 0.1
    plasticity_params["mu_minus"] = 0.1
    plasticity_params["w0_minus"] = 0.
    experiment_params = [(neuron_params, plasticity_params, simulation_params, bias, alpha, c_tot, num_seeds) for bias, alpha, c_tot in mesh]
    pool = multiprocessing.Pool(processes=128)
    results_list = pool.map(get_RF_stats, experiment_params)
    save_results(results_list, filename='figure_2FG_nlta_00.pickle')

    plasticity_params["add"] = 0
    plasticity_params["nlta"] = 1
    plasticity_params["mlt"] = 0
    plasticity_params["FS"] = 0
    plasticity_params["mu_plus"] = 0.1
    plasticity_params["mu_minus"] = 0.1
    plasticity_params["w0_minus"] = 0.5
    experiment_params = [(neuron_params, plasticity_params, simulation_params, bias, alpha, c_tot, num_seeds) for bias, alpha, c_tot in mesh]
    pool = multiprocessing.Pool(processes=128)
    results_list = pool.map(get_RF_stats, experiment_params)
    save_results(results_list, filename='figure_2FG_nlta_05.pickle')

    plasticity_params["add"] = 1
    plasticity_params["mlt"] = 0
    plasticity_params["nlta"] = 0
    plasticity_params["FS"] = 0
    experiment_params = [(neuron_params, plasticity_params, simulation_params, bias, alpha, c_tot, num_seeds) for bias, alpha, c_tot in mesh]
    pool = multiprocessing.Pool(processes=128)
    results_list = pool.map(get_RF_stats, experiment_params)
    save_results(results_list, filename='figure_2FG_add.pickle')

    plasticity_params["add"] = 1
    plasticity_params["mlt"] = 0
    plasticity_params["nlta"] = 0
    plasticity_params["FS"] = 0
    experiment_params = [(neuron_params, plasticity_params, simulation_params, bias, alpha, c_tot, num_seeds) for bias, alpha, c_tot in mesh]
    pool = multiprocessing.Pool(processes=128)
    results_list = pool.map(get_square_RFs, experiment_params)
    save_results(results_list, filename='figure_2FG_add_square.pickle')
  
    plasticity_params["add"] = 0
    plasticity_params["mlt"] = 1
    plasticity_params["nlta"] = 0
    plasticity_params["FS"] = 0
    experiment_params = [(neuron_params, plasticity_params, simulation_params, bias,  alpha, c_tot, num_seeds) for bias, alpha, c_tot in mesh]
    pool = multiprocessing.Pool(processes=128)
    results_list = pool.map(get_RF_stats, experiment_params)
    save_results(results_list, filename='figure_2FG_mlt.pickle')
