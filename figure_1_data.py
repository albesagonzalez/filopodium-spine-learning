import seaborn as sns

from brian2 import *

import matplotlib.cm as cm
import pickle
import os

from aux import c_timed_array, get_zero_current, get_dynamical_terms, get_vm_corr, get_q_a
from run_network_functions import run_FS_network

from aux import make_data_dir
make_data_dir()


def run_simulation_1(pattern, filename):

    patterns = []
    current_time = 0*second
    c = pattern["c"]
    patterns.append(pattern)
    current_time += pattern["duration"]
    simulation_params["total_time"] = current_time
    simulation_params["c"] = c_timed_array(patterns, simulation_params)
    simulation_params["I_ext"] = get_zero_current(simulation_params, 0)

    spike_ref_mon, spike_pre_mon, spike_post_mon, w_trajs_FS, mu_trajs_FS, post_mon  = run_FS_network(neuron_params, plasticity_params, simulation_params)
    fp_FS, fm_FS, factor_FS, competition_FS, cooperation_FS = get_dynamical_terms(w_trajs_FS, mu_trajs_FS, patterns, neuron_params, plasticity_params, simulation_params)
    w_FS = np.mean(w_trajs_FS[:, -10:], axis=1)
    filo_index_FS = np.where(w_FS < plasticity_params["w0_minus"])[0]
    spine_index_FS = np.where(w_FS >= plasticity_params["w0_minus"])[0]

    plasticity_params["add"] = 1
    plasticity_params["mlt"] = 0
    plasticity_params["nlta"] = 0
    plasticity_params["FS"] = 0
    spike_ref_mon, spike_pre_mon, spike_post_mon, w_trajs_add, mu_trajs_add, post_mon, = run_FS_network(neuron_params, plasticity_params, simulation_params)
    w_add = np.mean(w_trajs_add[:, -10:], axis=1)
    filo_index_add = np.where(w_add < plasticity_params["w0_minus"])[0]
    spine_index_add = np.where(w_add >= plasticity_params["w0_minus"])[0]
    fp_add, fm_add, factor_add, competition_add, cooperation_add = get_dynamical_terms(w_trajs_add, mu_trajs_add, patterns, neuron_params, plasticity_params, simulation_params)

    current_time = 0*second
    A_duration = 400*second
    #simulation_params["N_pre"] = len(spine_index_FS)
    patterns_sub = []
    pattern = {}
    pattern["start_time"] = current_time
    pattern["duration"] = A_duration
    #pattern["c"] = c[spine_index_FS]
    pattern["c"] = c
    patterns_sub.append(pattern)
    current_time += pattern["duration"]
    simulation_params["total_time"] = current_time
    simulation_params["c"] = c_timed_array(patterns_sub, simulation_params)
    simulation_params["I_ext"] = get_zero_current(simulation_params, 0)

    plasticity_params["mu_plus"] = np.mean(mu_trajs_FS[spine_index_FS, -1])
    plasticity_params["mu_minus"] = np.mean(mu_trajs_FS[spine_index_FS, -1])
    print(plasticity_params["mu_minus"])
    plasticity_params["add"] = 0
    plasticity_params["mlt"] = 0
    plasticity_params["nlta"] = 1
    plasticity_params["FS"] = 0
    simulation_params["w"] = 0.3
    spike_ref_mon, spike_pre_mon, spike_post_mon, w_trajs_nlta, mu_trajs_nlta, post_mon = run_FS_network(neuron_params, plasticity_params, simulation_params)
    w_nlta = np.mean(w_trajs_nlta[:, -10:], axis=1)
    filo_index_nlta = np.where(w_nlta < plasticity_params["w0_minus"])[0]
    spine_index_nlta = np.where(w_nlta >= plasticity_params["w0_minus"])[0]
    fp_nlta, fm_nlta, factor_nlta, competition_nlta, cooperation_nlta = get_dynamical_terms(w_trajs_nlta, mu_trajs_nlta, patterns_sub, neuron_params, plasticity_params, simulation_params)

    results = {}
    results["patterns"] = patterns 
    results["w_trajs_FS"] = w_trajs_FS
    results["mu_trajs_FS"] = mu_trajs_FS
    results["filo_index_FS"] = filo_index_FS
    results["spine_index_FS"] = spine_index_FS
    results["fp_FS"] = fp_FS
    results["fm_FS"] = fm_FS
    results["factor_FS"] = factor_FS
    results["competition_FS"] = competition_FS
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
    results["w_trajs_nlta"] = w_trajs_nlta
    results["mu_trajs_nlta"] = mu_trajs_nlta
    results["filo_index_nlta"] = filo_index_nlta
    results["spine_index_nlta"] = spine_index_nlta
    results["competition_nlta"] = competition_nlta
    results["cooperation_nlta"] = cooperation_nlta
    results["fp_nlta"] = fp_nlta
    results["fm_nlta"] = fm_nlta
    results["factor_nlta"] = factor_nlta
        
    dir = "Data/{}".format(filename)
    with open(dir, 'wb') as handle:
        pickle.dump(dict(results), handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

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
    plasticity_params["FS"] = 1
    plasticity_params["mu_plus"] = 0
    plasticity_params["mu_minus"] = 0
    plasticity_params["tau_mu"] = 20*second
    plasticity_params["mu_3"] = 1
    plasticity_params["tau_plus"] = 20*ms
    plasticity_params["tau_minus"] = 20*ms
    plasticity_params["w0_plus"] = 1
    plasticity_params["lmbda"] = 0.006
    plasticity_params["w0_minus"] = 0.5
    plasticity_params["alpha"] = 1.35
    plasticity_params["mu_filo"] = 0.01
    plasticity_params["mu_spine"] = 0.1
    plasticity_params["q"], plasticity_params["a"] = get_q_a(plasticity_params)

    #define network architecture and simulation specs
    simulation_params = {}
    simulation_params["total_time"] = 400*second
    simulation_params["integration_dt"] = 0.5*ms
    simulation_params["input_dt"] = 1*second
    simulation_params["w_recording_dt"] = 1*second
    simulation_params["N_pre"] = 1000
    simulation_params["r_pre"] = 30*Hz
    simulation_params["N_post"] = 1
    simulation_params["class_pools"] = False
    simulation_params["w"] = 0.3
    simulation_params["seed"] = 0

    c_tot = 60

    current_time = 0*second
    max_time = 400*second
    pattern = {}

    pattern["start_time"] = current_time
    pattern["duration"] = max_time

    c_mu = 0.3
    c_sigma = 0.1
    pattern["c"] = np.clip(c_mu + c_sigma*np.random.randn(1000), 0, None)
    pattern["c"] = pattern["c"]/np.sum(pattern["c"])*c_tot
    run_simulation_1(pattern, filename='figure_1_gaussian.pickle')


    pattern["c"] = np.zeros((1000))
    pattern["c"][np.arange(400, 600)] = c_tot/200
    run_simulation_1(pattern, filename='figure_1_squared.pickle')

    kappa = 8
    pattern["c"] = get_vm_corr(0, kappa, c_tot)
    corr_matrix = np.zeros((1000, 1000))
    for i in range(1000):
        corr_matrix[i] = np.roll(pattern["c"], i)
    results = {}
    results["corr_matrix"] = corr_matrix
    dir = "Data/figure_corr_heatmap.pickle"
    with open(dir, 'wb') as handle:
        pickle.dump(dict(results), handle, protocol=pickle.HIGHEST_PROTOCOL)

    run_simulation_1(pattern, filename='figure_1_von_mises.pickle')
















