import seaborn as sns

from brian2 import *

import matplotlib.cm as cm
import pickle

from aux import c_timed_array, get_zero_current
from run_network_functions_mult import run_FS_network


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
    plasticity_params["mult"] = 0
    plasticity_params["q"] = 8
    plasticity_params["a"] = 0.
    plasticity_params["mu_plus"] = 0
    plasticity_params["mu_minus"] = 0
    plasticity_params["tau_mu"] = 20*second
    plasticity_params["mu_3"] = 1
    plasticity_params["tau_plus"] = 20*ms
    plasticity_params["tau_minus"] = 20*ms
    plasticity_params["w0_plus"] = 1
    plasticity_params["lmbda"] = 0.006
    plasticity_params["w0_minus"] = 0.5
    plasticity_params["alpha"] = 1.5

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

    c_mu = 0.15
    c_sigma = 0.1
    patterns = []
    current_time = 0*second
    max_time = 400*second
    pattern = {}
    pattern["start_time"] = current_time
    pattern["duration"] = max_time
    pattern["c"] = np.clip(c_mu + c_sigma*np.random.randn(1000), 0, None)
    patterns.append(pattern)
    current_time += pattern["duration"]
    simulation_params["total_time"] = current_time
    simulation_params["c"] = c_timed_array(patterns, simulation_params)
    simulation_params["I_ext"] = get_zero_current(simulation_params, 0)
    spike_ref_mon, spike_pre_mon, spike_post_mon, w_trajs, mu_trajs, post_mon  = run_FS_network(neuron_params, plasticity_params, simulation_params)
    w = np.mean(w_trajs[:, 50:], axis=1)

    results = {}
    results["patterns"] = patterns
    results["w"] = w
        
    with open("Data/figure_1.pickle", 'wb') as handle:
        pickle.dump(dict(results), handle, protocol=pickle.HIGHEST_PROTOCOL)