import itertools
import multiprocessing


from collections import defaultdict, OrderedDict
import pickle
from brian2 import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from aux import c_timed_array, get_zero_current, get_dynamical_terms
from run_network_functions import run_FS_network

from scipy.stats import vonmises


def save_results_nonoverlap(results_list, filename):
    results = {}
    results["tau_mem"] = []

    for w_trajs, spine_index, tau_mem in results_list:
        results["tau_mem"].append(tau_mem)

    with open('Data/{}'.format(filename), 'wb') as handle:
        pickle.dump(dict(results), handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_results_overlap(results_list, filename):
    results = {}
    results["overlap_A"], results["overlap_B"] = [], []

    for overlap_A, overlap_B in results_list:
        results["overlap_A"].append(overlap_A)
        results["overlap_B"].append(overlap_B)

    with open('Data/{}'.format(filename), 'wb') as handle:
        pickle.dump(dict(results), handle, protocol=pickle.HIGHEST_PROTOCOL)



def get_vm_corr(pref_deg, kappa, c_tot):
   x = np.linspace(pi + pref_deg, -pi + pref_deg, 1000)
   vm = vonmises(kappa=kappa)
   return vm.pdf(x)/np.sum(vm.pdf(x))*c_tot


def get_lifetime(w_trajs, start_time, spines, pp, sp):
    start_index = int(start_time/sp["w_recording_dt"])
    for time_index in range(start_index, w_trajs.shape[1]):
        n_decayed = len(np.where(w_trajs[spines, time_index] <= pp["w0_minus"])[0])
        if n_decayed >= len(spines)/2:
            break
    return (time_index-start_index)*sp["w_recording_dt"]


    
def get_overlaps(params):
    neuron_params, plasticity_params, simulation_params, r_pre, a, alpha, kappa_B, c_tot_B, delta_theta, return_trajs = params

    simulation_params["r_pre"] = r_pre
    plasticity_params["alpha"] = alpha
    plasticity_params["a"] = a

    current_time = 0*second
    A_duration = 400*second
    B_duration = 400*second

    c_tot_A = 100
    kappa_A = 8

    patterns = []
    pattern = {}
    pattern["start_time"] = current_time
    pattern["duration"] = B_duration
    pattern["c"] = get_vm_corr(-pi/2 - delta_theta, kappa_B, c_tot_B)
    patterns.append(pattern)
    current_time += pattern["duration"]

    simulation_params["total_time"] = current_time
    simulation_params["c"] = c_timed_array(patterns, simulation_params)
    simulation_params["I_ext"] = get_zero_current(simulation_params, 0)

    spike_ref_mon, spike_pre_mon, spike_post_mon, w_trajs_B, mu_trajs, post_mon = run_FS_network(neuron_params, plasticity_params, simulation_params)
    w_B = np.mean(w_trajs_B[:,-10:], axis=1)
    spine_index_B = np.where(w_B >= plasticity_params["w0_minus"])[0]

    current_time = 0*second
    A_duration = 400*second
    B_duration = 400*second

    patterns = []
    pattern = {}
    pattern["start_time"] = current_time
    pattern["duration"] = A_duration
    pattern["c"] = get_vm_corr(-pi/2, kappa_A, c_tot_A)
    patterns.append(pattern)
    current_time += pattern["duration"]

    pattern = {}
    pattern["start_time"] = current_time
    pattern["duration"] = B_duration
    pattern["c"] = get_vm_corr(-pi/2 - delta_theta, kappa_B, c_tot_B)
    patterns.append(pattern)
    current_time += pattern["duration"]

    simulation_params["total_time"] = current_time
    simulation_params["c"] = c_timed_array(patterns, simulation_params)
    simulation_params["I_ext"] = get_zero_current(simulation_params, 0)

    spike_ref_mon, spike_pre_mon, spike_post_mon, w_trajs_AB, mu_trajs, post_mon = run_FS_network(neuron_params, plasticity_params, simulation_params)

    A_stop_index = int(patterns[1]["start_time"]/simulation_params["w_recording_dt"]) - 1
    w_A = np.mean(w_trajs_AB[:,A_stop_index-10:A_stop_index], axis=1)
    w_B_prime = np.mean(w_trajs_AB[:,-10:], axis=1)

    spine_index_A = np.where(w_A >= plasticity_params["w0_minus"])[0]

    overlap_A = np.dot(w_A, w_B_prime)/(np.linalg.norm(w_A)*np.linalg.norm(w_B_prime))
    overlap_B = np.dot(w_B, w_B_prime)/(np.linalg.norm(w_B)*np.linalg.norm(w_B_prime))
    
    if return_trajs:
        return w_trajs_B, w_trajs_AB, spine_index_A, spine_index_B, overlap_A, overlap_B
    else:
        #return get_consolidation_vectorized(overlap_A, overlap_B, threshold=0.7)  
        return overlap_A, overlap_B



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
    neuron_params["g_exc_hat"] = 0.148*nsiemens
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
    plasticity_params["w0_minus"] = 0.5
    plasticity_params["lmbda"] = 0.006
    #plasticity_params["alpha"] = 1.75
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
    '''

    c_tot_A = 80
    c_tot_B = 80
    kappa_A = 8
    kappa_B = 8


    current_time = 0*second
    patterns = []
    pattern = {}
    pattern["start_time"] = current_time
    pattern["duration"] = 200*second
    pattern["c"] = get_vm_corr(pi/2, kappa_B, c_tot_B)
    patterns.append(pattern)
    current_time += pattern["duration"]

    simulation_params["total_time"] = current_time
    simulation_params["c"] = c_timed_array(patterns, simulation_params)
    simulation_params["I_ext"] = get_zero_current(simulation_params, 0)

    plasticity_params["a"] = 0.
    plasticity_params["add"] = 0
    plasticity_params["mult"] = 0
    spike_ref_mon, spike_pre_mon, spike_post_mon, w_trajs, mu_trajs, post_mon = run_FS_network(neuron_params, plasticity_params, simulation_params)
    w_B = np.mean(w_trajs[:,-10:], axis=1)
    spine_index_B = np.where(w_B >= plasticity_params["w0_minus"])[0]

    results = {}
    results["w_trajs"] = w_trajs
    results["spine_index_B"] = spine_index_B

    with open("Data/figure_3B0.pickle", 'wb') as handle:
        pickle.dump(dict(results), handle, protocol=pickle.HIGHEST_PROTOCOL)


    current_time = 0*second
    patterns = []
    pattern = {}
    pattern["start_time"] = 0*second
    pattern["duration"] = 200*second
    pattern["c"] = get_vm_corr(-pi/2, kappa_A, c_tot_A)
    patterns.append(pattern)
    current_time += pattern["duration"]

    pattern = {}
    pattern["start_time"] = current_time
    pattern["duration"] = 400*second
    pattern["c"] = get_vm_corr(pi/2, kappa_B, c_tot_B)
    patterns.append(pattern)
    current_time += pattern["duration"]

    simulation_params["total_time"] = current_time
    simulation_params["c"] = c_timed_array(patterns, simulation_params)
    simulation_params["I_ext"] = get_zero_current(simulation_params, 0)


    plasticity_params["a"] = 0.
    plasticity_params["add"] = 0
    plasticity_params["mult"] = 0
    spike_ref_mon, spike_pre_mon, spike_post_mon, w_trajs, mu_trajs, post_mon = run_FS_network(neuron_params, plasticity_params, simulation_params)
    A_stop_index = int(patterns[1]["start_time"]/simulation_params["w_recording_dt"]) - 1
    w_A = np.mean(w_trajs[:,A_stop_index-10:A_stop_index], axis=1)
    spine_index_A = np.where(w_A >= plasticity_params["w0_minus"])[0]

    results = {}
    results["w_trajs"] = w_trajs
    results["spine_index_A"] = spine_index_A
    results["spine_index_B"] = spine_index_B

    with open("Data/figure_3B1.pickle", 'wb') as handle:
        pickle.dump(dict(results), handle, protocol=pickle.HIGHEST_PROTOCOL)


    plasticity_params["a"] = 0.3
    spike_ref_mon, spike_pre_mon, spike_post_mon, w_trajs, mu_trajs, post_mon = run_FS_network(neuron_params, plasticity_params, simulation_params)
    A_stop_index = int(patterns[1]["start_time"]/simulation_params["w_recording_dt"]) - 1
    w_A = np.mean(w_trajs[:,A_stop_index-10:A_stop_index], axis=1)
    spine_index_A = np.where(w_A >= plasticity_params["w0_minus"])[0]

    results = {}
    results["w_trajs"] = w_trajs
    results["spine_index_A"] = spine_index_A
    results["spine_index_B"] = spine_index_B

    with open("Data/figure_3B2.pickle", 'wb') as handle:
        pickle.dump(dict(results), handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    plasticity_params["a"] = 1
    spike_ref_mon, spike_pre_mon, spike_post_mon, w_trajs, mu_trajs, post_mon = run_FS_network(neuron_params, plasticity_params, simulation_params)
    A_stop_index = int(patterns[1]["start_time"]/simulation_params["w_recording_dt"]) - 1
    w_A = np.mean(w_trajs[:,A_stop_index-10:A_stop_index], axis=1)
    spine_index_A = np.where(w_A >= plasticity_params["w0_minus"])[0]

    results = {}
    results["w_trajs"] = w_trajs
    results["spine_index_A"] = spine_index_A
    results["spine_index_B"] = spine_index_B

    with open("Data/figure_3B3.pickle", 'wb') as handle:
        pickle.dump(dict(results), handle, protocol=pickle.HIGHEST_PROTOCOL)


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
    
    '''

    spans = OrderedDict()
    spans["r_pre"] = {"min": 30*Hz, "max": 30*Hz, "num_values": 1}
    spans["a"] = {"min": 0., "max": 1., "num_values": 3}
    spans["alpha"] =  {"min": 1.15, "max": 1.75, "num_values": 10}
    spans["kappa"] =  {"min": 8, "max": 8, "num_values": 1}
    spans["c_tot"] =  {"min": 40., "max": 120., "num_values": 10}
    spans["delta_theta"] =  {"min": pi, "max": pi, "num_values": 1}
    '''
    for key, value in spans.items():
        spans[key]["range"] = np.linspace(value["min"], value["max"], value["num_values"])
    mesh = list(itertools.product(*[span["range"] for span in spans.values()]))

    plasticity_params["add"] = 0
    plasticity_params["mult"] = 0
    experiment_params = [(neuron_params, plasticity_params, simulation_params, r_pre, a, alpha, kappa, c_tot, delta_theta, False) for r_pre, a, alpha, kappa, c_tot, delta_theta in mesh]
    pool = multiprocessing.Pool(processes=128)
    results_list = pool.map(get_overlaps, experiment_params)
    save_results_overlap(results_list, filename='figure_3_overlap_FS.pickle')

    '''

    spans["a"] = {"min": 0., "max": 0., "num_values": 1}
    for key, value in spans.items():
        spans[key]["range"] = np.linspace(value["min"], value["max"], value["num_values"])
    mesh = list(itertools.product(*[span["range"] for span in spans.values()]))

    plasticity_params["add"] = 1
    plasticity_params["mult"] = 0
    experiment_params = [(neuron_params, plasticity_params, simulation_params, r_pre, a, alpha, kappa, c_tot, delta_theta, False) for r_pre, a, alpha, kappa, c_tot, delta_theta in mesh]
    pool = multiprocessing.Pool(processes=128)
    results_list = pool.map(get_overlaps, experiment_params)
    #make_grid_plot(results_list, show='w_trajs', filename='fig_3_grid_overlap_add.png')
    save_results_overlap(results_list, filename='figure_3_overlap_add.pickle')
