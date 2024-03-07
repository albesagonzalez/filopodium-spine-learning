import itertools
import multiprocessing


from collections import defaultdict, OrderedDict
import pickle
from brian2 import *

from aux import c_timed_array, get_zero_current, get_vm_corr, get_q_a
from run_network_functions import run_FS_network

from aux import make_data_dir
make_data_dir()


def save_results_overlap(results_list, filename):
    results = {}
    results["RF"], results["overlap_A"], results["overlap_B"] = [], [],[] 

    for RF, overlap_A, overlap_B in results_list:
        results["RF"].append(RF)
        results["overlap_A"].append(overlap_A)
        results["overlap_B"].append(overlap_B)

    with open('Data/{}'.format(filename), 'wb') as handle:
        pickle.dump(dict(results), handle, protocol=pickle.HIGHEST_PROTOCOL)

    
def get_overlaps(params):
    neuron_params, plasticity_params, simulation_params, r_pre, mu_spine, alpha, kappa_B, c_tot_B, delta_theta, num_seeds = params

    for seed in range(num_seeds):
        simulation_params["seed"] = seed

        simulation_params["r_pre"] = r_pre
        plasticity_params["alpha"] = alpha
        plasticity_params["mu_spine"] = mu_spine
        plasticity_params["q"], plasticity_params["a"] = get_q_a(plasticity_params)

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
        w_AB= np.mean(w_trajs_AB[:,-10:], axis=1)

        overlap_A = np.dot(w_A, w_AB)/(np.linalg.norm(w_A)*np.linalg.norm(w_AB))
        overlap_B = np.dot(w_B, w_AB)/(np.linalg.norm(w_B)*np.linalg.norm(w_AB))
        

        try:
            overlap_A_av += overlap_A
            overlap_B_av += overlap_B

        except:
            overlap_A_av = overlap_A
            overlap_B_av = overlap_B


    return w_AB, overlap_A_av/num_seeds, overlap_B_av/num_seeds



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
    plasticity_params["mlt"] = 0
    plasticity_params["nlta"] = 0
    plasticity_params["FS"] = 1
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
    plasticity_params["mu_filo"] = 0.01
    plasticity_params["mu_spine"] = 0.1
    plasticity_params["q"], plasticity_params["a"] = get_q_a(plasticity_params)

    #define network architecture and simulation specs
    simulation_params = {}
    simulation_params["total_time"] = 200*second
    simulation_params["integration_dt"] = 0.5*ms
    simulation_params["input_dt"] = 1*second
    simulation_params["w_recording_dt"] = 1*second
    simulation_params["N_pre"] = 1000
    simulation_params["r_pre"] = 30*Hz
    simulation_params["N_post"] = 1
    simulation_params["class_pools"] = False
    simulation_params["w"] = 0.3
    simulation_params["seed"] = 0


    c_tot_A = 60
    c_tot_B = 60
    kappa_A = 8
    kappa_B = 8

    #####################################
    #simulate RF formation with pattern B
    #####################################

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

  
    plasticity_params["add"] = 0
    plasticity_params["mlt"] = 0
    spike_ref_mon, spike_pre_mon, spike_post_mon, w_trajs, mu_trajs, post_mon = run_FS_network(neuron_params, plasticity_params, simulation_params)
    w_B = np.mean(w_trajs[:,-10:], axis=1)
    spine_index_B = np.where(w_B >= plasticity_params["w0_minus"])[0]

    results = {}
    results["w_trajs"] = w_trajs
    results["spine_index_B"] = spine_index_B

    with open("Data/figure_4B1.pickle", 'wb') as handle:
        pickle.dump(dict(results), handle, protocol=pickle.HIGHEST_PROTOCOL)


    ########################################################
    #simulate RF formation with pattern A and then pattern B
    ########################################################

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

    ###############################
    #for a = 0. (total overwriting)
    ###############################


    spike_ref_mon, spike_pre_mon, spike_post_mon, w_trajs, mu_trajs, post_mon = run_FS_network(neuron_params, plasticity_params, simulation_params)
    A_stop_index = int(patterns[1]["start_time"]/simulation_params["w_recording_dt"]) - 1
    w_A = np.mean(w_trajs[:,A_stop_index-10:A_stop_index], axis=1)
    spine_index_A = np.where(w_A >= plasticity_params["w0_minus"])[0]

    results = {}
    results["w_trajs"] = w_trajs
    results["spine_index_A"] = spine_index_A
    results["spine_index_B"] = spine_index_B

    with open("Data/figure_4B2.pickle", 'wb') as handle:
        pickle.dump(dict(results), handle, protocol=pickle.HIGHEST_PROTOCOL)


    ##################################
    #for a = 0.4 (partial overwriting)
    ##################################

    plasticity_params["mu_spine"] = 0.15
    plasticity_params["q"], plasticity_params["a"] = get_q_a(plasticity_params)
    #plasticity_params["a"] = 0.4
    spike_ref_mon, spike_pre_mon, spike_post_mon, w_trajs, mu_trajs, post_mon = run_FS_network(neuron_params, plasticity_params, simulation_params)
    A_stop_index = int(patterns[1]["start_time"]/simulation_params["w_recording_dt"]) - 1
    w_A = np.mean(w_trajs[:,A_stop_index-10:A_stop_index], axis=1)
    spine_index_A = np.where(w_A >= plasticity_params["w0_minus"])[0]

    results = {}
    results["w_trajs"] = w_trajs
    results["spine_index_A"] = spine_index_A
    results["spine_index_B"] = spine_index_B

    with open("Data/figure_4B3.pickle", 'wb') as handle:
        pickle.dump(dict(results), handle, protocol=pickle.HIGHEST_PROTOCOL)

        
    ###########################
    #for a = 1 (no overwriting)
    ###########################
    
    plasticity_params["mu_spine"] = 0.3
    plasticity_params["q"], plasticity_params["a"] = get_q_a(plasticity_params)
    #plasticity_params["a"] = 1
    spike_ref_mon, spike_pre_mon, spike_post_mon, w_trajs, mu_trajs, post_mon = run_FS_network(neuron_params, plasticity_params, simulation_params)
    A_stop_index = int(patterns[1]["start_time"]/simulation_params["w_recording_dt"]) - 1
    w_A = np.mean(w_trajs[:,A_stop_index-10:A_stop_index], axis=1)
    spine_index_A = np.where(w_A >= plasticity_params["w0_minus"])[0]

    results = {}
    results["w_trajs"] = w_trajs
    results["spine_index_A"] = spine_index_A
    results["spine_index_B"] = spine_index_B

    with open("Data/figure_4B4.pickle", 'wb') as handle:
        pickle.dump(dict(results), handle, protocol=pickle.HIGHEST_PROTOCOL)

        
    ##############################################################################
    #parameter sweep across potentation/depression imbalance and total correlation
    ##############################################################################

    num_seeds = 1

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
    


    ########
    #FS-STDP
    ########

    spans = OrderedDict()
    spans["r_pre"] = {"min": 30*Hz, "max": 30*Hz, "num_values": 1}
    spans["mu_spine"] = {"min": 0.1, "max": 0.4, "num_values": 3}
    spans["alpha"] =  {"min": 1.15, "max": 1.75, "num_values": 10}
    spans["kappa"] =  {"min": 8, "max": 8, "num_values": 1}
    spans["c_tot"] =  {"min": 40., "max": 120., "num_values": 10}
    spans["delta_theta"] =  {"min": pi, "max": pi, "num_values": 1}


    for key, value in spans.items():
        spans[key]["range"] = np.linspace(value["min"], value["max"], value["num_values"])
    mesh = list(itertools.product(*[span["range"] for span in spans.values()]))

    plasticity_params["add"] = 0
    plasticity_params["mlt"] = 0
    plasticity_params["nlta"] = 0
    plasticity_params["FS"] = 1
    experiment_params = [(neuron_params, plasticity_params, simulation_params, r_pre, mu_spine, alpha, kappa, c_tot, delta_theta, num_seeds) for r_pre, mu_spine, alpha, kappa, c_tot, delta_theta in mesh]
    pool = multiprocessing.Pool(processes=128)
    results_list = pool.map(get_overlaps, experiment_params)
    save_results_overlap(results_list, filename='figure_4_overlap_FS.pickle')


    ########
    #add-STDP
    ########

    spans["mu_spine"] = {"min": 0., "max": 0., "num_values": 1}
    for key, value in spans.items():
        spans[key]["range"] = np.linspace(value["min"], value["max"], value["num_values"])
    mesh = list(itertools.product(*[span["range"] for span in spans.values()]))

    plasticity_params["add"] = 1
    plasticity_params["mlt"] = 0
    experiment_params = [(neuron_params, plasticity_params, simulation_params, r_pre, mu_spine, alpha, kappa, c_tot, delta_theta, num_seeds) for r_pre, mu_spine, alpha, kappa, c_tot, delta_theta in mesh]
    pool = multiprocessing.Pool(processes=128)
    results_list = pool.map(get_overlaps, experiment_params)
    save_results_overlap(results_list, filename='figure_4_overlap_add.pickle')
