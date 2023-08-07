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

plt.rcParams['figure.figsize'] = (6,5)

def get_consolidation_sign(a, b, threshold):
  return np.where(
      np.logical_and(a >= threshold, b < threshold), 1,
      np.where(
          np.logical_and(a >= threshold, b >= threshold), 0,
          -1
      )
  )

get_consolidation_FS = np.vectorize(get_consolidation_sign, signature='(3, 10, 10),(3, 10, 10), () -> (3, 10, 10)')

get_consolidation_add = np.vectorize(get_consolidation_sign, signature='(10, 10),(10, 10), () -> (10, 10)')


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



def make_3A():

  plt.plot(get_vm_corr(-pi/2, 8, 80), color = 'purple')
  plt.plot(get_vm_corr(pi/2, 8, 40), color = 'yellow', alpha=0.5)
  plt.plot(get_vm_corr(pi/2, 8, 100), color = 'yellow')
  #plt.plot(get_vm_corr(pi/2, 8, 50), color = 'yellow', alpha=0.3)
  #plt.plot(get_vm_corr(pi/2, 8, 80), color = 'yellow', alpha=0.7)
  plt.xlabel(r"Neuron ID", fontsize=20)
  #plt.xticks([0, 250, 500, 750, 1000], [r"$\theta_0-\pi$", r"$\theta_0-\pi/2$", r"$\theta_0$",  r"$\theta_0 +\pi/2$",  r"$\theta_0 +\pi$"], fontsize=18)
  plt.xticks([0, 250, 500, 750, 1000], [r"$-\pi$", r"$-\pi/2$", r"$0$",  r"$\pi/2$",  r"$\pi$"], fontsize=18)
  plt.ylabel(r"correlation $c$", fontsize=20)
  plt.yticks([0, 1], fontsize=18)

  legend_elements = [Line2D([0], [0], color='purple', label=r"pattern A"),
                  Line2D([0], [0], color='yellow', label=r"pattern B"),
                  Line2D([0], [0], color='yellow', label=r"$c_{tot} = 40$", alpha=0.4),
                  Line2D([0], [0], color='yellow', label=r"$c_{tot} = 100$"),                 
                  ]
  plt.legend(handles=legend_elements, frameon=False, fontsize=16, ncol=2, mode="expand")
  sns.despine()
  plt.tight_layout()
  plt.savefig('Figures/3/PNG/A.png', dpi=300, transparent=True)
  plt.savefig('Figures/3/SVG/A.svg', dpi=300, transparent=True)
  plt.close()


def make_3B(w_trajs, spine_index_A, spine_index_B, B_num):
  filo_index = np.setdiff1d(np.setdiff1d(np.arange(1000), spine_index_B), spine_index_A)
  if B_num == 0:
      for w_traj in w_trajs:
        plt.plot(w_traj, color='grey', linewidth=0.1, alpha=0.1)
      for w_traj in w_trajs[spine_index_B]:
        plt.plot(w_traj, color='yellow', linewidth=0.1, alpha=0.2)
      plt.plot(np.mean(w_trajs[spine_index_B], axis=0), color='yellow', linewidth=5)
      plt.plot(np.mean(w_trajs[filo_index], axis=0), color='grey', linewidth=5)
      plt.xticks([0, 200], fontsize=18)

  else:   
    for w_traj in w_trajs:
      plt.plot(w_traj, color='grey', linewidth=0.1, alpha=0.1)
    for w_traj in w_trajs[spine_index_A]:
      plt.plot(w_traj, color='purple', linewidth=0.1, alpha=0.2)
    for w_traj in w_trajs[spine_index_B]:
      plt.plot(w_traj, color='yellow', linewidth=0.1, alpha=0.2)
    plt.plot(np.mean(w_trajs[spine_index_A], axis=0), color='purple', linewidth=5)
    plt.plot(np.mean(w_trajs[spine_index_B], axis=0), color='yellow', linewidth=5)
    plt.plot(np.mean(w_trajs[filo_index], axis=0), color='grey', linewidth=5)
    plt.xticks([0, 200, 400, 600], fontsize=18)
  plt.yticks([0, 1], fontsize=18)
  sns.despine()
  plt.xlabel(r"time (s)", fontsize=20)
  plt.ylabel(r"weight $w$", fontsize=20)
  sns.despine()
  plt.tight_layout()
  plt.savefig('Figures/3/PNG/B{}.png'.format(B_num), dpi=300, transparent=True)
  plt.savefig('Figures/3/SVG/B{}.svg'.format(B_num), dpi=300, transparent=True)
  plt.close()


def make_3C(data_FS, data_add):
  
  tau_mem_FS = np.array(data_FS["tau_mem"]).reshape((3, 10, 10))
  tau_mem_add = np.array(data_add["tau_mem"]).reshape((1, 10, 10))

  titles = [r"$\tau_{mem}, a=0.$", r"$\tau_{mem}, a=0.5$", r"$\tau_{mem}, a = 1$", r"$\tau_{mem}$ (add)"]
  all_taus = [tau_mem_FS[0], tau_mem_FS[1], tau_mem_FS[2], tau_mem_add[0]]
  #vmin, vmax, cmap = get_cmap(r_FS, r_add, r_mult, order='r')
  fig, axs = plt.subplots(1, 4, sharey='row', figsize=(15, 5))
  for index, ax in enumerate(axs):
    ax.set_title(titles[index], fontsize=20)
    im = ax.imshow(all_taus[index], vmin=0, vmax=200, origin='lower', cmap='Greys')
    ax.set_xlabel(r"$c_{tot}$", fontsize=20)
    ax.set_xticks([0, 3, 6, 9], [0, 40, 80, 120])
    ax.set_yticks([0, 3, 6, 9], [1., 1.25, r"1.5", 1.75])
    ax.tick_params(axis='both', labelsize=18)       
    if index == 0:
      ax.set_ylabel(r"$\alpha$", fontsize=20)
      sns.despine()

  fig.subplots_adjust(right=0.8)
  cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
  cbar = fig.colorbar(im, cax=cbar_ax, ticks=[0, 100, 200])
  cbar.ax.tick_params(labelsize=18)
  cbar.ax.set_yticklabels(['0 s', '100 s', '200 s']) 
  plt.show()
  plt.savefig('Figures/3/PNG/C.png', dpi=300, transparent=True)
  plt.savefig('Figures/3/SVG/C.svg', dpi=300, transparent=True)
  plt.close()


def make_3C(cons_FS, cons_add):
    def get_green_cmap():
      c = ["red", "orange", "green"]  
      v = [0, 0.5, 1]
      l = list(zip(v,c))
      cmap=LinearSegmentedColormap.from_list('rg',l, N=256)
      return cmap

    titles = [r"protection $a=0$", r"protection $a=0.5$", r"protection $a=1$", r"add-STDP"]
    all_overlaps = [cons_FS[0], cons_FS[1], cons_FS[2], cons_add]
    fig, axs = plt.subplots(1, 4, sharey='row', figsize=(20, 7.5))
    for index, ax in enumerate(axs):
        ax.set_title(titles[index], fontsize=20)
        im = ax.imshow(all_overlaps[index], vmin=-1, vmax=1, origin='lower', cmap=get_green_cmap())
        ax.set_xlabel(r"total correlation $c_{tot}$", fontsize=20)
        ax.set_xticks([0, 3, 6, 9], [40, 60, 90, 120])
        ax.set_yticks([0, 3, 6, 9], [1.15, 1.35, r"1.55", 1.75])
        ax.tick_params(axis='both', labelsize=18)       
        if index == 0:
         ax.set_ylabel(r"pot./dep. imabalance $\alpha$", fontsize=20)
        sns.despine()

    plt.show()
    plt.savefig('Figures/3/PNG/C.png', dpi=300, transparent=True)
    plt.savefig('Figures/3/SVG/C.svg', dpi=300, transparent=True)
    plt.close()



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


    make_3A()

    with open('Data/figure_3B0.pickle', 'rb') as handle:
      data = pickle.load(handle)
    w_trajs = data["w_trajs"]
    spine_index_B = data["spine_index_B"]
    make_3B(w_trajs, spine_index_A=None, spine_index_B=spine_index_B, B_num=0)

    with open('Data/figure_3B1.pickle', 'rb') as handle:
      data = pickle.load(handle)
    w_trajs = data["w_trajs"]
    spine_index_A = data["spine_index_A"]
    spine_index_B = data["spine_index_B"]

    make_3B(w_trajs, spine_index_A, spine_index_B, B_num=1)

    with open('Data/figure_3B2.pickle', 'rb') as handle:
      data = pickle.load(handle)
    w_trajs = data["w_trajs"]
    spine_index_A = data["spine_index_A"]
    spine_index_B = data["spine_index_B"]
    make_3B(w_trajs, spine_index_A, spine_index_B, B_num=2)

    with open('Data/figure_3B3.pickle', 'rb') as handle:
      data = pickle.load(handle)
    w_trajs = data["w_trajs"]
    spine_index_A = data["spine_index_A"]
    spine_index_B = data["spine_index_B"]
    make_3B(w_trajs, spine_index_A, spine_index_B, B_num=3)


    with open('Data/figure_3_overlap_FS.pickle', 'rb') as handle:
        data_FS = pickle.load(handle)
    overlap_A = np.array(data_FS["overlap_A"]).reshape((3, 10, 10))
    overlap_B = np.array(data_FS["overlap_B"]).reshape((3, 10, 10))
    consolidation_FS = get_consolidation_FS(overlap_A, overlap_B, threshold=0.5)
    with open('Data/figure_3_overlap_add.pickle', 'rb') as handle:
        data_add = pickle.load(handle)
    overlap_A = np.array(data_add["overlap_A"]).reshape((10, 10))
    overlap_B = np.array(data_add["overlap_B"]).reshape((10, 10))
    consolidation_add= get_consolidation_add(overlap_A, overlap_B, threshold=0.5)
    make_3C(consolidation_FS, consolidation_add)