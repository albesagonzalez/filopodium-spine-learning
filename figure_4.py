import itertools
import multiprocessing


from collections import defaultdict, OrderedDict
import pickle
from brian2 import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from aux_functions import get_vm_corr

#create figure directories (if don't exist)
from aux_functions import make_fig_dirs
make_fig_dirs(fig_num='4')
make_fig_dirs(fig_num='supp')

#define plotting variables
coop_colour = plt.cm.tab20(6)
comp_colour = plt.cm.tab20(0)
total_colour = plt.cm.tab20(6)
partial_colour = plt.cm.tab20(15)
no_colour = plt.cm.tab20(4)
filo_colour = plt.cm.tab20(14)
spine_colour = plt.cm.tab20(8)
A_colour = spine_colour
B_colour = plt.cm.tab20(2)

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


def make_4A():

  plt.plot(get_vm_corr(-pi/2, 8, 80), color = A_colour)
  plt.plot(get_vm_corr(pi/2, 8, 40), color = B_colour, alpha=0.5)
  plt.plot(get_vm_corr(pi/2, 8, 80), color = B_colour)
  plt.xlabel(r"Neuron ID", fontsize=20)
  plt.xticks([0, 250, 500, 750, 1000], [r"$-\pi$", r"$-\pi/2$", r"$0$",  r"$\pi/2$",  r"$\pi$"], fontsize=18)
  plt.ylabel(r"correlation $c_i$", fontsize=20)
  plt.yticks([0, 1], fontsize=18)

  legend_elements = [Line2D([0], [0], color=A_colour, label=r"pattern A"),
                  Line2D([0], [0], color=B_colour, label=r"pattern B"),
                  Line2D([0], [0], color=B_colour, label=r"$c_{tot} = 40$", alpha=0.4),
                  Line2D([0], [0], color=B_colour, label=r"$c_{tot} = 80$"),                 
                  ]
  plt.legend(handles=legend_elements, frameon=False, fontsize=18, ncol=2, mode="expand")
  sns.despine()
  plt.tight_layout()
  plt.savefig('Figures/4/PNG/A.png', dpi=300, transparent=True)
  plt.savefig('Figures/4/SVG/A.svg', dpi=300, transparent=True)
  plt.close()


def make_4B(w_trajs, spine_index_A, spine_index_B, B_num):
  filo_index = np.setdiff1d(np.setdiff1d(np.arange(1000), spine_index_B), spine_index_A)
  if B_num == 0:
      for w_traj in w_trajs:
        plt.plot(w_traj, color=filo_colour, linewidth=0.1, alpha=0.1)
      for w_traj in w_trajs[spine_index_B]:
        plt.plot(w_traj, color=B_colour, linewidth=0.1, alpha=0.2)
      plt.plot(np.mean(w_trajs[spine_index_B], axis=0), color=B_colour, linewidth=5)
      plt.plot(np.mean(w_trajs[filo_index], axis=0), color=filo_colour, linewidth=5)
      plt.xticks([0, 200], fontsize=18)

  else:   
    for w_traj in w_trajs:
      plt.plot(w_traj, color=filo_colour, linewidth=0.1, alpha=0.1)
    for w_traj in w_trajs[spine_index_A]:
      plt.plot(w_traj, color=A_colour, linewidth=0.1, alpha=0.2)
    for w_traj in w_trajs[spine_index_B]:
      plt.plot(w_traj, color=B_colour, linewidth=0.1, alpha=0.2)
    plt.plot(np.mean(w_trajs[spine_index_A], axis=0), color=A_colour, linewidth=5)
    plt.plot(np.mean(w_trajs[spine_index_B], axis=0), color=B_colour, linewidth=5)
    plt.plot(np.mean(w_trajs[filo_index], axis=0), color=filo_colour, linewidth=5)
    plt.xticks([0, 200, 400, 600], fontsize=18)
  plt.yticks([0, 1], fontsize=18)
  sns.despine()
  plt.xlabel(r"time (s)", fontsize=20)
  plt.ylabel(r"weight $w$", fontsize=20)
  sns.despine()
  plt.tight_layout()
  plt.savefig('Figures/4/PNG/B{}.png'.format(B_num), dpi=300, transparent=True)
  plt.savefig('Figures/4/SVG/B{}.svg'.format(B_num), dpi=300, transparent=True)
  plt.close()


def make_4D(cons_FS, cons_add):
    def get_green_cmap():
      c = [total_colour, partial_colour, no_colour]  
      v = [0, 0.5, 1]
      l = list(zip(v,c))
      cmap=LinearSegmentedColormap.from_list('rg',l, N=256)
      return cmap

    titles = [r"$\mu_{spine} = 0.1$", "$\mu_{spine} = 0.25$", "$\mu_{spine} = 0.4$", r"add-STDP"]
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
    plt.savefig('Figures/4/PNG/D.png', dpi=300, transparent=True)
    plt.savefig('Figures/4/SVG/D.svg', dpi=300, transparent=True)
    plt.close()


def make_4supp(spine_index_A, spine_index_B, data_FS, data_add):
  for rule, RFs in enumerate([RF_FS[0], RF_FS[1], RF_FS[2], RF_add]):
    N = int(np.sqrt(len(RFs)))
    fig, axs = plt.subplots(N, N, sharex='col', sharey='row')
    for index, RF  in enumerate(RFs):
      i = index//N
      j = index%N
      ax = axs[N-i-1, j]
      filo_index = np.where(RF < 0.5)[0]
      ax.scatter(filo_index, RF[filo_index], color=filo_colour, s=0.1)
      ax.scatter(spine_index_A, RF[spine_index_A], color=A_colour, s=0.1)
      ax.scatter(spine_index_B, RF[spine_index_B], color=B_colour, s=0.1)
      ax.set_yticks([])
      ax.set_xticks([])
      ax.set_ylim([0, 1])
      sns.despine()
    fig.text(0.04, 0.5, r'pot./dep. imbalance $\alpha$', va='center', rotation='vertical')
    fig.text(0.5, 0.04, r'total correlation $c_{tot}$', ha='center')

    plt.savefig('Figures/supp/SVG/{}.svg'.format(rule+13), dpi=300, transparent=True)
    plt.savefig('Figures/supp/PNG/{}.png'.format(rule+13), dpi=300, transparent=True)
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


    make_4A()

    with open('Data/figure_4B1.pickle', 'rb') as handle:
      data = pickle.load(handle)
    w_trajs = data["w_trajs"]
    spine_index_B = data["spine_index_B"]
    make_4B(w_trajs, spine_index_A=None, spine_index_B=spine_index_B, B_num=0)

    with open('Data/figure_4B2.pickle', 'rb') as handle:
      data = pickle.load(handle)
    w_trajs = data["w_trajs"]
    spine_index_A = data["spine_index_A"]
    spine_index_B = data["spine_index_B"]

    make_4B(w_trajs, spine_index_A, spine_index_B, B_num=1)

    with open('Data/figure_4B3.pickle', 'rb') as handle:
      data = pickle.load(handle)
    w_trajs = data["w_trajs"]
    spine_index_A = data["spine_index_A"]
    spine_index_B = data["spine_index_B"]
    make_4B(w_trajs, spine_index_A, spine_index_B, B_num=2)

    with open('Data/figure_4B4.pickle', 'rb') as handle:
      data = pickle.load(handle)
    w_trajs = data["w_trajs"]
    spine_index_A = data["spine_index_A"]
    spine_index_B = data["spine_index_B"]
    make_4B(w_trajs, spine_index_A, spine_index_B, B_num=3)


    with open('Data/figure_4_overlap_FS.pickle', 'rb') as handle:
        data_FS = pickle.load(handle)
    overlap_A = np.array(data_FS["overlap_A"]).reshape((3, 10, 10))
    overlap_B = np.array(data_FS["overlap_B"]).reshape((3, 10, 10))
    consolidation_FS = get_consolidation_FS(overlap_A, overlap_B, threshold=0.5)
    with open('Data/figure_4_overlap_add.pickle', 'rb') as handle:
        data_add = pickle.load(handle)
    overlap_A = np.array(data_add["overlap_A"]).reshape((10, 10))
    overlap_B = np.array(data_add["overlap_B"]).reshape((10, 10))
    consolidation_add= get_consolidation_add(overlap_A, overlap_B, threshold=0.5)
    make_4D(consolidation_FS, consolidation_add)

    RF_FS = np.array(data_FS["RF"]).reshape((3, 100, -1))
    RF_add = np.array(data_add["RF"]).reshape((100, -1))

    make_4supp(spine_index_A, spine_index_B, RF_FS, RF_add)