import itertools
import multiprocessing

from collections import defaultdict, OrderedDict
import pickle
from brian2 import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from argparse import Namespace

from aux import c_timed_array, get_zero_current
#from run_network_functions import run_FS_network
from run_network_functions_mult import run_FS_network

from sys import exit

from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.feature_selection import r_regression

from scipy.stats import vonmises
from scipy.stats import pearsonr

plt.rcParams['figure.figsize'] = (6,5)

def get_rg_cmap():
  c = ["red", "white", "green"]  
  v = [0, 0.5, 1]
  l = list(zip(v,c))
  cmap=LinearSegmentedColormap.from_list('rg',l, N=256)
  return cmap

def make_grid_plot(results_list, show, rule):

  N = int(np.sqrt(len(results_list)))

  fig, axs = plt.subplots(N, N, sharex='col', sharey='row')

  # Loop over each subplot and customize it
  for index, (w_trajs, filo_num, filo_mu, filo_sgm, spine_num, spine_mu, spine_sgm, DI, corr) in enumerate(results_list):
        i = index//N
        j = index%N
        ax = axs[N-i-1, j]
        if show=='w_trajs':
          for w_traj in w_trajs:
              ax.plot(w_traj, color='grey', linewidth=0.1)
          #ax.set_xticks([0, 200])
        if show=='RFs':
           ax.scatter(np.arange(w_trajs.shape[0]), np.mean(w_trajs[:, -10:], axis=1), s=0.1)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylim([0, 1])
        #ax.tick_params(axis='both', labelsize=18)
        sns.despine()
        if i == N - 1:
            if show=='w_trajs':
              #ax.set_xlabel(r"time (s)", fontsize=20)
              pass
            if show=='RFs':
              #ax.set_xticks([0, 250, 500, 750, 1000], [r"$-\pi$", r"$-\pi/2$", r"$0$",  r"$\pi/2$",  r"$\pi$"], fontsize=18)
              #ax.set_xlabel(r"Neuron ID", fontsize=20)
              pass
        if j == 0:
            #ax.set_ylabel(r"$w$", fontsize=20)
            pass
        
  # Show the plot
  plt.savefig('grid_{}_{}.png'.format(show, rule), dpi=300)



def get_vm_corr(pref_deg, kappa, c_tot):
   x = np.linspace(pi + pref_deg, -pi + pref_deg, 1000)
   vm = vonmises(kappa=kappa)
   return vm.pdf(x)/np.sum(vm.pdf(x))*c_tot




def make_2A():
  for index, c_tot in enumerate([60, 120]):
    plt.plot(get_vm_corr(0, kappa=8, c_tot=c_tot), color = 'red', alpha=(index/2+0.5))
  plt.xlabel(r"Neuron ID", fontsize=20)
  #plt.xticks([0, 250, 500, 750, 1000], [r"$\theta_0-\pi$", r"$\theta_0-\pi/2$", r"$\theta_0$",  r"$\theta_0 +\pi/2$",  r"$\theta_0 +\pi$"], fontsize=18)
  plt.xticks([0, 250, 500, 750, 1000], [r"$-\pi$", r"$-\pi/2$", r"$0$",  r"$\pi/2$",  r"$\pi$"], fontsize=18)
  plt.ylabel(r"correlation $c_i$", fontsize=20)
  plt.yticks([0, 1], fontsize=18)

  legend_elements = [Line2D([0], [0], color='black', label=r"$c_{tot} = 60$",  alpha=0.43),
                    Line2D([0], [0], color='black', label=r"$c_{tot} = 120$")                  
                    ]

  plt.legend(handles=legend_elements, frameon=False, fontsize=16)
  sns.despine()
  plt.tight_layout()
  plt.show()
  plt.savefig('Figures/2/SVG/A.svg', dpi=300, transparent=True)
  plt.savefig('Figures/2/PNG/A.png', dpi=300, transparent=True)
  plt.close()


def make_2B(c, w, filo_index, spine_index):
  w_spine = w[spine_index]
  regr = LinearRegression()
  regr.fit(c[spine_index].reshape(-1, 1), w_spine.reshape(-1, 1))
  cw_slope = regr.coef_
  y_pred = regr.predict(c[spine_index].reshape(-1, 1))
  corr = r_regression(c[spine_index].reshape(-1, 1), w_spine.reshape(-1, 1))[0]
  plt.plot(np.linspace(0, 0.6),  regr.predict(np.linspace(0, 0.6).reshape(-1, 1)), color='red', linewidth=2)
  plt.text(s=r"$r = ${}".format(str(corr)[:4]), x=0.35, y=1, fontsize=20, color='red')

  plt.scatter(c[filo_index], w[filo_index], color='blue', s=1)
  plt.scatter(c[spine_index], w[spine_index],color='red', s=1)
  plt.xlabel(r"correlation $c_i$", fontsize=20)
  plt.xticks([0, 0.25, 0.5, 0.75], fontsize=18)
  plt.yticks([0, plasticity_params["w0_minus"], 1], [0,r"$w_0$", 1], fontsize=18)
  plt.ylabel(r"weight $w$", fontsize=20)
  sns.despine()
  plt.tight_layout()
  plt.show()
  plt.savefig('Figures/2/SVG/B.svg', dpi=300, transparent=True)
  plt.savefig('Figures/2/PNG/B.png', dpi=300, transparent=True)
  plt.close()


def make_2C(FR, y0):
  num_neurons = len(FR)
  plt.scatter(np.arange(num_neurons), np.roll(FR, int(num_neurons/2)), color='black', s=2)
  plt.xticks([0, 250, 500, 750, 1000], [r"$-\pi$", r"$-\pi/2$", r"$0$",  r"$\pi/2$",  r"$\pi$"], fontsize=18)
  plt.axhline(y0/Hz, xmax=0.5, linestyle='dashed', color='black')
  plt.text(s=r"$y_{pref}$", x=150, y=18, fontsize=20)
  plt.xlabel(r"Neuron ID", fontsize=20)
  plt.ylabel(R"postsyn. activity $y_\theta$ (Hz)", fontsize=20)
  plt.yticks([0, 5, 10, 15, 20, 25], fontsize=20)
  sns.despine()
  plt.tight_layout()
  plt.show()
  plt.savefig('Figures/2/SVG/C.svg', dpi=300, transparent=True)
  plt.savefig('Figures/2/PNG/C.png', dpi=300, transparent=True)
  plt.close()


def make_2D(w_trajs_FS, w_trajs_add, w_trajs_mult):
  titles = ['FS-STDP', 'add-STDP', 'mult-STDP']
  all_trajs = [w_trajs_FS, w_trajs_add, w_trajs_mult]
  fig, axs = plt.subplots(1, 3, sharey='row', figsize=(17, 7.5))
  for index, ax in enumerate(axs):
    ax.set_title(titles[index], fontsize=25)
    w_trajs = all_trajs[index]
    w = np.mean(w_trajs[:, 10:], axis=1)
    spine_index = np.where(w >= 0.5)[0]
    for w_traj in all_trajs[index]:
      ax.plot(w_traj, color='grey', linewidth=0.1, alpha=0.1)
    ax.plot(np.mean(w_trajs[spine_index], axis=0), color='grey', linewidth=5)
    ax.set_xticks([0, 200])
    ax.set_xlabel(r"time (s)", fontsize=20)
    ax.set_yticks([0, 1])
    ax.tick_params(axis='both', labelsize=18)       
    if index == 0:
      ax.set_ylabel(r"weight $w$", fontsize=20)
      sns.despine()
  plt.tight_layout()
  plt.savefig('Figures/2/SVG/D.svg', dpi=300, transparent=True)
  plt.savefig('Figures/2/PNG/D.png', dpi=300, transparent=True)
  plt.close()

def make_2E(w_trajs_FS, w_trajs_add, w_trajs_mult):
  all_trajs = [w_trajs_FS, w_trajs_add, w_trajs_mult]
  fig, axs = plt.subplots(1, 3, sharey='row', figsize=(23, 12))
  for index, ax in enumerate(axs):
    RFs = np.mean(all_trajs[index][:, -10:], axis=1)
    ax.scatter(np.arange(RFs.shape[0]), RFs, color='black', s=1)
    ax.set_xticks([0, 250, 500, 750, 1000], [r"$-\pi$", r"$-\pi/2$", r"$0$",  r"$\pi/2$",  r"$\pi$"], fontsize=18)
    ax.set_xlabel(r"Neuron ID", fontsize=20)
    ax.set_yticks([0, 1])
    ax.tick_params(axis='both', labelsize=18)       
    if index == 0:
      ax.set_ylabel(r"weight $w$", fontsize=20)
      sns.despine()
  plt.tight_layout()
  plt.savefig('Figures/2/SVG/E.svg', dpi=300, transparent=True)
  plt.savefig('Figures/2/PNG/E.png', dpi=300, transparent=True)
  plt.close()


def make_2F(data_FS, data_add, data_mult):
  n = 10
  r_FS = np.nan_to_num(np.array(data_FS["corr"]).reshape((n, n)))
  r_add = np.nan_to_num(np.array(data_add["corr"]).reshape((n, n)))
  r_mult = np.nan_to_num(np.array(data_mult["corr"]).reshape((n, n)))

  #all_r= [r_FS, r_FS - r_add, r_FS - r_mult, r_mult - r_add]
  #titles = [r"$r_{FS}$", r"$r_{FS}- r_{add}$", r"$r_{FS} - r_{mult}$", r"$r_{mult} - r_{add}$"]
  all_r= [r_FS, r_FS - r_add, r_FS - r_mult]
  #all_r= [r_FS, r_add, r_mult]
  titles = [r"$r_{FS}$", r"$r_{FS}- r_{add}$", r"$r_{FS} - r_{mult}$"]
  fig, axs = plt.subplots(1, 3, sharey='row', figsize=(15, 5))
  for index, ax in enumerate(axs):
    ax.set_title(titles[index], fontsize=20)
    im = ax.imshow(all_r[index], vmin=-1, vmax=1, origin='lower', cmap=get_rg_cmap())
    ax.set_xlabel(r"total correlation $c_{tot}$", fontsize=20)
    #ax.set_xticks([0, 9, 19, 29], [0, 40, 80, 120])
    #ax.set_yticks([0, 9, 19, 29], [1., 1.25, r"1.5", 1.75])
    ax.set_xticks([0, 3, 6, 9], [0, 40, 80, 120])
    ax.set_yticks([0, 3, 6, 9], [1., 1.25, r"1.5", 1.75])
    ax.tick_params(axis='both', labelsize=18)       
    if index == 0:
      ax.set_ylabel(r"pot./dep. imbalance $\alpha$", fontsize=20)
      sns.despine()
  fig.subplots_adjust(right=0.8)
  cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
  cb = fig.colorbar(im, cax=cbar_ax, ticks=[-1, 0, 1])
  cb.ax.tick_params(labelsize=16)
  plt.savefig('Figures/2/SVG/F.svg', dpi=300, transparent=True)
  plt.savefig('Figures/2/PNG/F.png', dpi=300, transparent=True)
  plt.close()


def make_2G(data_FS, data_add, data_mult):

  n = 10
  DI_FS = np.nan_to_num(np.array(data_FS["cw_slope"]).reshape((n, n)))
  DI_add = np.nan_to_num(np.array(data_add["cw_slope"]).reshape((n, n)))
  DI_mult = np.nan_to_num(np.array(data_mult["cw_slope"]).reshape((n, n)))


  #all_DI = [DI_FS, DI_FS - DI_add, DI_FS - DI_mult, DI_add - DI_mult]
  #titles = [r"$DI_{FS}$", r"$DI_{FS}- DI_{add}$", r"$DI_{FS} - DI_{mult}$", r"$DI_{add} - DI_{mult}$"]
  all_DI = [DI_FS, DI_FS - DI_add, DI_FS - DI_mult]
  #all_DI = [DI_FS, DI_add, DI_mult]
  titles = [r"$DI_{FS}$", r"$DI_{FS}- DI_{add}$", r"$DI_{FS} - DI_{mult}$"]
  fig, axs = plt.subplots(1, 3, sharey='row', figsize=(15, 5))
  for index, ax in enumerate(axs):
    ax.set_title(titles[index], fontsize=20)
    im = ax.imshow(all_DI[index], vmin=-1, vmax=1, origin='lower', cmap=get_rg_cmap())
    #im = ax.imshow(all_DI[index], origin='lower')
    #im = ax.imshow(all_DI[index], origin='lower')
    #plt.colorbar(im, ax)
    #plt.colorbar(im, ax)
    ax.set_xlabel(r"total correlation $c_{tot}$", fontsize=20)
    #ax.set_xticks([0, 9, 19, 29], [0, 40, 80, 120])
    #ax.set_yticks([0, 9, 19, 29], [1., 1.25, r"1.5", 1.75])
    ax.set_xticks([0, 3, 6, 9], [0, 40, 80, 120])
    ax.set_yticks([0, 3, 6, 9], [1., 1.25, r"1.5", 1.75])
    ax.tick_params(axis='both', labelsize=18)       
    if index == 0:
      ax.set_ylabel(r"pot./dep. imbalance $\alpha$", fontsize=20)
      sns.despine()

  fig.subplots_adjust(right=0.8)
  #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
  #fig.colorbar(im, cax=cbar_ax)
  plt.savefig('Figures/2/SVG/G.svg', dpi=300, transparent=True)
  plt.savefig('Figures/2/PNG/G.png', dpi=300, transparent=True)
  plt.close()
      




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
    neuron_params["g_exc_hat"] = 0.148*nsiemens
    neuron_params["g_inh_hat"] = 0.25*nsiemens
    neuron_params["tau_exc"] = 5*ms
    neuron_params["tau_inh"] = 5*ms

    #define learning parameters
    plasticity_params = {}
    plasticity_params["add"] = 0
    plasticity_params["mult"] = 0
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



    make_2A()

    with open('Data/figure_2B.pickle', 'rb') as handle:
      data_B = pickle.load(handle)
    globals().update(data_B)
    make_2B(c, w_FS, filo_index, spine_index)


    with open('Data/figure_2C.pickle', 'rb') as handle:
      data_C = pickle.load(handle)
    globals().update(data_C)
    make_2C(FR_FS, y_0)


    with open('Data/figure_2DE.pickle', 'rb') as handle:
      data_DE = pickle.load(handle)
    globals().update(data_DE)

    make_2D(w_trajs_FS, w_trajs_add, w_trajs_mult)
    make_2E(w_trajs_FS, w_trajs_add, w_trajs_mult)


    with open('Data/figure_2FG_FS.pickle', 'rb') as handle:
      data_FS = pickle.load(handle)

    with open('Data/figure_2FG_add.pickle', 'rb') as handle:
      data_add = pickle.load(handle)

    with open('Data/figure_2FG_mult.pickle', 'rb') as handle:
      data_mult = pickle.load(handle)


    make_2F(data_FS, data_add, data_mult)
    make_2G(data_FS, data_add, data_mult)
