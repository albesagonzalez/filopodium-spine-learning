import seaborn as sns

from brian2 import *

import matplotlib.cm as cm
import pickle

from aux import c_timed_array, get_zero_current


coop_colour = plt.cm.tab20(6)
comp_colour = plt.cm.tab20(0)
red = plt.cm.tab20(6)
orange = plt.cm.tab20(3)
green = plt.cm.tab20(4)
filo_colour = plt.cm.tab20(14)
spine_colour = plt.cm.tab20(8)
A_colour = spine_colour
B_colour = plt.cm.tab20(2)


plt.rcParams['figure.figsize'] = (6,5)


def f_plus_fp(w, alpha, w0_plus, q, a,  **kwargs):
    return np.abs((w0_plus - w - 1e-6))**((w+a)/q)

def f_minus_fp(w, alpha, w0_minus, q, a, **kwargs):
    return alpha*np.abs(w - w0_minus + 1e-6)**((w+a)/q)

def gutig_f_plus_fp(w, alpha, w0_plus, mu_plus, **kwargs):
    return np.abs((1 - w - 1e-6))**(mu_plus)

def gutig_f_minus_fp(w, alpha, w0_minus, q, mu_minus, **kwargs):
    return alpha*np.abs(w - 0 + 1e-6)**(mu_minus)

def add_subplot_axes(ax,rect, text, facecolor='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height],facecolor=facecolor)
    subax.set_xticks([0, 1])
    subax.set_yticks([])
    subax.set_ylim([-1, 1.1])
    props = dict(boxstyle='round', facecolor='white')
    subax.text(0.15, -0.6, text, fontsize=14, bbox=props)
    return subax

def make_1B():

  def kernel_filopodia(delta_t, w, alpha, mode):
    tau = 20*ms
    if mode == "ltp":
      return np.exp(-delta_t/tau)
    if mode == "ltd":
      return -alpha*np.exp(-delta_t/tau)

  def kernel_spine(delta_t, w, alpha, mode):
    tau = 20*ms
    w0 = 0.5
    q = 8
    a = 0.5
    mu = (w + a)/q
    if mode == 'ltp':
      return ((1 - w)**mu)*np.exp(-delta_t/tau)
    if mode == 'ltd':
      return -alpha*(np.abs(w - w0)**mu)*np.exp(-delta_t/tau)

  fig, axs = plt.subplots(2, 1, sharex='col', figsize=(6,7))
  delta_t = np.linspace(0, 50)*ms
  alpha_values = [1.25]
  for index, alpha in enumerate(alpha_values):
    axs[0].plot(-delta_t/ms, [kernel_filopodia(dt, w=0, alpha=alpha, mode="ltd") for dt in delta_t], color=filo_colour)
  axs[0].plot(delta_t/ms, [kernel_filopodia(dt, w=0, alpha=1, mode="ltp") for dt in delta_t], color=filo_colour)
  axs[0].set_title('filopodia-like kernel', fontsize=20, pad=-2)
  axs[0].set_yticks([-1, 0, 1], [-1, 0, 1], fontsize=18)
  axs[0].set_ylabel(r" ", fontsize=20)

  legend_elements = [Line2D([0], [0], color=spine_colour, label=r"$w = w_0 + 0.01$", alpha=0.2),
                   Line2D([0], [0], color=spine_colour, label=r"$w = w_0 + 0.25$", alpha=1/3+0.2),        
                   Line2D([0], [0], color=spine_colour, label=r"$w = 0.99$", alpha=2/3+0.2)]

  w_values = [0.51, 0.75, 0.99]
  for index, w in enumerate(w_values):
    axs[1].plot(-delta_t/ms, [kernel_spine(dt, w=w, alpha=1.25, mode="ltd") for dt in delta_t], color=spine_colour, alpha=index/3+0.3)
    axs[1].plot(delta_t/ms, [kernel_spine(dt, w=w, alpha=1.25, mode="ltp") for dt in delta_t], color=spine_colour, alpha=index/3+0.3)
  axs[1].set_title('spine-like kernel', fontsize=20, pad=-2)
  axs[1].set_xticks([-50, 0, 50],[-50, 0, 50], fontsize=18)
  axs[1].set_xlabel(r"time lag $t_{post} - t_{pre}$ (s)", fontsize=20)
  axs[1].set_yticks([-1, 0, 1], [-1, 0, 1], fontsize=18)
  axs[1].set_ylabel(r" ", fontsize=20)
  axs[1].legend(handles=legend_elements, frameon=False, fontsize=15)
  fig.text(0.025, 0.5, r"weight change $\Delta w$", va='center', rotation='vertical', fontsize=20)
  sns.despine()
  plt.savefig('Figures/1/PNG/B.png', dpi=300, transparent=True)
  plt.savefig('Figures/1/SVG/B.svg', dpi=300, transparent=True)


def make_1E(plasticity_params):
    fig = plt.figure(figsize=(6.5,5.5))

    axis = fig.add_subplot(1, 1, 1)
    a_values = np.linspace(0, 1, 3)
    w_values = np.linspace(0, 1, 1001)

    for index, a in enumerate(a_values):
        plasticity_params["a"] = a
        f_plus_values = f_plus_fp(w_values, **plasticity_params)
        f_minus_values = f_minus_fp(w_values, **plasticity_params)
        plt.plot(w_values, (f_minus_values - f_plus_values), color=comp_colour, linewidth=0.3*(index+1))
        plt.plot(w_values, f_plus_values, color=coop_colour, linewidth=0.6*(index+1))
        plt.axhline(0, color='grey', linestyle='--', alpha=0.5)
        plt.axvline(plasticity_params["w0_minus"], color='grey', linestyle='--', alpha=0.5)
        plt.xlabel(r"weight $w$", fontsize=20)
        plt.xticks([0, plasticity_params["w0_minus"], 1], [0, r"$w_0$", 1], fontsize=18)
        plt.ylim([0, 1])
        plt.xlim([0, 1])
        sns.despine()
        plt.yticks([-1, 0, 1], fontsize=18)


    subpos_add = [0.1,0.1,0.3,0.3]
    subpos_weight = [0.65,0.1,0.3,0.3]
    subax1 = add_subplot_axes(axis,subpos_add, text='add-STDP')
    subax2 = add_subplot_axes(axis,subpos_weight, text='mult-STDP')

    plasticity_params["mu_minus"] = 0
    plasticity_params["mu_plus"] = 0
    f_plus_values = gutig_f_plus_fp(w_values, **plasticity_params)
    f_minus_values = gutig_f_minus_fp(w_values, **plasticity_params)
    subax1.plot(w_values, f_minus_values - f_plus_values, color=comp_colour)
    subax1.plot(w_values, f_plus_values, color=coop_colour)
    subax1.axhline(0, color='grey', linestyle='--')

    plasticity_params["mu_minus"] = 0.1
    plasticity_params["mu_plus"] = 0.1
    f_plus_values = gutig_f_plus_fp(w_values, **plasticity_params)
    f_minus_values = gutig_f_minus_fp(w_values, **plasticity_params)
    subax2.plot(w_values, f_minus_values - f_plus_values, color=comp_colour)
    subax2.plot(w_values, f_plus_values, color=coop_colour)
    subax2.axhline(0, color='grey', linestyle='--')

    custom_lines = [Line2D([0], [0], color=comp_colour, lw=4),
                    Line2D([0], [0], color=coop_colour, lw=4)]
    plt.legend(custom_lines, [r"$\Delta f\;(w)$ (competition)", r"$f_+(w)$ (cooperation)"], fontsize=16, bbox_to_anchor=(0., -0.3, 0.1, 3), frameon=False)


    plt.savefig('Figures/1/PNG/E.png', dpi=300, transparent=True)
    plt.savefig('Figures/1/SVG/E.svg', dpi=300, transparent=True)



def make_G1(filo_index, spine_index, patterns):
    fig = plt.figure()
    plt.hist(patterns[0]["c"][filo_index], bins=np.linspace(0, 1, 50), color=filo_colour, label='filopodia', alpha=0.85)
    plt.hist(patterns[0]["c"][spine_index], bins=np.linspace(0, 1, 50), color=spine_colour, label='spines', alpha=0.85)
    plt.xlabel(r"correlation $c_i$", fontsize=20)
    plt.xticks([0, 1], fontsize=18)
    plt.yticks([])
    plt.legend(frameon=False, fontsize=18, loc=(0.6, 0.6))
    sns.despine()
    #plt.tight_layout()
    plt.savefig('Figures/1/PNG/G1.png', dpi=300, transparent=True)
    plt.savefig('Figures/1/SVG/G1.svg', dpi=300, transparent=True)



def make_G2(filo_index, spine_index, plasticity_params, w):
    fig = plt.figure()
    n, bins, patches = plt.hist(w[filo_index], bins=np.linspace(0, 1, 50), color=filo_colour)
    n, bins, patches = plt.hist(w[spine_index], bins=np.linspace(0, 1, 50), color=spine_colour)
    plt.xticks([0, plasticity_params["w0_minus"], 1], [0,r"$w_0$", 1], fontsize=18)
    plt.xlabel(r"weight $w$", fontsize=20)
    plt.yticks([])
    sns.despine()
    #plt.tight_layout()
    plt.savefig('Figures/1/PNG/G2.png', dpi=300, transparent=300)
    plt.savefig('Figures/1/SVG/G2.svg', dpi=300, transparent=300)


def make_G3(filo_index, spine_index, w, patterns):
    fig = plt.figure()
    plt.scatter(patterns[0]["c"][filo_index], w[filo_index], color=filo_colour, s=1)
    plt.scatter(patterns[0]["c"][spine_index], w[spine_index],color=spine_colour, s=1)
    plt.xlabel(r"correlation $c_i$", fontsize=20)
    plt.xticks([0, 1], fontsize=18)
    plt.yticks([0, plasticity_params["w0_minus"], 1], [0,r"$w_0$", 1], fontsize=18)
    plt.ylabel(r"weight $w$", fontsize=20)
    sns.despine()
    #plt.tight_layout()
    plt.savefig('Figures/1/PNG/G3.png', dpi=300, transparent=True)
    plt.savefig('Figures/1/SVG/G3.svg', dpi=300, transparent=True)


if __name__ == "__main__":
    
    #define diagrams plasticity params
    plasticity_params = {}
    plasticity_params["mu_plus"] = 0
    plasticity_params["mu_minus"] = 0
    plasticity_params["tau_mu"] = 10*second
    plasticity_params["mu_3"] = 1
    plasticity_params["tau_plus"] = 20*ms
    plasticity_params["tau_minus"] = 20*ms
    plasticity_params["w0_plus"] = 1
    plasticity_params["w0_minus"] = 0.5
    plasticity_params["lmbda"] = 0.006
    plasticity_params["alpha"] = 1.25
    plasticity_params["a"] = 0
    plasticity_params["q"] = 8

    make_1B()
    make_1E(plasticity_params)

    with open('Data/figure_1.pickle', 'rb') as handle:
      data = pickle.load(handle)

    patterns = data["patterns"]
    w = data["w"]
    filo_index = np.where(w < plasticity_params["w0_minus"])[0]
    spine_index = np.where(w >= plasticity_params["w0_minus"])[0]

    plt.rcParams['figure.figsize'] = (6,5)
    make_G1(filo_index, spine_index, patterns)
    make_G2(filo_index, spine_index, plasticity_params, w)
    make_G3(filo_index, spine_index, w, patterns)




