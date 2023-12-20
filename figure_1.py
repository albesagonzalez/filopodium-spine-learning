import seaborn as sns

from brian2 import *
import pickle

from aux import make_fig_dirs, get_q_a
make_fig_dirs(fig_num='1')

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


def f_plus_fp(w, mu, alpha, w0_plus, q, a,  **kwargs):
    return np.abs((w0_plus - w - 1e-6))**(mu)

def f_minus_fp(w, mu, alpha, w0_minus, q, a, **kwargs):
    #return alpha*np.abs(w - w0_minus + 1e-6)**((w+a)/q)
    return alpha*np.abs(w - w0_minus + 1e-6)**(mu)

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
    subax.set_yticks([0])
    subax.set_ylim([-1, 1.1])
    props = dict(boxstyle='round', facecolor='white')
    subax.text(0.15, -0.6, text, fontsize=14, bbox=props)
    return subax

def make_1B():

  def kernel_filopodia(delta_t, w, alpha, mode):
    tau = 20*ms
    mu = 0.01
    w0 = 0.5
    if mode == 'ltp':
      return ((1 - w)**mu)*np.exp(-delta_t/tau)
    if mode == 'ltd':
      return -alpha*(np.abs(w - w0)**mu)*np.exp(-delta_t/tau)

  def kernel_spine(delta_t, w, alpha, mode):
    tau = 20*ms
    mu = 0.1
    w0 = 0.5
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
    def mu_from_w(w):
      return (plasticity_params["a"] + w)/plasticity_params["q"]

    fig = plt.figure(figsize=(6.5,5.5))

    axis = fig.add_subplot(1, 1, 1)
    a = 0
    w_values = np.linspace(0, 1, 1001)
    mu_range_filo = [mu_from_w(0), mu_from_w(0.1)]
    mu_range_spine = [mu_from_w(0.5), mu_from_w(1)]


    plt.plot(w_values, f_minus_fp(w_values, mu_range_filo[0], **plasticity_params) - f_plus_fp(w_values, mu_range_filo[0], **plasticity_params), c=filo_colour, linestyle='dashed')
    plt.plot(w_values, f_minus_fp(w_values, mu_range_filo[1], **plasticity_params) - f_plus_fp(w_values, mu_range_filo[1], **plasticity_params), c=filo_colour, linestyle='dashed')
    plt.fill_between(w_values, f_minus_fp(w_values, mu_range_filo[0], **plasticity_params) - f_plus_fp(w_values, mu_range_filo[0], **plasticity_params), f_minus_fp(w_values, mu_range_filo[1], **plasticity_params) - f_plus_fp(w_values, mu_range_filo[1], **plasticity_params), facecolor=filo_colour, alpha=0.5)
    plt.plot(w_values, f_plus_fp(w_values, mu_range_filo[0], **plasticity_params), c=filo_colour)
    plt.plot(w_values, f_plus_fp(w_values, mu_range_filo[1], **plasticity_params), c=filo_colour)
    plt.fill_between(w_values, f_plus_fp(w_values, mu_range_filo[0], **plasticity_params), f_plus_fp(w_values, mu_range_filo[1], **plasticity_params), facecolor=filo_colour, alpha=0.5)
    plt.plot(w_values, f_minus_fp(w_values, mu_range_spine[0]*np.ones(w_values.shape), **plasticity_params) - f_plus_fp(w_values, mu_range_spine[0]*np.ones(w_values.shape), **plasticity_params), c=spine_colour, linestyle='dashed')
    plt.plot(w_values, f_minus_fp(w_values, mu_range_spine[1], **plasticity_params) - f_plus_fp(w_values, mu_range_spine[1], **plasticity_params), c=spine_colour, linestyle='dashed')
    plt.fill_between(w_values, f_plus_fp(w_values, mu_range_spine[0], **plasticity_params), f_plus_fp(w_values, mu_range_spine[1], **plasticity_params), facecolor=spine_colour, alpha=0.5)
    plt.plot(w_values, f_plus_fp(w_values, mu_range_spine[0], **plasticity_params), c=spine_colour)
    plt.plot(w_values, f_plus_fp(w_values, mu_range_spine[1], **plasticity_params), c=spine_colour)
    plt.fill_between(w_values, f_minus_fp(w_values, mu_range_spine[0], **plasticity_params) - f_plus_fp(w_values, mu_range_spine[0], **plasticity_params), f_minus_fp(w_values, mu_range_spine[1], **plasticity_params) - f_plus_fp(w_values, mu_range_spine[1], **plasticity_params), facecolor=spine_colour, alpha=0.5)

    plt.axhline(0, color='black', alpha=0.8)
    plt.axvline(plasticity_params["w0_minus"], color='black', alpha=0.8)
    plt.xlabel(r"weight $w$", fontsize=20)
    plt.xticks([0, plasticity_params["w0_minus"], 1], [0, r"$w_0$", 1], fontsize=18)
    plt.yticks([0], fontsize=18)
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    sns.despine()
    plt.yticks([-1, 0, 1], fontsize=18)

    subpos_add = [0.1,0.1,0.3,0.3]
    subpos_weight = [0.65,0.1,0.3,0.3]
    subax1 = add_subplot_axes(axis,subpos_add, text='add-STDP')
    subax2 = add_subplot_axes(axis,subpos_weight, text=r"nlta$^*$-STDP, $\mu = 0.1$")

    plasticity_params["mu_minus"] = 0
    plasticity_params["mu_plus"] = 0
    f_plus_values = gutig_f_plus_fp(w_values, **plasticity_params)
    f_minus_values = gutig_f_minus_fp(w_values, **plasticity_params)
    subax1.plot(w_values, f_minus_values - f_plus_values, color='black', linestyle='dashed')
    subax1.plot(w_values, f_plus_values, color='black')
    subax1.axhline(0, color='black', alpha=0.8)

    plasticity_params["mu_minus"] = 0.1
    plasticity_params["mu_plus"] = 0.1
    f_plus_values = gutig_f_plus_fp(w_values, **plasticity_params)
    f_minus_values = gutig_f_minus_fp(w_values, **plasticity_params)
    subax2.plot(w_values, f_minus_values - f_plus_values, color='black', linestyle='dashed')
    subax2.plot(w_values, f_plus_values, color='black')
    subax2.axhline(0, color='black', alpha=0.8)

    custom_lines = [Line2D([0], [0], color='black', lw=4, linestyle='dashed'),
                    Line2D([0], [0], color='black', lw=4)]
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


def make_G4(filo_index, spine_index,w_trajs, mu_trajs):
    fig, axs = plt.subplots(2, 1, sharex='col', figsize=(4,5))
    for w_traj, mu_traj in zip(w_trajs[filo_index], mu_trajs[filo_index]):
      axs[0].plot(w_traj, color=filo_colour, linewidth=0.1, alpha=0.05)
      axs[1].plot(mu_traj, color=filo_colour, linewidth=0.1, alpha=0.05)
    for w_traj, mu_traj in zip(w_trajs[spine_index], mu_trajs[spine_index]):
      axs[0].plot(w_traj, color=spine_colour, linewidth=0.1, alpha=0.4)
      axs[1].plot(mu_traj, color=spine_colour, linewidth=0.1, alpha=0.4)
    axs[0].axhline(0.1, linestyle='dashed', color=filo_colour)
    axs[0].axhline(0.75, linestyle='dashed', color=spine_colour)
    axs[0].plot(np.mean(w_trajs[spine_index], axis=0), color=spine_colour, linewidth=5)
    axs[0].plot(np.mean(w_trajs[filo_index], axis=0), color=filo_colour, linewidth=5)
    axs[1].axhline(0.01, linestyle='dashed', color=filo_colour)
    axs[1].axhline(0.1, linestyle='dashed', color=spine_colour)
    axs[0].text(410, 0.11, r"$w_{filo}$", fontsize=18, color=filo_colour)
    axs[0].text(410, 0.8, r"$w_{spine}$", fontsize=18, color=spine_colour)
    axs[1].text(410, 0.02, r"$\mu_{filo}$", fontsize=18, color=filo_colour)
    axs[1].text(410, 0.11, r"$\mu_{spine}$", fontsize=18, color=spine_colour)
    axs[1].plot(np.mean(mu_trajs[spine_index], axis=0), color=spine_colour, linewidth=5)
    axs[1].plot(np.mean(mu_trajs[filo_index], axis=0), color=filo_colour, linewidth=5)

    axs[0].set_yticks([0, 1],[0, 1], fontsize=18)
    axs[0].set_ylabel(r"weight $w$", fontsize=20)
    axs[1].set_ylabel(r"FS parameter $\mu$", fontsize=20)
    axs[1].set_xticks([0, 200, 400], [0, 200, 400], fontsize=18)
    axs[1].set_xlabel(r"time (s)", fontsize=20)
    axs[1].set_yticks([0, 0.1, 0.2], [0, 0.1, 0.2], fontsize=18)

    sns.despine()
    #plt.tight_layout()
    plt.savefig('Figures/1/PNG/G4.png', dpi=300, transparent=True)
    plt.savefig('Figures/1/SVG/G4.svg', dpi=300, transparent=True)




def make_supp123(w_trajs_add, fp_add, fm_add, competition_add, cooperation_add, filo_index_add, spine_index_add, w_trajs_nlta, fp_nlta, fm_nlta, competition_nlta, cooperation_nlta, filo_index_nlta, spine_index_nlta, w_trajs_FS, fp_FS, fm_FS, competition_FS, cooperation_FS, filo_index_FS, spine_index_FS):
  fig, axs = plt.subplots(3, 3,  sharey='row', sharex='col', figsize=(10,9))
  axs[0, 0].set_title("add-STDP", fontsize=20)
  for w_traj in w_trajs_add[filo_index_add]:
    axs[0, 0].plot(w_traj, color=filo_colour, linewidth=0.1, alpha=0.1)
  for w_traj in w_trajs_add[spine_index_add]:
    axs[0, 0].plot(w_traj, color=spine_colour, linewidth=0.1, alpha=0.1)
  axs[0, 0].plot(np.mean(w_trajs_add[spine_index_add], axis=0), color=A_colour, linewidth=5)
  axs[0, 0].plot(np.mean(w_trajs_add[filo_index_add], axis=0), color=filo_colour, linewidth=5)
  axs[0, 0].set_ylabel(r"weight $w$", fontsize=20)
  axs[0, 1].set_title(r"nlta$^*$-STDP ($\mu = 0.1$)", fontsize=20)
  for w_traj in w_trajs_nlta[filo_index_nlta]:
    axs[0, 1].plot(w_traj, color=filo_colour, linewidth=0.1, alpha=0.1)
  for w_traj in w_trajs_nlta[spine_index_nlta]:
    axs[0, 1].plot(w_traj, color=spine_colour, linewidth=0.1, alpha=0.1)
  axs[0, 1].plot(np.mean(w_trajs_nlta[spine_index_nlta], axis=0), color=A_colour, linewidth=5)
  axs[0, 1].plot(np.mean(w_trajs_nlta[filo_index_nlta], axis=0), color=filo_colour, linewidth=5)
  axs[0, 2].set_title(r"FS-STDP", fontsize=20)
  for w_traj in w_trajs_FS[filo_index_FS]:
    axs[0, 2].plot(w_traj, color=filo_colour, linewidth=0.1, alpha=0.1)
  for w_traj in w_trajs_FS[spine_index_FS]:
    axs[0, 2].plot(w_traj, color=spine_colour, linewidth=0.1, alpha=0.1)
  axs[0, 2].plot(np.mean(w_trajs_FS[spine_index_FS], axis=0), color=A_colour, linewidth=5)
  axs[0, 2].plot(np.mean(w_trajs_FS[filo_index_FS], axis=0), color=filo_colour, linewidth=5)
  axs[1, 0].plot(np.mean(competition_add[filo_index_add], axis=0), color=filo_colour, linestyle='dashed')
  axs[1, 0].plot(np.mean(cooperation_add[filo_index_add], axis=0), color=filo_colour)
  axs[1, 0].set_ylabel("mean-field\ninteractions", fontsize=20)
  #axs[1, 0].plot(np.mean(competition_add[spine_index_add], axis=0), color=spine_colour, linestyle='dashed', alpha=0.3)
  #axs[1, 0].plot(np.mean(cooperation_add[spine_index_add], axis=0), color=spine_colour, alpha=0.3)
  #axs[1, 1].plot(np.mean(competition_nlta[filo_index_nlta], axis=0),color =filo_colour, linestyle='dashed', alpha=0.3)
  #axs[1, 1].plot(np.mean(cooperation_nlta[filo_index_nlta], axis=0),color =filo_colour, alpha=0.3)
  axs[1, 1].plot(np.mean(competition_nlta[spine_index_nlta], axis=0), color=spine_colour, linestyle='dashed')
  axs[1, 1].plot(np.mean(cooperation_nlta[spine_index_nlta], axis=0), color=spine_colour)
  axs[1, 2].plot(np.mean(competition_FS[filo_index_FS], axis=0),color =filo_colour, linestyle='dashed')
  axs[1, 2].plot(np.mean(cooperation_FS[filo_index_FS], axis=0),color =filo_colour)
  axs[1, 2].plot(np.mean(competition_FS[spine_index_FS], axis=0), color=spine_colour, linestyle='dashed')
  axs[1, 2].plot(np.mean(cooperation_FS[spine_index_FS], axis=0), color=spine_colour)
  axs[1, 0].set_ylim([-10, None])
  #axs[1, 0].plot(np.mean(fp_add[spine_index_add], axis=0), color=spine_colour)
  #axs[1, 0].plot(np.mean(fm_add[spine_index_add] - fp_FS[spine_index_add], axis=0), color=spine_colour, linestyle='dashed')
  axs[2, 0].set_xlabel(r"time (s)", fontsize=20)
  axs[2, 0].plot(np.mean(fp_add[filo_index_add], axis=0), color=filo_colour)
  axs[2, 0].plot(np.mean(fm_add[filo_index_add] - fp_FS[filo_index_add], axis=0), color=filo_colour, linestyle='dashed')
  axs[2, 0].set_ylabel("competition/\ncooperation", fontsize=20)
  axs[2, 1].plot(np.mean(fp_nlta[spine_index_nlta], axis=0), color=spine_colour)
  axs[2, 1].plot(np.mean(fm_nlta[spine_index_nlta] - fp_FS[spine_index_nlta], axis=0), color=spine_colour, linestyle='dashed')
  #axs[2, 1].plot(np.mean(fp_nlta[filo_index_nlta], axis=0), color=filo_colour, alpha=0.3)
  #axs[2, 1].plot(np.mean(fm_nlta[filo_index_nlta] - fp_FS[filo_index_nlta], axis=0), color=filo_colour, linestyle='dashed', alpha=0.3)
  axs[2, 1].set_xlabel(r"time (s)", fontsize=20)
  axs[2, 2].plot(np.mean(fp_FS[spine_index_FS], axis=0), color=spine_colour)
  axs[2, 2].plot(np.mean(fm_FS[spine_index_FS] - fp_FS[spine_index_FS], axis=0),color=spine_colour, linestyle='dashed')
  axs[2, 2].plot(np.mean(fp_FS[filo_index_FS], axis=0), color=filo_colour)
  axs[2, 2].plot(np.mean(fm_FS[filo_index_FS] - fp_FS[filo_index_FS], axis=0), color=filo_colour, linestyle='dashed')
  axs[2, 2].set_xlabel(r"time (s)", fontsize=20)
  custom_lines = [Line2D([0], [0], color='black', lw=4, linestyle='dashed'),
                    Line2D([0], [0], color='black', lw=4)]
  plt.legend(custom_lines, [r"$\Delta f\;(w_i)\sum w_j$", r"$f_+(w_i)\sum C_{ij}^+w_j$"], fontsize=16, bbox_to_anchor=(0., -0.3, 0.1, 3), frameon=False)
  sns.despine()
  plt.savefig('Figures/supp/SVG/123.svg', dpi=300, transparent=True)
  plt.savefig('Figures/supp/PNG/123.png', dpi=300, transparent=True)
  plt.close()





if __name__ == "__main__":
    
    #define diagrams plasticity params
    plasticity_params = {}
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
    plasticity_params["mu_filo"] = 0.01
    plasticity_params["mu_spine"] = 0.1
    plasticity_params["q"], plasticity_params["a"] = get_q_a(plasticity_params)
    

    make_1B()
    make_1E(plasticity_params)

    #with open('Data/figure_1.pickle', 'rb') as handle:
    #with open('Data/figure_1_gaussian.pickle', 'rb') as handle:
    #with open('Data/figure_1_squared.pickle', 'rb') as handle:
    with open('Data/figure_1_von_mises.pickle', 'rb') as handle:
      data = pickle.load(handle)
    globals().update(data)


    plt.rcParams['figure.figsize'] = (4,5)
    w_FS = np.mean(w_trajs_FS[:, -10:], axis=1)
    make_G1(filo_index_FS, spine_index_FS, patterns)
    make_G2(filo_index_FS, spine_index_FS, plasticity_params, w_FS)
    make_G3(filo_index_FS, spine_index_FS, w_FS, patterns)
    make_G4(filo_index_FS, spine_index_FS,w_trajs_FS, mu_trajs_FS)


    make_supp123(w_trajs_add, fp_add, fm_add, competition_add, cooperation_add, filo_index_add, spine_index_add, w_trajs_nlta, fp_nlta, fm_nlta, competition_nlta, cooperation_nlta, filo_index_nlta, spine_index_nlta, w_trajs_FS, fp_FS, fm_FS, competition_FS, cooperation_FS, filo_index_FS, spine_index_FS)





