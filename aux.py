from brian2 import *
from scipy.stats import vonmises
import os


def generate_spike_trains(N, r, time, input_dt, **kwargs):
    num_bins = int(time/input_dt)
    prob = r*input_dt
    spike_trains = np.zeros((N, num_bins))
    for bin in range(num_bins):
        rand = np.random.rand(N)
        spike_index = np.where(prob > rand)[0]
        spike_trains[spike_index, bin] = 1
    return spike_trains

def get_freq_train(spike_train, r, input_dt, c, **kwargs):
    prob_train = np.zeros((len(c), len(spike_train)))
    upsilon = r*input_dt + np.sqrt(c)*(1 - r*input_dt)
    phi = r*input_dt*(1 - np.sqrt(c))
    for bin, spike_train_value in enumerate(spike_train):
        prob_train[:, bin] = upsilon if spike_train_value==1 else phi

    return prob_train/input_dt

def create_input(patterns, simulation_params):
    pre_input = []
    class_input = []
    for pattern in patterns:
        common_spike_train = generate_spike_trains(N=1, r=simulation_params["r_pre"], time=pattern["duration"], **simulation_params)[0]
        pre_freq_trains = get_freq_train(common_spike_train, r=simulation_params["r_pre"], c=pattern["c"], **simulation_params)
        pre_input.append(pre_freq_trains)
    return TimedArray(np.transpose(np.concatenate(pre_input, axis=1))*Hz, dt=simulation_params["input_dt"])


def c_timed_array(patterns, simulation_params):
    c_values = []
    for lap, pattern in enumerate(patterns):
        num_input_steps = int(pattern["duration"]/simulation_params["input_dt"])
        c_pattern = np.tile(pattern["c"], (num_input_steps, 1))
        c_values.append(c_pattern)
    return TimedArray(np.concatenate(c_values, axis=0), dt=simulation_params["input_dt"])


def get_zero_current(simulation_params, I_0):
    num_bins = int(simulation_params["total_time"]/simulation_params["input_dt"])
    #I_ext = 150e-12*np.ones((num_bins, simulation_params["N_post"]))
    I_ext = I_0*np.ones((num_bins, simulation_params["N_post"]))
    return TimedArray(I_ext*amp, dt=simulation_params["input_dt"])


def get_dynamical_terms(w_trajs, mu_trajs, patterns, neuron_params, plasticity_params, simulation_params):
  def f_plus(w, mu, plasticity_params):
    if plasticity_params["add"] or plasticity_params["mlt"]:
      return np.ones(w.shape)
    else:
      return np.abs(plasticity_params["w0_plus"] - w)**mu

  def f_minus(w, mu, plasticity_params):
    if plasticity_params["add"]:
      return plasticity_params["alpha"]*np.ones(w.shape)
    elif plasticity_params["mlt"]:
      return plasticity_params["alpha"]*np.abs(w - plasticity_params["w0_minus"])
    else:
      return plasticity_params["alpha"]*np.abs(w - plasticity_params["w0_minus"])**mu
  competition = np.zeros(w_trajs.shape)
  cooperation = np.zeros(w_trajs.shape)
  for pattern in patterns:
    C_plus = np.zeros((simulation_params["N_pre"], simulation_params["N_pre"]))
    for i in range(simulation_params["N_pre"]):
      for j in range(simulation_params["N_pre"]):
        if i != j:
          C_plus[i, j] = np.sqrt(pattern["c"][i]*pattern["c"][j])
    np.fill_diagonal(C_plus, 1)
    C_plus = C_plus/(neuron_params["tau_m"]*simulation_params["r_pre"])
    start_index = int(pattern["start_time"]/simulation_params["w_recording_dt"])
    end_index = int((pattern["start_time"] + pattern["duration"])/simulation_params["w_recording_dt"])
    w, mu = w_trajs[:, start_index:end_index], mu_trajs[:, start_index:end_index]
    sum_w = np.sum(np.transpose(w), axis=1)
    fp_w = f_plus(w, mu, plasticity_params)
    fm_w = f_minus(w, mu, plasticity_params)
    competition[:, start_index:end_index] = (fm_w - fp_w)*sum_w
    cooperation[:, start_index:end_index] = fp_w*C_plus.dot(w)

  factor = plasticity_params["lmbda"]*neuron_params["tau_m"]*(simulation_params["r_pre"]**2)/simulation_params["N_pre"]
  return fp_w, fm_w, factor, competition, cooperation

def get_vm_corr(pref_deg, kappa, c_tot, bias=0.):
   x = np.linspace(pi + pref_deg, -pi + pref_deg, 1000)
   vm = vonmises(kappa=kappa)
   return vm.pdf(x)/np.sum(vm.pdf(x))*c_tot + bias


def make_data_dir():
  current_directory = os.getcwd()
  final_directory = os.path.join(current_directory, r'Data')
  if not os.path.exists(final_directory):
      os.makedirs(final_directory)

def make_fig_dirs(fig_num):
  current_directory = os.getcwd()
  final_directory = os.path.join(current_directory, r'Figures')
  if not os.path.exists(final_directory):
      os.makedirs(final_directory)
  final_directory = os.path.join(final_directory, fig_num)
  if not os.path.exists(final_directory):
      os.makedirs(final_directory)
  PNG_directory = os.path.join(final_directory, r'PNG')
  if not os.path.exists(PNG_directory):
      os.makedirs(PNG_directory)
  SVG_directory = os.path.join(final_directory, r'SVG')
  if not os.path.exists(SVG_directory):
      os.makedirs(SVG_directory)


