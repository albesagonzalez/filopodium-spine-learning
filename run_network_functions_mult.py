from brian2 import *
from brian2.importexport import DictImportExport



def run_FS_network(neuron_params, plasticity_params, simulation_params):
  start_scope()

  N_pre_exc = simulation_params["N_pre"]
  N_pre_inh = 0.2*simulation_params["N_pre"]
  N_post = simulation_params["N_post"]

  pre_exc_rate = simulation_params["r_pre"]
  pre_inh_rate = 10*Hz

  if "ref_times" in simulation_params:
    ref_times = simulation_params["ref_times"]
    indices = np.zeros(len(ref_times))
    Poisson_ref = SpikeGeneratorGroup(1, indices, ref_times)

  else:
    Poisson_ref = PoissonGroup(1, rates= pre_exc_rate)


  c = simulation_params["c"]
  eqs_pre_neurons = '''
  ref : 1
  r : Hz
  p = ref*(r*dt + sqrt(c(t, i))*(1 - r*dt)) + (1 - ref)*r*dt*(1 - sqrt(c(t, i))) : 1
  '''
  inputs_exc = NeuronGroup(N_pre_exc, eqs_pre_neurons, threshold='rand() < p')
  inputs_exc.ref = 0
  inputs_exc.r = pre_exc_rate
  inputs_exc.run_regularly('ref = 0', when='before_synapses')

  inputs_inh = PoissonGroup(N_pre_inh, rates= pre_inh_rate)

  eqs_post_neurons = '''
  dv/dt = (v_rest - v)/tau_m + exc_part*(E_exc - v) + inh_part*(E_inh - v): volt
  exc_part = g_exc_hat*g_exc/C_m : 1/second
  inh_part = g_inh_hat*g_inh/C_m : 1/second
  dg_exc/dt = (aux_exc - g_exc)/tau_exc : 1
  daux_exc/dt = -aux_exc/tau_exc : 1
  dg_inh/dt = (aux_inh - g_inh)/tau_inh : 1
  daux_inh/dt = -aux_inh/tau_inh : 1
  C_m : farad
  R_m : ohm
  tau_m : second
  v_thres : volt
  v_rest : volt
  E_exc : volt
  E_inh : volt
  g_exc_hat : siemens
  g_inh_hat : siemens
  tau_exc : second
  tau_inh : second
  '''
  neurons_post = NeuronGroup(N_post, eqs_post_neurons, threshold='v>v_thres', reset='v = v_rest', events={'compute_wsum': 'True'},
                        method='euler')
  neurons_post.v = -70*mV

  S_ref = Synapses(Poisson_ref, inputs_exc, on_pre='ref = 1')

  if plasticity_params["add"]:
    S_exc = Synapses(inputs_exc, neurons_post,
                '''w : 1
                    add : 1
                    mult : 1
                    q : 1
                    a : 1
                    tau_plus : second
                    tau_minus : second
                    tau_mu : second
                    w0_plus : 1
                    w0_minus : 1
                    lmbda : 1
                    alpha : 1
                    delta_plus = lmbda : 1
                    delta_minus = alpha*lmbda : 1
                    mu_3 : 1
                    z_3 = 1 : 1
                    mu_minus : 1
                    mu_plus : 1
                    dz_minus/dt = -z_minus / tau_minus : 1 (event-driven)
                    dz_plus/dt = -z_plus / tau_plus : 1 (event-driven)
                    ''',
                on_pre='''
                        aux_exc += w
                        z_plus += 1
                        w -= alpha * lmbda * z_minus
                        w = clip(w, 0, w0_plus)
                        ''',
                on_post='''
                        z_minus += 1
                        w += lmbda * z_3 * z_plus
                        w = clip(w, 0, w0_plus)
                        '''
                )
  elif plasticity_params["mult"]:
    S_exc = Synapses(inputs_exc, neurons_post,
                '''w : 1
                    add : 1
                    mult : 1
                    q : 1
                    a : 1
                    tau_plus : second
                    tau_minus : second
                    tau_mu : second
                    w0_plus : 1
                    w0_minus : 1
                    lmbda : 1
                    alpha : 1
                    delta_plus = lmbda : 1
                    delta_minus = alpha*lmbda : 1
                    mu_3 : 1
                    z_3 = 1 : 1
                    mu_minus : 1
                    mu_plus : 1
                    dz_minus/dt = -z_minus / tau_minus : 1 (event-driven)
                    dz_plus/dt = -z_plus / tau_plus : 1 (event-driven)
                    ''',
                on_pre='''
                        aux_exc += w
                        z_plus += 1
                        w -= (w - 0) * alpha * lmbda * z_minus
                        w = clip(w, 0, w0_plus)
                        ''',
                on_post='''
                        z_minus += 1
                        w += lmbda * z_3 * z_plus
                        w = clip(w, 0, w0_plus)
                        '''
                )
  else:
    S_exc = Synapses(inputs_exc, neurons_post,
                '''w : 1
                    add : 1
                    mult : 1
                    q : 1
                    a : 1
                    tau_plus : second
                    tau_minus : second
                    tau_mu : second
                    w0_plus : 1
                    w0_minus : 1
                    lmbda : 1
                    alpha : 1
                    delta_plus = lmbda : 1
                    delta_minus = alpha*lmbda : 1
                    mu_3 : 1
                    z_3 = 1 : 1
                    dmu_minus/dt = -(mu_minus - (w+a)/q)/tau_mu : 1 (event-driven)
                    dmu_plus/dt = -(mu_plus - (w+a)/q)/tau_mu : 1 (event-driven)
                    dz_minus/dt = -z_minus / tau_minus : 1 (event-driven)
                    dz_plus/dt = -z_plus / tau_plus : 1 (event-driven)
                    ''',
                on_pre='''
                        aux_exc += w
                        z_plus += 1
                        w -= abs(w - w0_minus)**mu_minus * alpha * lmbda * z_minus
                        w = clip(w, 0, w0_plus)
                        ''',
                on_post='''
                        z_minus += 1
                        w += abs(w0_plus - w)**mu_plus * lmbda * z_3**mu_3 * z_plus
                        w = clip(w, 0, w0_plus)
                        '''
                )

  S_inh = Synapses(inputs_inh, neurons_post,
              '''w : 1''',
              on_pre='''aux_inh += w'''
              )

  S_ref.connect()
  S_exc.connect()
  S_inh.connect()

  S_exc.w = simulation_params["w"]
  S_exc.mu_minus = 0
  S_exc.mu_plus = 0
  S_inh.w = 1

  DictImportExport.import_data(neurons_post, neuron_params)
  DictImportExport.import_data(S_exc, plasticity_params)

  spike_ref_mon = SpikeMonitor(Poisson_ref)
  spike_pre_mon = SpikeMonitor(inputs_exc)
  spike_post_mon = SpikeMonitor(neurons_post)
  w_mon = StateMonitor(S_exc, ['w', 'mu_minus'], record=np.arange(N_pre_exc), dt=simulation_params["w_recording_dt"])
  post_mon = StateMonitor(neurons_post, ['v', 'g_exc'], record=np.arange(N_post))


  defaultclock.dt = simulation_params["integration_dt"]

  run(simulation_params["total_time"], report='text')

  return  spike_ref_mon, spike_pre_mon, spike_post_mon, w_mon.w, w_mon.mu_minus, post_mon