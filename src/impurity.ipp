#include "impurity.hpp"

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::define_parameters(parameters_type &parameters) {
  Base::define_parameters(parameters);

  //alps::define_convenience_parameters(parameters);
  parameters
      .description("Continous-time hybridization expansion impurity solver")
      .define < unsigned
  long > ("timelimit", "Total simulation time (in units of second)")
      .define<double>("thermalization_time",
                      -1,
                      "Thermalization time (in units of second). The default value is 10 % of timelimit.")
      .define<std::string>("outputfile",
                           alps::fs::remove_extensions(origin_name(parameters)) + ".out.h5",
                           "name of the output file")
      .define<int>("verbose", 0, "Verbose output for a non-zero value")
      .define<int>("sliding_window.max", 1000, "Max number of windows")
      .define<int>("sliding_window.min", 1, "Min number of windows")
          //Model definition
      .define<int>("model.sites", "Number of sites/orbitals")
      .define<int>("model.spins", "Number of spins")
      .define<double>("model.beta", "Inverse temperature")
      .define<int>("model.n_tau_hyb",
                   "Hybridization function is defined on a uniform mesh of N_TAU + 1 imaginary points.")
          //Updates
      .define<int>("update.multi_pair_ins_rem", 2, "Perform 1, 2, ..., k-pair updates.")
      .define<int>("update.n_global_updates", 10, "Global updates are performed every N_GLOBAL_UPDATES updates.")
      .define<std::string>("update.swap_vector", "", "Definition of global flavor-exchange updates.")
      .define<int>("update.single_operator_shift", 1, "Perform shifts of a single operator if a non-zero value is specified.")
      .define<int>("update.operator_pair_flavor_update", 1, "Perform changes of flavors of a pair of operators if a non-zero value is specified.")
      .define<int>("update.rebuild_inverse_matrix", 10, "Inverse of inverse matrix is rebuild from scratch to avoid accumulation of numerical errors. This specifies the interval in units of MC steps.")
          //Measurement
      .define<int>("measurement.n_non_worm_meas",
                   10,
                   "Non-worm measurements are performed every N_NON_WORM_MEAS updates.")
          //Single-particle GF
      .define<int>(           "measurement.G1.n_legendre", 100, "Number of legendre polynomials for measuring G(tau)")
      .define<int>(           "measurement.G1.n_tau", 2000, "G(tau) is computed on a uniform mesh of measurement.G1.n_tau + 1 points.")
      .define<int>(           "measurement.G1.n_matsubara", 2000, "G(i omega_n) is computed on a uniform mesh of measurement.G1.n_matsubara frequencies.")
      .define<int>(           "measurement.G1.max_matrix_size", 100000, "Max size of inverse matrix for measurement.")
      .define<int>(           "measurement.G1.max_num_data_accumulated", 10, "Number of measurements before accumulated data are passed to ALPS library.")
      .define<double>(        "measurement.G1.aux_field", 1.0, "Auxiliary field for avoiding a singular matrix")
          //Equal-time single-particle GF
      .define<int>("measurement.equal_time_G1.on", 0, "Set a non-zero value to activate measurement.")
          //Two-particle GF
      .define<double>(        "measurement.G2.aux_field", 1.0, "Auxiliary field for avoiding a singular matrix")
      .define<int>(           "measurement.G2.matsubara.on", 0, "Set a non-zero value to activate Matsubara measurement of G2.")
      .define<std::string>(   "measurement.G2.matsubara.frequencies_PH", "", "Text file containing a list of frequencies on which G2 is measured (in particle-hole convention)")
      .define<int>(           "measurement.G2.matsubara.max_matrix_size", 20, "Max size of inverse matrix for measurement.")
      .define<int>(           "measurement.G2.legendre.on", 0, "Set a non-zero value to activate Legendre measurement of G2.")
      .define<int>(           "measurement.G2.legendre.n_legendre", 0, "Number of legendre polynomials for measurement")
      .define<int>(           "measurement.G2.legendre.n_bosonic_freq", 20, "Number of bosonic frequencies for measurement")
      .define<int>(           "measurement.G2.legendre.max_matrix_size", 5, "Max size of inverse matrix for measurement.")
      .define<int>(           "measurement.G2.legendre.max_num_data_accumulated", 1, "Number of measurements before accumulated data are passed to ALPS library.")
          //Two-time two-particle GF
      .define<int>("measurement.two_time_G2.on", 0, "Set a non-zero value to activate measurement.")
      .define<int>("measurement.two_time_G2.n_legendre",
                   50,
                   "Number of legendre coefficients for measuring two-time two-particle Green's function.")
          //
          //Equal-time two-particle GF
      .define<int>("measurement.equal_time_G2.on", 0, "Set a non-zero value to activate measurement.")
          //
          //Density-density correlations
      .define<std::string>("measurement.nn_corr.def",
                           "",
                           "Input file for definition of density-density correlation functions")
      .define<int>("measurement.nn_corr.n_tau",
                   0,
                   "Number of imaginary time points for measurement (tau=0, ...., beta/2)")
      .define<int>("measurement.nn_corr.n_def",
                   0,
                   "Number of density-density correlation functions")
      .define<int>("measurement.max_order_histogram",
                   1000,
                   "Expansion order (per flavor) up to which histogram is measured.");

  IMP_MODEL::define_parameters(parameters);
}


template<typename IMP_MODEL>
HybridizationSimulation<IMP_MODEL>::HybridizationSimulation(parameters_type const &p, int rank)
    : alps::mcbase(p, rank),
      par(p),
      BETA(parameters["model.beta"]),      //inverse temperature
      SITES(parameters["model.sites"]),          //number of sites
      SPINS(parameters["model.spins"]),          //number of spins
      FLAVORS(SPINS * SITES),                             //flavors, i.e. #spins * #sites
      N(parameters["model.n_tau_hyb"]),                  //time slices
      N_non_worm_meas(parameters["measurement.n_non_worm_meas"]),
      thermalization_time(parameters["thermalization_time"]),
      start_time(time(NULL)),
      p_model(new IMP_MODEL(p, rank == 0)),//impurity model
      F(new HybridizationFunction<SCALAR>(
          BETA,
          parameters["model.delta_input_file"].template as<std::string>(),
          N, FLAVORS
        )
      ),
#ifdef ALPS_HAVE_MPI
      comm(),
#endif
      N_win_standard(1),
      sweeps(0),                                                                 //sweeps done up to now
      mc_config(F),
      config_space_extra_weight(0),
      worm_space_extra_weight_map(),
      operator_pair_flavor_updater(FLAVORS),
      single_op_shift_updater(BETA, FLAVORS, N),
      worm_insertion_removers(0),
      sliding_window(1, p_model, BETA),
      global_shift_acc_rate(),
      swap_acc_rate(0),
      timings(5, 0.0),
      verbose(p["verbose"].template as<int>() != 0),
      thermalized(false),
      pert_order_recorder(),
      config_spaces_visited_in_measurement_steps(0)
{

  if (thermalization_time < 0) {
    thermalization_time = static_cast<double>(0.1 * parameters["timelimit"].template as<double>());
  }
  if (thermalization_time > 0.9 * parameters["timelimit"].template as<double>()) {
    throw std::runtime_error("timelimit is too short in comparison with thermalization_time.");
  }


  /////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////
  ////Vectors Initialization Part//////////////////////////////////////
  ////Resize Vectors and Matrices so they have the right size./////////
  /////////////////////////////////////////////////////////////////////
  resize_vectors();

  /////////////////////////////////////////////////////////////////////
  ////Initialize Monte Carlo configuration  ///////////////////////////
  /////////////////////////////////////////////////////////////////////
  if (p["sliding_window.max"].template as<int>() < 1) {
    throw std::runtime_error("sliding_window.max cannot be smaller than 1.");
  }
  if (p["sliding_window.max"].template as<int>() < p["sliding_window.max"].template as<int>()) {
    throw std::runtime_error("sliding_window.max cannot be smaller than sliding_window.max.");
  }
  sliding_window.set_uniform_mesh(2*p["sliding_window.min"].template as<int>());
  mc_config.trace = sliding_window.compute_trace();
  if (comm.rank() == 0 && verbose) {
    std::cout << "initial trace = " << mc_config.trace
              << " with N_SECTION = " << sliding_window.get_n_section()
              << std::endl;
  }

  //Equal-time two-particle Green's function
  //read_eq_time_two_particle_greens_meas();

  //Two-time correlation functions
  //read_two_time_correlation_functions();

  if (comm.rank() == 0 && verbose) {
    std::cout << "The number of blocks in the inverse matrix is " << mc_config.M.num_blocks() << "." << std::endl;
    for (int block = 0; block < mc_config.M.num_blocks(); ++block) {
      std::cout << "flavors in block " << block << " : ";
      for (int flavor = 0; flavor < mc_config.M.num_flavors(block); ++flavor) {
        std::cout << mc_config.M.flavors(block)[flavor] << " ";
      }
      std::cout << std::endl;
    }
  }

  const int rank_ins_rem = par["update.multi_pair_ins_rem"].template as<int>();
  if (rank_ins_rem < 1) {
    throw std::runtime_error("update.multi_pair_ins_rem is not valid.");
  }
  for (int k = 1; k < rank_ins_rem + 1; ++k) {
    typedef InsertionRemovalUpdater<SCALAR, EXTENDED_SCALAR, SW_TYPE> TypeOffDiag;
    //typedef InsertionRemovalDiagonalUpdater<SCALAR, EXTENDED_SCALAR, SW_TYPE> TypeDiag;
    ins_rem_updater.push_back(
        boost::shared_ptr<TypeOffDiag>(
            new TypeOffDiag(k, FLAVORS)
        )
    );
  }

  create_worm_updaters();

  create_observables();
}


template<typename IMP_MODEL>
bool HybridizationSimulation<IMP_MODEL>::is_thermalized() const {
  return thermalized;
}

template<typename IMP_MODEL>
double HybridizationSimulation<IMP_MODEL>::fraction_completed() const {
  return 0.0;
}

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::update() {
  //////////////////////////////////
  // Monte Carlo updates
  //////////////////////////////////
  for (int imeas = 0; imeas < N_non_worm_meas; imeas++) {    // accumulate measurements from N_non_worm_meas updates before storing
    sweeps++;

    pert_order_recorder << mc_config.pert_order();

    std::vector<std::chrono::high_resolution_clock::time_point> times;

    times.push_back(std::chrono::high_resolution_clock::now());

    /** one sweep of the window */
    do_one_sweep();

    times.push_back(std::chrono::high_resolution_clock::now());

    /** Perform global updates which might cost O(beta)
    Ex: flavor exchanges, global shift */
    if (sweeps % par["update.n_global_updates"].template as<int>() == 0) {
      global_updates();
    }

    /** update parameters for MC moves and window size */
    if (!is_thermalized()) {
      update_MC_parameters();
    }

    times.push_back(std::chrono::high_resolution_clock::now());

    if (is_thermalized()) {
      //std::cout << "Calling measure_every_step" << std::endl;
      measure_every_step();
    }

    times.push_back(std::chrono::high_resolution_clock::now());

    if (sweeps % par["update.rebuild_inverse_matrix"].template as<int>() == 0) {
      mc_config.M.rebuild_inverse_matrix();
    }

    times.push_back(std::chrono::high_resolution_clock::now());

    for(auto i=0; i<times.size()-1; ++i) {
      timings[i] += std::chrono::duration_cast<std::chrono::nanoseconds>(times[i+1] - times[i]).count();
    }

    sanity_check();
  }//loop up to N_non_worm_meas

//times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now()).count());
}

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::measure_every_step() {
  check_true(is_thermalized(), "Must be thermalized!");

  switch (mc_config.current_config_space()) {
    case Z_FUNCTION:
      measure_scalar_observable<SCALAR>(measurements, "kLkR",
                                        static_cast<double>(
                                          measure_kLkR(sliding_window.get_operators(), BETA,
                                                                         0.5 * BETA * random())) * mc_config.sign);
      measure_scalar_observable<SCALAR>(measurements,
                                        "k",
                                        static_cast<double>(sliding_window.get_operators().size()) * mc_config.sign);
      break;

    case G1:
      p_G1_legendre_meas->measure_via_hyb(mc_config, measurements, random, par["measurement.G1.max_matrix_size"],
                                 par["measurement.G1.aux_field"]
      );
      break;

    case G2:
      if (p_G2_meas) {
        p_G2_meas->measure_via_hyb(mc_config, random,
                                   par["measurement.G2.matsubara.max_matrix_size"],
                                   par["measurement.G2.aux_field"]
        );
      }
      if (p_G2_legendre_meas) {
        p_G2_legendre_meas->measure_via_hyb(mc_config, measurements, random,
                                   par["measurement.G2.legendre.max_matrix_size"],
                                   par["measurement.G2.aux_field"]
        );
      }
      break;

    default:
      throw std::runtime_error("Used unsupported worm");
  }

  if (worm_meas.find(mc_config.current_config_space()) != worm_meas.end()) {
    for (auto& ptr_m: worm_meas.at(mc_config.current_config_space())) {
      ptr_m->measure(mc_config, sliding_window, measurements);
    }
  }

  //measure configuration space volume
  num_steps_in_config_space[get_config_space_position(mc_config.current_config_space())] += 1.0;
}

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::measure() {
  assert(is_thermalized());
  auto start = std::chrono::high_resolution_clock::now();

  //Measure the volumes of the configuration spaces
  {
    measurements["Z_function_space_num_steps"] << num_steps_in_config_space[0];
    for (int w = 0; w < worm_types.size(); ++w) {
      measurements["worm_space_num_steps_" + get_config_space_name(worm_types[w])] << num_steps_in_config_space[w + 1];
    }

    num_steps_in_config_space /= config_space_extra_weight;
    measurements["Z_function_space_volume"] << num_steps_in_config_space[0];
    for (int w = 0; w < worm_types.size(); ++w) {
      measurements["worm_space_volume_" + get_config_space_name(worm_types[w])] << num_steps_in_config_space[w + 1];
    }

    std::fill(num_steps_in_config_space.begin(), num_steps_in_config_space.end(), 0.0);
  }

  //Acceptance rate
  {
    for (int k = 1; k < par["update.multi_pair_ins_rem"].template as<int>() + 1; ++k) {
      ins_rem_updater[k - 1]->measure_acc_rate(measurements);
    }
    single_op_shift_updater.measure_acc_rate(measurements);
    operator_pair_flavor_updater.measure_acc_rate(measurements);

    //worm updates
    for (int i = 0; i < worm_insertion_removers.size(); ++i) {
      worm_insertion_removers[i]->measure_acc_rate(measurements);
    }
    for (typename worm_updater_map_t::iterator it = worm_movers.begin(); it != worm_movers.end();
         ++it) {
      it->second->measure_acc_rate(measurements);
    }
    for (typename std::map<std::string, boost::shared_ptr<LocalUpdaterType> >::iterator
             it = specialized_updaters.begin(); it != specialized_updaters.end(); ++it) {
      it->second->measure_acc_rate(measurements);
    }
  }

  if (mc_config.current_config_space() == Z_FUNCTION) {
    assert(!mc_config.p_worm);
    measure_Z_function_space();
  }

  auto end = std::chrono::high_resolution_clock::now();
  timings.back() = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
  measurements["TimingsSecPerNMEAS"] << timings;
  std::fill(timings.begin(), timings.end(), 0.0);
}

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::measure_Z_function_space() {
  // measure the perturbation order
  {
    const std::vector<int> &order_creation_flavor = count_creation_operators(FLAVORS, mc_config);
    const int N_order = par["measurement.max_order_histogram"].template as<int>();
    for (int flavor = 0; flavor < FLAVORS; ++flavor) {
      std::vector<double> order_creation_meas(FLAVORS *N_order,
      0.0);
      if (order_creation_flavor[flavor] < N_order) {
        order_creation_meas[flavor * N_order + order_creation_flavor[flavor]] = 1.0;
      }
      measurements["order"] << order_creation_meas;
    }
    {
      std::vector<double> tmp;
      std::copy(order_creation_flavor.begin(), order_creation_flavor.end(), std::back_inserter(tmp));
      measurements["PerturbationOrderFlavors"] << tmp;
    }
  }

  single_op_shift_updater.measure_acc_rate(measurements);
  for (int k = 1; k < par["update.multi_pair_ins_rem"].template as<int>() + 1; ++k) {
    ins_rem_updater[k - 1]->measure_acc_rate(measurements);
  }

  operator_pair_flavor_updater.measure_acc_rate(measurements);

  //measure acceptance rate of global shift
  if (global_shift_acc_rate.has_samples()) {
    measurements["Acceptance_rate_global_shift"] << global_shift_acc_rate.compute_acceptance_rate();
    global_shift_acc_rate.reset();
  }

  //measure acceptance rate of swap update
  if (swap_acc_rate.size() > 0 && swap_acc_rate[0].has_samples()) {
    std::vector<double> acc_swap(swap_acc_rate.size());
    for (int iupdate = 0; iupdate < swap_acc_rate.size(); ++iupdate) {
      assert(swap_acc_rate[iupdate].has_samples());
      acc_swap[iupdate] = swap_acc_rate[iupdate].compute_acceptance_rate();
      swap_acc_rate[iupdate].reset();
    }
    measurements["Acceptance_rate_swap"] << acc_swap;
  }

  std::cout << "debug " << mycast<double>(mc_config.sign) << std::endl;
  measurements["Sign"] << mycast<double>(mc_config.sign);

}

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::do_one_sweep() {
  //Propose higher-order insertion/removal updates less frequently
  std::vector<double> proposal_rates;
  {
    double p = 1.0;
    for (int update_rank = 0; update_rank < par["update.multi_pair_ins_rem"].template as<int>(); ++update_rank) {
      proposal_rates.push_back(p);
      p *= 0.25;
    }
  }
  boost::random::discrete_distribution<> dist(proposal_rates);

  int rank_ins_rem = dist(random.engine()) + 1;
  int current_n_section = 2 * std::max(N_win_standard / rank_ins_rem, 1);
  if (current_n_section != sliding_window.get_n_section()) {
    sliding_window.set_uniform_mesh(current_n_section, 0, ITIME_LEFT);
  }

  const int num_move = std::max(current_n_section - 2, 1);
  for (int move = 0; move < num_move; ++move) {
    double pert_order_sum = 0;
    //insertion and removal of operators hybridized with the bath
    for (int update = 0; update < FLAVORS; ++update) {
      ins_rem_updater[rank_ins_rem - 1]->update(random, BETA, mc_config, sliding_window);
      std::cout << "sign after ins_rem_updater " << mc_config.sign << std::endl;
      pert_order_sum += mc_config.pert_order();
    }

    if (par["update.operator_pair_flavor_update"].template as<int>() != 0) {
      for (int update = 0; update < FLAVORS; ++update) {
        operator_pair_flavor_updater.update(random, BETA, mc_config, sliding_window);
        std::cout << "sign after operator_pair_flavor_updater " << mc_config.sign << std::endl;
      }
    }

    //shift move of operators hybridized with the bath
    if (par["update.single_operator_shift"].template as<int>() != 0) {
      for (int update = 0; update < FLAVORS * rank_ins_rem; ++update) {
        single_op_shift_updater.update(random, BETA, mc_config, sliding_window);
        std::cout << "sign after single_op_shift " << mc_config.sign << std::endl;
      }
    }

    if (is_thermalized()) {
      config_spaces_visited_in_measurement_steps[get_config_space_position(mc_config.current_config_space())] = true;
    }

    transition_between_config_spaces();

    sliding_window.move_window_to_next_position();
  }
  sanity_check();
}

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::transition_between_config_spaces() {
  //Worm insertion/removal
  if (worm_types.size() == 0) {
    return;
  }

  for (int update = 0; update < FLAVORS; ++update) {
    //worm insertion and removal
    if (mc_config.current_config_space() == Z_FUNCTION) {
      const int i_worm = static_cast<int>(random() * worm_insertion_removers.size());
      worm_insertion_removers[i_worm]->update(random, BETA, mc_config, sliding_window, worm_space_extra_weight_map);
    } else {
      const int i_worm = get_config_space_position(mc_config.current_config_space()) - 1;
      assert (i_worm >= 0);
      worm_insertion_removers[i_worm]->update(random, BETA, mc_config, sliding_window, worm_space_extra_weight_map);
    }
    adjust_worm_space_weight();

    //G1 worm insertion and removal by changing hybridization lines
    if (mc_config.current_config_space() == Z_FUNCTION || mc_config.current_config_space() == G1) {
      specialized_updaters["G1_ins_rem_hyb"]->update(random,
                                                     BETA,
                                                     mc_config,
                                                     sliding_window,
                                                     worm_space_extra_weight_map);
      std::cout << "sign after G1 " << mc_config.sign << " " << mc_config.current_config_space() << std::endl;
      adjust_worm_space_weight();
    }

    //G2 worm insertion and removal by changing hybridization lines
    if (specialized_updaters.find("G2_ins_rem_hyb") != specialized_updaters.end() &&
        (mc_config.current_config_space() == Z_FUNCTION || mc_config.current_config_space() == G2)) {
      specialized_updaters["G2_ins_rem_hyb"]->update(random,
                                                     BETA,
                                                     mc_config,
                                                     sliding_window,
                                                     worm_space_extra_weight_map);
      std::cout << "sign after G1_hyb " << mc_config.sign << " " << mc_config.current_config_space() << std::endl;
      adjust_worm_space_weight();
    }

    //worm move
    for (typename worm_updater_map_t::iterator it = worm_movers.begin(); it != worm_movers.end();
         ++it) {
      if (it->first == mc_config.current_config_space()) {
        it->second->update(random, BETA, mc_config, sliding_window, worm_space_extra_weight_map);
        std::cout << "sign after worm_move " << mc_config.sign << " " << mc_config.current_config_space() << std::endl;
      }
    }
  }
}

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::global_updates() {
  auto n_section_back = sliding_window.get_n_section();
  sliding_window.set_uniform_mesh(1,
    0 , //new_position_right_edge
    ITIME_LEFT, 
    1  //new_position_left_edge
  );

  //jump between configuration spaces without a window
  transition_between_config_spaces();

  std::vector<SCALAR> det_vec = mc_config.M.compute_determinant_as_product();

  //Swap flavors
  if (swap_vector.size() > 0) {
    //do updates randomly
    std::vector<int> execute_ordering;
    for (int i = 0; i < swap_vector.size(); ++i) {
      execute_ordering.push_back(i);
    }
    MyRandomNumberGenerator rnd(random);
    std::random_shuffle(execute_ordering.begin(), execute_ordering.end(), rnd);

    for (int itry = 0; itry < swap_vector.size(); ++itry) {
      const int iupdate = execute_ordering[itry];
      const bool accepted = global_update<SCALAR, EXTENDED_SCALAR>(random, BETA,
                                                                   mc_config,
                                                                   det_vec,
                                                                   sliding_window,
                                                                   FLAVORS,
                                                                   ExchangeFlavor(&swap_vector[iupdate].first[0]),
                                                                   WormExchangeFlavor(&swap_vector[iupdate].first[0]),
                                                                   std::max(N_win_standard, 10)
      );

      if (accepted) {
        swap_acc_rate[iupdate].accepted();
        sanity_check();
      } else {
        swap_acc_rate[iupdate].rejected();
        sanity_check();
      }
    }
  }

  //Shift operators to restore translational symmetry
  {
    const double shift = random() * BETA;
    const bool accepted = global_update<SCALAR, EXTENDED_SCALAR>(random, BETA,
                                                                 mc_config,
                                                                 det_vec,
                                                                 sliding_window,
                                                                 FLAVORS,
                                                                 OperatorShift(BETA, shift),
                                                                 WormShift(BETA, shift),
                                                                 std::max(N_win_standard, 10)
    );
    if (accepted) {
      global_shift_acc_rate.accepted();
      mc_config.check_nan();
    } else {
      global_shift_acc_rate.rejected();
      if (p_model->translationally_invariant()) {
        std::cerr << "Warning: a global shift is rejected!" << std::endl;
      }
    }
    sanity_check();
  }

  std::cout << "sign after global " << mc_config.sign << " " << mc_config.current_config_space() << std::endl;

  sliding_window.set_uniform_mesh(n_section_back, 0, ITIME_LEFT);
  sanity_check();
}

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::update_MC_parameters() {
  assert(!is_thermalized());
  if (is_thermalized()) {
    throw std::logic_error("called update_MC_parameters after thermalized");
  }

  //record expansion order
  static std::list<double> min_pert_order_hist;
  min_pert_order_hist.push_back(mc_config.pert_order());
  if (min_pert_order_hist.size() > 100) {
    min_pert_order_hist.pop_front();
  }
  const double min_expansion_order_ave =
      std::accumulate(min_pert_order_hist.begin(), min_pert_order_hist.end(), 0.0) / min_pert_order_hist.size();

  //new window size for single-pair insertion and removal update
  N_win_standard = static_cast<std::size_t>(
      std::max(
          par["sliding_window.min"].template as<int>(),
          std::min(
              static_cast<int>(std::ceil(min_expansion_order_ave / FLAVORS)),
              par["sliding_window.max"].template as<int>()
          )

      )
  );
  if (verbose && comm.rank() == 0 && sweeps % 10 == 0) {
    std::cout << " new window size = " << N_win_standard << " sweep = " << sweeps << " pert_order = "
              << mc_config.pert_order() << std::endl;
  }

  //Update parameters for single-operator shift updates
  single_op_shift_updater.update_parameters();
  for (int k = 1; k < par["update.multi_pair_ins_rem"].template as<int>() + 1; ++k) {
    ins_rem_updater[k - 1]->update_parameters();
  }
  for (int i = 0; i < worm_insertion_removers.size(); ++i) {
    worm_insertion_removers[i]->update_parameters();
  }
  for (typename worm_updater_map_t::iterator it = worm_movers.begin(); it != worm_movers.end();
       ++it) {
    it->second->update_parameters();
  }
  for (typename std::map<std::string, boost::shared_ptr<LocalUpdaterType> >::iterator
           it = specialized_updaters.begin(); it != specialized_updaters.end(); ++it) {
    it->second->update_parameters();
  }

}

/////////////////////////////////////////////////
// Something to be done before measurement
/////////////////////////////////////////////////
template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::prepare_for_measurement() {
  single_op_shift_updater.finalize_learning();
  for (int k = 1; k < par["update.multi_pair_ins_rem"].template as<int>() + 1; ++k) {
    ins_rem_updater[k - 1]->finalize_learning();
  }

  for (int w = 0; w < worm_types.size(); ++w) {
    worm_insertion_removers[w]->finalize_learning();
  }
  for (typename worm_updater_map_t::iterator it = worm_movers.begin(); it != worm_movers.end();
       ++it) {
    it->second->finalize_learning();
  }
  for (typename std::map<std::string, boost::shared_ptr<LocalUpdaterType> >::iterator it = specialized_updaters.begin();
       it != specialized_updaters.end(); ++it) {
    it->second->finalize_learning();
  }

  if (comm.rank() == 0) {
    std::cout << "Thermalization process done after " << sweeps << " steps." << std::endl;
    std::cout << "The number of segments for sliding window update is " << N_win_standard << "."
              << std::endl;
    std::cout << "Perturbation orders (averaged over processes) are the following:" << std::endl;
  }
  const std::vector<int> &order_creation_flavor = count_creation_operators(FLAVORS, mc_config);
  if (comm.rank() == 0) {
    for (int flavor = 0; flavor < FLAVORS; ++flavor) {
      std::cout << " flavor " << flavor << " " << order_creation_flavor[flavor] << std::endl;
    }
    std::cout << std::endl;
  }
  if (p_flat_histogram_config_space) {
    if (!p_flat_histogram_config_space->converged() && verbose) {
      std::cout <<
                boost::format(
                    "Warning: flat histogram is not yet obtained for MPI rank %1%. It may be safer to increase thermalization time!"
                ) % comm.rank() << std::endl;
    }
    p_flat_histogram_config_space->synchronize(comm);
    // Apply the new worm space weights
    config_space_extra_weight[0] = 1.0;
    for (int w = 0; w < worm_types.size(); ++w) {
      config_space_extra_weight[w + 1] = p_flat_histogram_config_space->weight_ratio(w + 1, 0);
      worm_space_extra_weight_map[worm_types[w]] = p_flat_histogram_config_space->weight_ratio(w + 1, 0);
    }
    p_flat_histogram_config_space->finish_learning(false);
  }
  measurements["Pert_order_start"] << pert_order_recorder.mean();

  if (verbose) {
    std::cout << std::endl << "Weight of configuration spaces for MPI rank " << comm.rank() << " : ";
    std::cout << " Z function space = " << config_space_extra_weight[0];
    for (int w = 0; w < worm_types.size(); ++w) {
      std::cout << " , " << get_config_space_name(worm_types[w]) << " = " << config_space_extra_weight[w + 1];
    }
    std::cout << std::endl;
  }
}

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::finish_measurement() {
  measurements["Pert_order_end"] << pert_order_recorder.mean();
  if (!is_thermalized()) {
    throw std::runtime_error("Thermalization process is not done.");
  }
  for (int i = 0; i < config_spaces_visited_in_measurement_steps.size(); ++i) {
    if (!config_spaces_visited_in_measurement_steps[i]) {
      throw std::runtime_error("Some configuration space was not visited in measurement steps. Thermalization time may be too short.");
    }
  }

  if (p_G2_meas) {
     p_G2_meas->finalize(par["outputfile"].template as<std::string>());
  }
}


template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::sanity_check() {
#ifndef NDEBUG
  mc_config.check_nan();
  mc_config.sanity_check(sliding_window);
#endif
}


template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::adjust_worm_space_weight() {
  if (!p_flat_histogram_config_space || is_thermalized()) return;

  //measure current configuration space
  //if (mc_config.current_config_space() == Z_FUNCTION) {
  //p_flat_histogram_config_space->measure(0);
  //} else {
  //const int i_worm = get_worm_position(mc_config.current_config_space());
  //p_flat_histogram_config_space->measure(i_worm + 1);
  //}
  p_flat_histogram_config_space->measure(get_config_space_position(mc_config.current_config_space()));

  // If the histogram is flat enough, make the modification factor smaller
  if (p_flat_histogram_config_space->flat_enough()) {
    p_flat_histogram_config_space->update_lambda(false);
  }

  // Apply the new worm space weights
  config_space_extra_weight[0] = 1.0;
  for (int w = 0; w < worm_types.size(); ++w) {
    config_space_extra_weight[w + 1] = p_flat_histogram_config_space->weight_ratio(w + 1, 0);
    worm_space_extra_weight_map[worm_types[w]] = p_flat_histogram_config_space->weight_ratio(w + 1, 0);
  }
}
