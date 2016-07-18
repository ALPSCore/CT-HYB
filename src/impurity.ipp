#include "impurity.hpp"

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::define_parameters(parameters_type &parameters) {
  Base::define_parameters(parameters);

  alps::define_convenience_parameters(parameters);
  parameters
      .description("Continous-time hybridization expansion impurity solver")
      .define<int>("SITES", "Number of sites/orbitals")
      .define<int>("SPINS", "Number of spins")
      .define<double>("BETA", "Inverse temperature")
      .define<int>("N_TAU", "Number of points (minus 1) for G(tau), number of Matsubara frequencies for G(i omega_n)")
      .define<int>("N_LEGENDRE_MEASUREMENT", 100, "Number of legendre coefficients for measuring G(tau)")
      .define<int>("N_LEGENDRE_N2_MEASUREMENT",
                   0,
                   "Number of legendre coefficients for measuring two-time correlation functions. Set 0 to deactivate the measurement.")
      .define<long>("SWEEPS", 1E+9, "Number of sweeps for total run")
      .define<long>("THERMALIZATION", 10, "Minimum number of sweeps for thermalization")
      .define<long>("MAX_THERMALIZATION_SWEEPS", 1E+9, "Maximimum number of Monte Carlo steps for thermalization")
      .define<int>("N_MEAS", 10, "Expensive measurements are performed every N_MEAS updates.")
      .define<int>("RANK_INSERTION_REMOVAL_UPDATE", 1, "1 for only single-pair update. k for up to k-pair update.")
      .define<int>("N_SWAP", 10, "We attempt to swap flavors every N_SWAP Monte Carlo steps.")
      .define<std::string>("SWAP_VECTOR",
                           "",
                           "Definition of global updates in which the flavors of creation and annihilation operators are exchanged. Refer to manual for details.")
      .define<double>("ACCEPTANCE_RATE_CUTOFF", 0.01, "cutoff for acceptance rate in sliding window update")
      .define<int>("MAX_N_SLIDING_WINDOW", 10000, "Maximum number of segments for sliding window update")
      .define<int>("MIN_N_SLIDING_WINDOW",
                   1,
                   "Mimimum number of segments for sliding window update. Please set a value larger than 1 for numerical stability at low T.")
      .define<int>("N_UPDATE_CUTOFF",
                   50,
                   "How many times the value of N_SLIDING_WINDOW is updated during thermalization.")
      .define<int>("Tmin", 1, "The scheduler checks longer than every Tmin seconds if the simulation is finished.")
      .define<int>("Tmax", 60, "The scheduler checks shorter than every Tmax seconds if the simulation is finished.")
      .define<int>("N_ORDER",
                   100,
                   "Histogram of expansion order for each flavor is measured up to an expansion order of N_ORDER")
      .define<int>("MAX_ORDER", 10000, "Sum of expansion orders of all flavors cannot go beyond this value")
      .define<int>("N_TAU_TWO_TIME_CORRELATION_FUNCTIONS",
                   0,
                   "Number of tau points for which two-time correlation functions are measured (tau=0, ...., beta/2)")
      .define<std::string>("TWO_TIME_CORRELATION_FUNCTIONS",
                           "",
                           "Input file for definition of two-time correlation functions to be measured")
      .define<int>("VERBOSE", 0, "If VERBOSE is not zero, more messages will be outputed.")
      .define<double>("WORM_SPACE_WEIGHT", 1.0, "Weight of worm space");

  IMP_MODEL::define_parameters(parameters);
}


template<typename IMP_MODEL>
HybridizationSimulation<IMP_MODEL>::HybridizationSimulation(parameters_type const &p, int rank)
    : alps::mcbase(p, rank),
      par(p),
      BETA(parameters["BETA"]),      //inverse temperature
      SITES(parameters["SITES"]),          //number of sites
      SPINS(parameters["SPINS"]),          //number of spins
      FLAVORS(SPINS * SITES),                             //flavors, i.e. #spins * #sites
      N(static_cast<int>(parameters["N_TAU"])),                  //time slices
      Np1(N + 1),
      N_meas(parameters["N_MEAS"]),
      N_swap(parameters["N_SWAP"]),
      total_sweeps(parameters["SWEEPS"]),                           //sweeps needed for total run
      p_model(new IMP_MODEL(p, rank == 0)),//impurity model
      F(new HybridizationFunction<SCALAR>(
          BETA, N, FLAVORS, p_model->get_F()
        )
      ),
#ifdef ALPS_HAVE_MPI
      comm(),
#endif
      thermalization_checker(parameters["THERMALIZATION"].template as<long>(),
                             parameters["MAX_THERMALIZATION_SWEEPS"].template as<long>()),          //minimum sweeps needed for thermalization
      N_win_standard(1),
      sweeps(0),                                                                 //sweeps done up to now
      mc_config(F),
      config_space_extra_weight(mc_config.num_config_spaces(), 1.0),
      operator_pair_flavor_updater(FLAVORS),
      single_op_shift_updater(BETA, FLAVORS, N),
      worm_movers(0),
      worm_insertion_removers(0),
      sliding_window(p_model.get(), BETA),
      g_meas_legendre(FLAVORS, p["N_LEGENDRE_MEASUREMENT"], N, BETA),
      //p_N2_meas(0),
      p_meas_corr(0),
      global_shift_acc_rate(),
      swap_acc_rate(0),
      num_steps_in_config_space(mc_config.num_config_spaces(), 0.0),
      timings(4, 0.0),
      verbose(p["VERBOSE"].template as<int>() != 0),
      thermalized(false) {
  /////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////
  ////Vectors Initialization Part//////////////////////////////////////
  ////Resize Vectors and Matrices so they have the right size./////////
  /////////////////////////////////////////////////////////////////////
  resize_vectors();

  /////////////////////////////////////////////////////////////////////
  ////Initialize Monte Carlo configuration  ///////////////////////////
  /////////////////////////////////////////////////////////////////////
  //if (p["N_SLIDING_WINDOW"].template as<int>() > p["MAX_N_SLIDING_WINDOW"].template as<int>()) {
  //throw std::runtime_error("N_SLIDING_WINDOW cannot be larger than MAX_N_SLIDING_WINDOW.");
  //}
  if (p["MAX_N_SLIDING_WINDOW"].template as<int>() < 1) {
    throw std::runtime_error("MAX_N_SLIDING_WINDOW cannot be smaller than 1.");
  }
  if (p["MAX_N_SLIDING_WINDOW"].template as<int>() < p["MIN_N_SLIDING_WINDOW"].template as<int>()) {
    throw std::runtime_error("MAX_N_SLIDING_WINDOW cannot be smaller than MIN_N_SLIDING_WINDOW.");
  }
  sliding_window.init_stacks(p["MIN_N_SLIDING_WINDOW"], mc_config.operators);
  mc_config.trace = sliding_window.compute_trace(mc_config.operators);
  if (global_mpi_rank == 0 && verbose) {
    std::cout << "initial trace = " << mc_config.trace << " with N_SLIDING_WINDOW = " << sliding_window.get_n_window()
        << std::endl;
  }

  //Equal-time two-particle Green's function
  read_eq_time_two_particle_greens_meas();

  //Two-time correlation functions
  read_two_time_correlation_functions();

  if (global_mpi_rank == 0 && verbose) {
    std::cout << "The number of blocks in the inverse matrix is " << mc_config.M.num_blocks() << "." << std::endl;
    for (int block = 0; block < mc_config.M.num_blocks(); ++block) {
      std::cout << "flavors in block " << block << " : ";
      for (int flavor = 0; flavor < mc_config.M.num_flavors(block); ++flavor) {
        std::cout << mc_config.M.flavors(block)[flavor] << " ";
      }
      std::cout << std::endl;
    }
  }

  const int rank_ins_rem = par["RANK_INSERTION_REMOVAL_UPDATE"].template as<int>();
  if (rank_ins_rem < 1) {
    throw std::runtime_error("RANK_INSERTION_REMOVAL_UPDATE is not valid.");
  }
  for (int k = 1; k < rank_ins_rem + 1; ++k) {
    typedef InsertionRemovalUpdater<SCALAR, EXTENDED_SCALAR, SW_TYPE> TypeOffDiag;
    typedef InsertionRemovalDiagonalUpdater<SCALAR, EXTENDED_SCALAR, SW_TYPE> TypeDiag;
    ins_rem_updater.push_back(
        boost::shared_ptr<TypeOffDiag>(
            new TypeOffDiag(k, FLAVORS)
        )
    );
    ins_rem_diagonal_updater.push_back(
        boost::shared_ptr<TypeDiag>(
            new TypeDiag(k, FLAVORS, BETA, N / 2)
        )
    );
  }

  create_worm_updaters();

  create_observables();
}


template<typename IMP_MODEL>
bool HybridizationSimulation<IMP_MODEL>::is_thermalized() const {
  return thermalization_checker.is_thermalized();
}

template<typename IMP_MODEL>
double HybridizationSimulation<IMP_MODEL>::fraction_completed() const {
  double work =
      (is_thermalized() ? (sweeps - thermalization_checker.get_actual_thermalization_steps()) / double(total_sweeps)
                        : 0.);
  if (work > 1.0) {
    work = 1.0;
  }
  return work;
}

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::update() {
#ifdef MEASURE_TIMING
  boost::timer::cpu_timer timer;
#endif

  //////////////////////////////////
  // Monte Carlo updates
  //////////////////////////////////
  for (int imeas = 0; imeas < N_meas; imeas++) {    // accumulate measurements from N_meas updates before storing
    sweeps++;

#ifdef MEASURE_TIMING
    const double time1 = timer.elapsed().wall * 1E-9;
#endif

    /** one sweep of the window */
    local_updates();

#ifdef MEASURE_TIMING
    const double time2 = timer.elapsed().wall * 1E-9;
    timings[0] += time2 - time1;
#endif

    //Perform global updates which might cost O(beta)
    //Ex: flavor exchanges, global shift
    global_updates();

    //update parameters for MC moves and window size
    if (!is_thermalized()) {
      update_MC_parameters();
    }

#ifdef MEASURE_TIMING
    const double time3 = timer.elapsed().wall * 1E-9;
    timings[1] += time3 - time2;
#endif

    if (is_thermalized()) {
      if (mc_config.current_config_space() == Z_FUNCTION_SPACE) {
        g_meas_legendre.measure(mc_config);
      }
      if (mc_config.current_config_space() == N2_SPACE) {
        p_N2_meas->measure_new(mc_config, measurements, random, sliding_window, "N2_correlation_function");
      }
      //measure configuration space volume
      num_steps_in_config_space[static_cast<int>(mc_config.current_config_space())] += 1.0;
    }

#ifdef MEASURE_TIMING
    const double time4 = timer.elapsed().wall * 1E-9;
    timings[2] += time4 - time3;
#endif

    sanity_check();
  }//loop up to N_meas
}

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::measure() {
  assert(is_thermalized());
#ifdef MEASURE_TIMING
  boost::timer::cpu_timer timer;
#endif

  //Measure the volumes of the configuration spaces
  {
    measurements["Z_function_space_num_steps"] << num_steps_in_config_space[0];
    for (int w = 0; w < worm_names.size(); ++w) {
      measurements["worm_space_num_steps_" + worm_names[w]] << num_steps_in_config_space[w + 1];
    }

    num_steps_in_config_space /= config_space_extra_weight;
    measurements["Z_function_space_volume"] << num_steps_in_config_space[0];
    for (int w = 0; w < worm_names.size(); ++w) {
      measurements["worm_space_volume_" + worm_names[w]] << num_steps_in_config_space[w + 1];
    }

    std::fill(num_steps_in_config_space.begin(), num_steps_in_config_space.end(), 0.0);
  }

  if (mc_config.current_config_space() == Z_FUNCTION_SPACE) {
    measure_Z_function_space();
  } else if (mc_config.current_config_space() == N2_SPACE) {
    //measure_N2_space();
  }

#ifdef MEASURE_TIMING
  timings[3] = timer.elapsed().wall * 1E-9;
  measurements["TimingsSecPerNMEAS"] << timings;
  std::fill(timings.begin(), timings.end(), 0.0);
#endif
}

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::measure_Z_function_space() {
  // measure the perturbation order
  {
    const std::vector<int> &order_creation_flavor = count_creation_operators(FLAVORS, mc_config);
    const int N_order = par["N_ORDER"].template as<int>();
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
  for (int k = 1; k < par["RANK_INSERTION_REMOVAL_UPDATE"].template as<int>() + 1; ++k) {
    ins_rem_diagonal_updater[k - 1]->measure_acc_rate(measurements);
  }

  //operator_pair_flavor_updater.measure_acc_rate(measurements);

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

  //Measure <n>
  measure_n();

  //Measure <n>
  measure_two_time_correlation_functions();

  //Measure Legendre coefficients of single-particle Green's function
  if (g_meas_legendre.has_samples()) {
    measure_simple_vector_observable<COMPLEX>(measurements, "Greens_legendre",
                                              to_std_vector(
                                                  g_meas_legendre.get_measured_legendre_coefficients(p_model->get_rotmat_Delta())
                                              )
    );
    measure_simple_vector_observable<COMPLEX>(measurements, "Greens_legendre_rotated",
                                              to_std_vector(
                                                  g_meas_legendre.get_measured_legendre_coefficients(Eigen::Matrix<
                                                      SCALAR,
                                                      Eigen::Dynamic,
                                                      Eigen::Dynamic>::Identity(FLAVORS, FLAVORS))
                                              )
    );
    g_meas_legendre.reset();
  }

  measurements["Sign"] << mycast<double>(mc_config.sign);

  //fidelity susceptibility
  /*
  {
      measure_scalar_observable<SCALAR>(measurements, "kLkR", static_cast<double>(measure_kLkR(operators, BETA,
                                                                                               0.5 * BETA *
                                                                                               random())) *
                                                              mc_config.sign);
      measure_scalar_observable<SCALAR>(measurements, "k", static_cast<double>(operators.size()) * mc_config.sign);
  }
  */

}

//Measure the expectation values of density operators
template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::measure_n() {
  assert(is_thermalized());
  MeasStaticObs<SlidingWindowManager<IMP_MODEL>, CdagC> meas(sliding_window, mc_config.operators);
  std::vector<CdagC> ops(FLAVORS);
  std::vector<EXTENDED_COMPLEX> result_meas(FLAVORS);
  for (int flavor = 0; flavor < FLAVORS; ++flavor) {
    boost::array<int, 2> flavors_tmp;
    flavors_tmp[0] = flavor;
    flavors_tmp[1] = flavor;
    ops[flavor] = CdagC(flavors_tmp);
  }

  //Measure <n>
  meas.perform_meas(ops, result_meas);

  //We measure only the real part because the Monte Carl average of a density operator should be real.
  // <n> = <n>_MC/<sign>_MC: <sign>_MC=real, <n>_MC=real, <n>=real
  //Note: we must take the real part of the quantity after it's multiplied by "sign/trace".
  std::vector<double> result_meas_Re(FLAVORS);
  EXTENDED_COMPLEX inv_trace = static_cast<EXTENDED_SCALAR>(EXTENDED_SCALAR(1.0) / mc_config.trace);
  for (int flavor = 0; flavor < FLAVORS; ++flavor) {
    result_meas_Re[flavor] =
        convert_to_scalar(static_cast<EXTENDED_REAL>(get_real(result_meas[flavor] * mc_config.sign * inv_trace)));
  }
  measurements["n"] << result_meas_Re;
}

//Measure two-time correlation functions by insertion
template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::measure_two_time_correlation_functions() {
  assert(is_thermalized());
  if (p_meas_corr.get() == 0) {
    return;
  }

  boost::multi_array<EXTENDED_COMPLEX, 2> result;
  p_meas_corr->perform_meas(sliding_window, mc_config.operators, result);
  const EXTENDED_COMPLEX coeff = EXTENDED_COMPLEX(mc_config.sign) / EXTENDED_COMPLEX(mc_config.trace);
  std::transform(result.origin(), result.origin() + result.num_elements(), result.origin(),
                 std::bind1st(std::multiplies<EXTENDED_COMPLEX>(), coeff));

  measure_simple_vector_observable<COMPLEX>(measurements,
                                            "Two_time_correlation_functions",
                                            to_complex_double_std_vector(result));
}

//for std::random_shuffle
class MyRandomNumberGenerator: public std::unary_function<unsigned int, unsigned int> {
 public:
  MyRandomNumberGenerator(alps::random01 &random) : random_(random) { };
  unsigned int operator()(unsigned int N) {
    return static_cast<unsigned int>(N * random_());
  }

 private:
  alps::random01 &random_;
};

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::local_updates() {
  assert(sliding_window.get_position_right_edge() == 0);

  boost::random::uniform_int_distribution<> dist(1, par["RANK_INSERTION_REMOVAL_UPDATE"].template as<int>());
  const int rank_ins_rem = dist(random.engine());
  const int current_n_window = std::max(N_win_standard / rank_ins_rem, 1);
  if (current_n_window != sliding_window.get_n_window()) {
    sliding_window.set_window_size(current_n_window, mc_config.operators, 0, ITIME_LEFT);
  }

  assert(sliding_window.get_position_right_edge() == 0);
  const int num_move = std::max(4 * current_n_window - 4, 1);
  for (int move = 0; move < num_move; ++move) {
    //insertion and removal of operators hybridized with the bath
    for (int update = 0; update < FLAVORS; ++update) {
      ins_rem_updater[rank_ins_rem - 1]->update(random, BETA, mc_config, sliding_window);
      ins_rem_diagonal_updater[rank_ins_rem - 1]->update(random, BETA, mc_config, sliding_window);
      //operator_pair_flavor_updater.update(random, BETA, mc_config, sliding_window);
    }

    //shift move of operators hybridized with the bath
    for (int update = 0; update < FLAVORS * rank_ins_rem; ++update) {
      single_op_shift_updater.update(random, BETA, mc_config, sliding_window);
    }

    //Worm insertion/removal
    if (worm_names.size() > 0) {
      //worm insertion and removal
      if (mc_config.current_config_space() == Z_FUNCTION_SPACE) {
        const int i_worm = static_cast<int>(random() * worm_insertion_removers.size());
        worm_insertion_removers[i_worm]->update(random, BETA, mc_config, sliding_window);
      } else {
        const int i_worm = static_cast<int>(mc_config.current_config_space()) - 1;
        worm_insertion_removers[i_worm]->update(random, BETA, mc_config, sliding_window);
      }

      //worm move
      const int i_worm = static_cast<int>(mc_config.current_config_space()) - 1;
      if (i_worm >= 0) {
        worm_movers[i_worm]->update(random, BETA, mc_config, sliding_window);
      }
    }

    //record expansion order to check if the Monte Carlo dynamics is thermalized
    thermalization_checker.add_sample(mc_config.M.size());

    //measure and adjust relative weight of Z-function and worm spaces
    if (!is_thermalized()) {
      adjust_worm_space_weight();
    }

    sliding_window.move_window_to_next_position(mc_config.operators);
  }
  sanity_check();
  assert(sliding_window.get_position_right_edge() == 0);
}

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::global_updates() {
  const std::size_t n_sliding_window_bak = sliding_window.get_n_window();
  sliding_window.set_window_size(1, mc_config.operators);

  std::vector<SCALAR> det_vec = mc_config.M.compute_determinant_as_product();

  //Swap flavors
  if (N_swap != 0 && sweeps % N_swap == 0 && swap_vector.size() > 0) {
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
        std::cerr << "A global shift is rejected!" << std::endl;
        exit(-1);
      }
    }
    sanity_check();
  }

  sliding_window.set_window_size(n_sliding_window_bak, mc_config.operators, 0, ITIME_LEFT);
  sanity_check();
}

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::update_MC_parameters() {
  assert(!is_thermalized());

  //Adjust window size according to perturbation order
  //collect expansion order
  std::vector<double> expansion_order_local(FLAVORS), expansion_order(FLAVORS);
  const std::vector<int> &tmp = count_creation_operators(FLAVORS, mc_config);
  for (int flavor = 0; flavor < FLAVORS; ++flavor) {
    expansion_order_local[flavor] = tmp[flavor];
  }
#ifdef ALPS_HAVE_MPI
  my_all_reduce<double>(comm, expansion_order_local, expansion_order, std::plus<double>());
  const double
      min_expansion_order = (1. * (*std::min_element(expansion_order.begin(), expansion_order.end()))) / comm.size();
#else
  expansion_order = expansion_order_local;
  const double min_expansion_order = (1.*(*std::min_element(expansion_order.begin(), expansion_order.end())));
#endif

  //new window size for single-pair insertion and removal update
  N_win_standard = static_cast<std::size_t>(
      std::max(
          par["MIN_N_SLIDING_WINDOW"].template as<int>(),
          std::min(
              static_cast<int>(std::ceil(min_expansion_order)),
              par["MAX_N_SLIDING_WINDOW"].template as<int>()
          )

      )
  );
  if (verbose && global_mpi_rank == 0) {
    std::cout << " new window size = " << N_win_standard << std::endl;
  }

  //Update parameters for single-operator shift updates
  single_op_shift_updater.update_parameters();

  //check if thermalization is checked
  if (!thermalized) {
    thermalization_checker.update(sweeps, (global_mpi_rank == 0 && verbose));
    if (p_flat_histogram_config_space) {
      thermalized = thermalization_checker.is_thermalized() && p_flat_histogram_config_space->converged();
    } else {
      thermalized = thermalization_checker.is_thermalized();
    }
  }
}

/////////////////////////////////////////////////
// Something to be done before measurement
/////////////////////////////////////////////////
template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::prepare_for_measurement() {
  g_meas_legendre.reset();
  single_op_shift_updater.finalize_learning();
  for (int k = 1; k < par["RANK_INSERTION_REMOVAL_UPDATE"].template as<int>() + 1; ++k) {
    ins_rem_diagonal_updater[k - 1]->finalize_learning();
  }
  if (global_mpi_rank == 0) {
    std::cout << "Thermalization process done after " << sweeps << " steps." << std::endl;
    std::cout << "The number of segments for sliding window update is " << N_win_standard << "."
        << std::endl;
    std::cout << "Perturbation orders (averaged over processes) are the following:" << std::endl;
  }
  const std::vector<int> &order_creation_flavor = count_creation_operators(FLAVORS, mc_config);
#ifdef ALPS_HAVE_MPI
  std::vector<int> tmp(FLAVORS, 0);
  my_all_reduce<int>(comm, order_creation_flavor, tmp, std::plus<int>());
  if (global_mpi_rank == 0) {
    for (int flavor = 0; flavor < FLAVORS; ++flavor) {
      std::cout << " flavor " << flavor << " " << tmp[flavor] / (1. * comm.size()) << std::endl;
    }
  }
#else
  for (int flavor = 0; flavor < FLAVORS; ++flavor) {
    std::cout << " flavor " << flavor << " " << order_creation_flavor[flavor] << std::endl;
  }
  std::cout << std::endl;
#endif

#ifdef ALPS_HAVE_MPI
  std::vector<double> tmp2(config_space_extra_weight.size(), 0);
  my_all_reduce<double>(comm, config_space_extra_weight, tmp2, std::plus<double>());
  config_space_extra_weight = tmp2;
  std::transform(
      config_space_extra_weight.begin(), config_space_extra_weight.end(), config_space_extra_weight.begin(),
      std::bind2nd(std::divides<double>(), 1.0*comm.size())
  );
  for (int w = 0; w < worm_names.size(); ++w) {
    worm_insertion_removers[w]->set_worm_space_weight(config_space_extra_weight[w + 1] );
  }
#endif
}

/**
 * Transform the single-particle Green's function back to the original basis
 */
template<typename MAT, typename MAT_COMPLEX, typename COMPLEX>
void
transform_G_back_to_original_basis(int FLAVORS,
                                   int SITES,
                                   int SPINS,
                                   int Np1,
                                   const MAT &rotmat_Delta,
                                   const MAT &inv_rotmat_Delta,
                                   std::vector<COMPLEX> &G) {
  assert(FLAVORS == SITES * SPINS);
  assert(G.size() == FLAVORS * FLAVORS * Np1);
  assert(SPINS == 2);

  MAT_COMPLEX mattmp(FLAVORS, FLAVORS), mattmp2(FLAVORS, FLAVORS);
  for (int time = 0; time < Np1; ++time) {
    for (int iflavor = 0; iflavor < FLAVORS; ++iflavor) {
      for (int jflavor = 0; jflavor < FLAVORS; ++jflavor) {
        mattmp(iflavor, jflavor) = G[(iflavor * FLAVORS + jflavor) * Np1 + time];
      }
    }
    mattmp2 = rotmat_Delta * mattmp * inv_rotmat_Delta;
    for (int iflavor = 0; iflavor < FLAVORS; ++iflavor) {
      for (int jflavor = 0; jflavor < FLAVORS; ++jflavor) {
        G[(iflavor * FLAVORS + jflavor) * Np1 + time] = mattmp2(iflavor, jflavor);
      }
    }
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
  if (mc_config.current_config_space() == Z_FUNCTION_SPACE) {
    p_flat_histogram_config_space->measure(0);
  } else {
    for (int iw = 0; iw < worm_names.size(); ++iw) {
      if (mc_config.p_worm->get_name() == worm_names[iw]) {
        p_flat_histogram_config_space->measure(iw + 1);
      }
    }
  }

  //adjust worm space weights
  config_space_extra_weight[0] = 1.0;
  for (int w = 0; w < worm_names.size(); ++w) {
    config_space_extra_weight[w + 1] = p_flat_histogram_config_space->weight_ratio(w + 1, 0);
    worm_insertion_removers[w]->set_worm_space_weight(config_space_extra_weight[w + 1] );
  }

  //If the histogram is flat enough,
  //we update the estimate of the volume of the configuration spaces.
  if (p_flat_histogram_config_space->flat_enough()) {
    p_flat_histogram_config_space->update_dos(false);
    config_space_extra_weight[0] = 1.0;
    for (int w = 0; w < worm_names.size(); ++w) {
      config_space_extra_weight[w + 1] = p_flat_histogram_config_space->weight_ratio(w + 1, 0);
      worm_insertion_removers[w]->set_worm_space_weight(config_space_extra_weight[w + 1] );
    }
    if (p_flat_histogram_config_space->converged()) {
      p_flat_histogram_config_space->finish_learning(false);
    }
  }
}
