#include "impurity.hpp"

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::define_parameters(parameters_type & parameters) {
  Base::define_parameters(parameters);

  alps::define_convenience_parameters(parameters);
  parameters
    .description("Continous-time hybridization expansion impurity solver")
    .define<int>("SITES", "number of sites/orbitals")
    .define<int>("SPINS", "number of spins")
    .define<double>("BETA", "inverse temperature")
    .define<int>("N_TAU", "number of points (minus 1) for G(tau), number of Matsubara frequencies for G(i omega_n)")
    .define<int>("N_LEGENDRE_MEASUREMENT", 100, "number of legendre coefficients for measuring G(tau)")
    .define<long>("SWEEPS", 1E+9, "number of sweeps for total run")
    .define<long>("THERMALIZATION", 500, "Minimum number of sweeps for thermalization")
    .define<int>("N_MEAS", 50, "Measurement is performed every N_MEAS updates.")
    .define<int>("N_MEAS_GREENS_FUNCTION", 10, "Measurement of Green's function is performed every N_MEAS_GREENS_FUNCTION updates.")
    .define<int>("N_SHIFT", 1, "how may shift moves attempted at each Monte Carlo step (N_SHIFT>0)")
    .define<int>("N_SWAP", 100, "We attempt to swap flavors every N_SWAP Monte Carlo steps.")
    .define<std::string>("SWAP_VECTOR", "",
                         "Definition of global updates in which the flavors of creation and annihilation operators are exchanged. Refer to manual for details.")
    .define<double>("ACCEPTANCE_RATE_CUTOFF", 0.01, "cutoff for acceptance rate in sliding window update")
    .define<double>("MIN_PAIR_DISTANCE", 0.0, "min of t_max for sliding window update")
    .define<int>("N_SLIDING_WINDOW", 1, "Number of segments for sliding window update")
    .define<int>("MAX_N_SLIDING_WINDOW", 1, "Maximum number of segments for sliding window update")
    .define<int>("N_UPDATE_CUTOFF", 50, "How many times the value of N_SLIDING_WINDOW is updated during thermalization.")
    .define<int>("Tmin", 1, "The scheduler checks longer than every Tmin seconds if the simulation is finished.")
    .define<int>("Tmax", 60, "The scheduler checks shorter than every Tmax seconds if the simulation is finished.")
    .define<int>("N_ORDER", 100, "Histogram of expansion order for each flavor is measured up to an expansion order of N_ORDER")
    .define<int>("MAX_ORDER", 10000, "Sum of expansion orders of all flavors cannot go beyond this value")
    .define<int>("N_TAU_TWO_TIME_CORRELATION_FUNCTIONS", 0, "Number of tau points for which two-time correlation functions are measured (tau=0, ...., beta/2)")
    .define<std::string>("TWO_TIME_CORRELATION_FUNCTIONS", "", "Input file for definition of two-time correlation functions to be measured")
    .define<int>("VERBOSE", 0, "If VERBOSE is not zero, more messages will be outputed.")
    ;

  IMP_MODEL::define_parameters(parameters);
}


template<typename IMP_MODEL>
HybridizationSimulation<IMP_MODEL>::HybridizationSimulation(parameters_type const & p, int rank)
  : alps::mcbase(p,rank),
    par(p),
    BETA(parameters["BETA"]),      //inverse temperature
    SITES(parameters["SITES"]),          //number of sites
    SPINS(parameters["SPINS"]),          //number of spins
    FLAVORS(SPINS*SITES),                             //flavors, i.e. #spins * #sites
    N(static_cast<int>(parameters["N_TAU"])),                  //time slices
    Np1(N+1),
    p_model(new IMP_MODEL(p,rank==0)),//impurity model
#ifdef ALPS_HAVE_MPI
    comm(),
#endif
    thermalization_checker(parameters["THERMALIZATION"].template as<long>()),          //minimum sweeps needed for thermalization
    total_sweeps(parameters["SWEEPS"]),                           //sweeps needed for total run
    N_meas(parameters["N_MEAS"]),
    N_meas_g(parameters["N_MEAS_GREENS_FUNCTION"]),
    N_shift(parameters["N_SHIFT"]),
    N_swap(parameters["N_SWAP"]),
    sweeps(0),                                                                 //sweeps done up to now
    M(0,0),
    sign(1),
    trace(std::numeric_limits<double>::max()),
    N_shift_flavor(FLAVORS, static_cast<int>(p["N_SHIFT"])),
    sliding_window(p_model.get(), BETA),
    acc_rate_cutoff(static_cast<double>(p["ACCEPTANCE_RATE_CUTOFF"])),
    max_distance_pair(BETA*std::max(0.1, p["MIN_PAIR_DISTANCE"].template as<double>())),
    weight_ins(  N/2, 0.5*BETA, 1,        max_distance_pair),
    weight_rem(  N/2, 0.5*BETA, 1,        max_distance_pair),
    weight_shift(N/2, 0.5*BETA, FLAVORS,  max_distance_pair),
    g_meas_legendre(FLAVORS,p["N_LEGENDRE_MEASUREMENT"],N,BETA),
    p_meas_corr(0),
    global_shift_acc_rate(),
    swap_acc_rate(0),
    prob_valid_rem_move(),
    timings(5, 0.0),
    verbose(p["VERBOSE"].template as<int>()!=0)
{
  /////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////
  ////Vectors Initialization Part//////////////////////////////////////
  ////Resize Vectors and Matrices so they have the right size./////////
  /////////////////////////////////////////////////////////////////////
  resize_vectors();

  /////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////
  ////ALPS Observables Part////////////////////////////////////////////
  ////Initialize the ALPS observables and store them //////////////////
  ////in the measurements container////////////////////////////////////
  /////////////////////////////////////////////////////////////////////
  create_observables();

  /////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////
  ////Initialize Monte Carlo configuration  ///////////////////////////
  /////////////////////////////////////////////////////////////////////
  operators.clear();
  creation_operators.clear();
  annihilation_operators.clear();
  if (p["N_SLIDING_WINDOW"].template as<int>() > p["MAX_N_SLIDING_WINDOW"].template as<int>()) {
    throw std::runtime_error("N_SLIDING_WINDOW cannot be larger than MAX_N_SLIDING_WINDOW.");
  }
  if (p["MAX_N_SLIDING_WINDOW"].template as<int>() < 1) {
    throw std::runtime_error("MAX_N_SLIDING_WINDOW cannot be smaller than 1.");
  }
  sliding_window.init_stacks(p["N_SLIDING_WINDOW"], operators);
  trace = sliding_window.compute_trace(operators);
  if (rank==0) {
    std::cout << "initial trace " << trace << " number of operators: "<<operators.size()<<"\n";
  }

  /////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////
  ////Check validity of parameters ////////////////////////////////////
  /////////////////////////////////////////////////////////////////////
  if (p["N_SHIFT"].template as<int>()==0) {
    throw std::runtime_error("N_shift=0 is a very bad idea!");
  }
  if (N_meas < 10 || N_meas%10 != 0) {
    throw std::runtime_error("N_MEAS must be larger than 10 and can be divided by 10!");
  }

  //Equal-time two-particle Green's function
  read_eq_time_two_particle_greens_meas();

  //Two-time correlation functions
  read_two_time_correlation_functions();

}


template<typename IMP_MODEL>
bool HybridizationSimulation<IMP_MODEL>::is_thermalized() const
{
#ifdef ALPS_HAVE_MPI
  return thermalization_checker.is_thermalized(sweeps, comm);
#else
  return thermalization_checker.is_thermalized(sweeps);
#endif
}

template<typename IMP_MODEL>
double HybridizationSimulation<IMP_MODEL>::fraction_completed() const
{
  double work=(is_thermalized() ? (sweeps-thermalization_checker.get_actual_thermalization_steps())/double(total_sweeps) : 0.);
  if (work>1.0) {
    work = 1.0;
  }
  return work;
}

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::update() {
  const int N_update_cutoff = par["N_UPDATE_CUTOFF"].template as<int>();
  const long interval_update_cutoff = static_cast<long>(std::max(static_cast<double>(par["THERMALIZATION"].template as<long>())/N_update_cutoff, 1.0));

  const unsigned int max_order = par["MAX_ORDER"].template as<unsigned int>();

  if (N_update_cutoff > par["THERMALIZATION"].template as<long>()) {
    throw std::runtime_error("N_UPDATE_CUTOFF must be smaller than THERMALIZATION_SWEEPS.");
  }

#ifdef MEASURE_TIMING
  boost::timer::cpu_timer timer;
#endif

  //////////////////////////////////
  // Monte Carlo updates
  //////////////////////////////////
  for (int imeas = 0; imeas < N_meas; imeas++) {    // accumulate measurements from N_meas updates before storing
    sweeps++;

    //std::cout << "sweeps " << sweeps << std::endl;
    thermalization_checker.add_sample(operators.size()/2);

#ifdef MEASURE_TIMING
    const double time1 = timer.elapsed().wall*1E-9;
#endif

    /**** try to insert or remove a pair of operators with the same flavor ****/
    for (int update = 0; update < FLAVORS; update++) {
      const int flavor_target = (int) (random() * FLAVORS);
      boost::tuple<int, bool, double, SCALAR, bool> r =
        insert_remove_pair_flavor<SCALAR,EXTENDED_SCALAR>(random, flavor_target,flavor_target, BETA,
                                  order_creation_flavor,order_annihilation_flavor, creation_operators, annihilation_operators,
                                  M, sign, trace, operators, max_distance_pair, sliding_window, max_order);

      check_consistency_operators(operators, creation_operators, annihilation_operators);
      sanity_check();

      //measure the distance dependence of weight for insertion
      if (boost::get<4>(r)) {//only if a valid move was generated
        const double op_distance = std::min(BETA - boost::get<2>(r), boost::get<2>(r));
        if (boost::get<0>(r) == 0) {
          //in case of insertion
          if (boost::get<1>(r)) {
            weight_ins.add_sample(op_distance, 1.0, 0);
          } else {
            weight_ins.add_sample(op_distance, 0.0, 0);
          }
        } else {
          prob_valid_rem_move.accepted();
          //in case of removal
          if (boost::get<1>(r)) {
            weight_rem.add_sample(op_distance, 1.0, 0);
          } else {
            weight_rem.add_sample(op_distance, 0.0, 0);
          }
        }
      } else {
        if (boost::get<0>(r) == 1) {
          //in case of removal
          prob_valid_rem_move.rejected();
        }
      }
    }

    /**** insert or remove a pair with random flavors ****/
    for (int update = 0; update < FLAVORS; update++) {
      int c_flavor = (int) (random() * FLAVORS);
      int a_flavor = (int) (random() * FLAVORS);
      if (c_flavor==a_flavor) {
        continue;
      }
      boost::tuple<int, bool, double, SCALAR, bool> r = insert_remove_pair_flavor(random, c_flavor,
                                a_flavor,
                                BETA,
                                order_creation_flavor,
                                order_annihilation_flavor,
                                creation_operators,
                                annihilation_operators, M,
                                sign, trace, operators,
                                max_distance_pair,
                                sliding_window, max_order);
      check_consistency_operators(operators, creation_operators, annihilation_operators);
      sanity_check();
      if (boost::get<4>(r)) {//only if a valid move was generated
        const double op_distance = std::min(BETA - boost::get<2>(r), boost::get<2>(r));
        const double acc = boost::get<1>(r) ? 1.0 : 0.0;
        if (boost::get<0>(r) == 0) {
          //insertion
          weight_ins.add_sample(op_distance, acc, 0);
        } else {
          //removal
          weight_rem.add_sample(op_distance, acc, 0);
        }
      }
    }

    /**** shift an operator ****/
    for (int ns = 0; ns < N_shift*FLAVORS; ns++) {
      sanity_check();
      boost::tuple<bool, double, bool, int> r = shift_lazy(random, BETA,
                                                      creation_operators, annihilation_operators, M, sign,
                                                      trace,
                                                      operators,
                                                      weight_shift.get_cutoff(),
                                                      sliding_window);
      check_consistency_operators(operators, creation_operators, annihilation_operators);
      sanity_check();
      if (boost::get<2>(r)) {
        const double op_distance = std::min(BETA - boost::get<1>(r), boost::get<1>(r));
        if (boost::get<0>(r)) {
          weight_shift.add_sample(op_distance, 1.0, boost::get<3>(r));
        } else {
          weight_shift.add_sample(op_distance, 0.0, boost::get<3>(r));
        }
      }
    }

    if (my_isnan(trace)) {
      throw std::runtime_error("trace is NaN after insertion/removal/shift update");
    }
    if (my_isnan(sign)) {
      throw std::runtime_error("sign is NaN after insertion/removal/shift update");
    }

#ifdef MEASURE_TIMING
    const double time2 = timer.elapsed().wall*1E-9;
    timings[0] += time2 - time1;
#endif

    //Perform global updates which might cost O(beta)
    expensive_updates();

    if (my_isnan(trace)) {
      throw std::runtime_error("trace is NaN after global update");
    }
    if (my_isnan(sign)) {
      throw std::runtime_error("sign is NaN after global update");
    }

    check_consistency_operators(operators, creation_operators, annihilation_operators);
    sanity_check();

    //update parameters for MC moves and window size
    if (N_update_cutoff>0 && !is_thermalized() && sweeps%interval_update_cutoff==interval_update_cutoff-1) {
      update_MC_parameters();
    }

#ifdef MEASURE_TIMING
    const double time3 = timer.elapsed().wall*1E-9;
    timings[1] += time3 - time2;
#endif

    // move the window to the next position
    sliding_window.move_window_to_next_position(operators);

#ifdef MEASURE_TIMING
    const double time4 = timer.elapsed().wall*1E-9;
    timings[2] += time4 - time3;
#endif

    // measure single-particle Green's function
    if (is_thermalized() && sweeps%N_meas_g == 0) {
      g_meas_legendre.measure(M, operators, creation_operators, annihilation_operators, sign);
    }

#ifdef MEASURE_TIMING
    const double time5 = timer.elapsed().wall*1E-9;
    timings[3] += time5 - time4;
#endif

    sanity_check();
  }//loop up to N_meas
}

//////////////////////////////////
// ALPS measurements
//////////////////////////////////
template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::measure() {
  assert(is_thermalized());
#ifdef MEASURE_TIMING
  boost::timer::cpu_timer timer;
#endif

  // measure the perturbation order
  const int N_order = par["N_ORDER"].template as<int>();
  for (int flavor=0; flavor<FLAVORS; ++flavor) {
    std::vector<double> order_creation_meas(FLAVORS*N_order, 0.0);
    if (order_creation_flavor[flavor]<N_order) {
      order_creation_meas[flavor*N_order+order_creation_flavor[flavor]] = 1.0;
    }
    measurements["order"] << order_creation_meas;
  }
  {
    std::vector<double> tmp;
    std::copy(order_creation_flavor.begin(), order_creation_flavor.end(), std::back_inserter(tmp));
    measurements["PerturbationOrderFlavors"] << tmp;
  }

  measurements["AbsTrace"] << convert_to_double(myabs(trace));

  // measure acceptance rate
  measurements["Insertion_attempted"] << to_std_vector(weight_ins.get_counter());
  measurements["Insertion_accepted"] << to_std_vector(weight_ins.get_sumval());
  weight_ins.reset();

  measurements["Removal_attempted"] << to_std_vector(weight_rem.get_counter());
  measurements["Removal_accepted"] << to_std_vector(weight_rem.get_sumval());
  weight_rem.reset();

  measurements["Shift_attempted"] << to_std_vector(weight_shift.get_counter());
  measurements["Shift_accepted"] << to_std_vector(weight_shift.get_sumval());
  weight_shift.reset();

  //measure acceptance rate of global shift
  if (global_shift_acc_rate.has_samples()) {
    measurements["Acceptance_rate_global_shift"] << global_shift_acc_rate.compute_acceptance_rate();
    global_shift_acc_rate.reset();
  }

  if (prob_valid_rem_move.has_samples()) {
    measurements["Probability_valid_removal_move"] << prob_valid_rem_move.compute_acceptance_rate();
    prob_valid_rem_move.reset();
  }

  //measure acceptance rate of swap update
  if (swap_acc_rate.size()>0 && swap_acc_rate[0].has_samples()) {
    std::vector<double> acc_swap(swap_acc_rate.size());
    for (int iupdate=0; iupdate<swap_acc_rate.size(); ++iupdate) {
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
                                              g_meas_legendre.get_measured_legendre_coefficients(Eigen::Matrix<SCALAR,Eigen::Dynamic,Eigen::Dynamic>::Identity(FLAVORS,FLAVORS))
                                            )
    );
    g_meas_legendre.reset();
  }

  measurements["Sign"] << mycast<double>(sign);

  //fidelity susceptibility
  /*
  {
      measure_scalar_observable<SCALAR>(measurements, "kLkR", static_cast<double>(measure_kLkR(operators, BETA,
                                                                                               0.5 * BETA *
                                                                                               random())) *
                                                              sign);
      measure_scalar_observable<SCALAR>(measurements, "k", static_cast<double>(operators.size()) * sign);
  }
  */

#ifdef MEASURE_TIMING
  timings[4] = timer.elapsed().wall*1E-9;
  measurements["TimingsSecPerNMEAS"] << timings;
  std::fill(timings.begin(), timings.end(), 0.0);
#endif
}

//Measure the expectation values of density operators
template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::measure_n() {
  assert(is_thermalized());
  MeasStaticObs<SlidingWindowManager<IMP_MODEL>,CdagC> meas(sliding_window, operators);
  std::vector<CdagC> ops(FLAVORS);
  std::vector<EXTENDED_COMPLEX> result_meas(FLAVORS);
  for (int flavor=0; flavor<FLAVORS; ++flavor) {
    boost::array<int,2> flavors_tmp;
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
  EXTENDED_COMPLEX inv_trace = static_cast<EXTENDED_SCALAR>(EXTENDED_SCALAR(1.0)/trace);
  for (int flavor=0; flavor<FLAVORS; ++flavor) {
    result_meas_Re[flavor] = convert_to_scalar(static_cast<EXTENDED_REAL>(get_real(result_meas[flavor]*sign*inv_trace)));
  }
  measurements["n"] << result_meas_Re;
}

//Measure two-time correlation functions
template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::measure_two_time_correlation_functions() {
  assert(is_thermalized());
  if (p_meas_corr.get()==0) {
    return;
  }

  boost::multi_array<EXTENDED_COMPLEX,2> result;
  p_meas_corr->perform_meas(sliding_window, operators, result);
  const EXTENDED_COMPLEX coeff = EXTENDED_COMPLEX(sign)/EXTENDED_COMPLEX(trace);
  std::transform(result.origin(), result.origin()+result.num_elements(), result.origin(),
                 std::bind1st(std::multiplies<EXTENDED_COMPLEX>(), coeff));

  measure_simple_vector_observable<COMPLEX>(measurements, "Two_time_correlation_functions", to_complex_double_std_vector(result));
}

//for std::random_shuffle
class MyRandomNumberGenerator : public std::unary_function<unsigned int, unsigned int> {
public:
  MyRandomNumberGenerator(alps::random01 &random) : random_(random) {};
  unsigned int operator()(unsigned int N) {
    return static_cast<unsigned int>(N*random_());
  }

private:
  alps::random01& random_;
};

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::expensive_updates() {
  const bool do_swap = (N_swap != 0 && sweeps%N_swap == 0 && swap_vector.size() > 0);
  //const bool do_global_shift = (N_swap != 0 && sweeps%N_swap == 0 && swap_vector.size() > 0);
  const bool do_global_shift = (sliding_window.get_position_right_edge()==0);

  if (!do_swap && !do_global_shift) {
    return;
  }
  if (random() < 0.1) {
    //avoid doing updates in a periodic way.
    return;
  }

  const std::size_t n_sliding_window_bak = sliding_window.get_n_window();
  sliding_window.set_window_size(1, operators);

  //Swap flavors
  if (do_swap) {
    //do updates randomly
    std::vector<int> execute_ordering;
    for (int i=0; i<swap_vector.size(); ++i) {
      execute_ordering.push_back(i);
    }
    MyRandomNumberGenerator rnd(random);
    std::random_shuffle(execute_ordering.begin(), execute_ordering.end(), rnd);

    for (int itry = 0; itry < swap_vector.size(); ++itry) {
      const int iupdate = execute_ordering[itry];
      const bool accepted = exchange_flavors<SCALAR,EXTENDED_SCALAR>(random, BETA, creation_operators,
                   annihilation_operators,
                   order_creation_flavor, order_annihilation_flavor,
                   M, sign, trace, operators,
                   sliding_window,
                   FLAVORS, swap_vector[iupdate].first.begin()
      );

      if (accepted) {
        swap_acc_rate[iupdate].accepted();
        if (my_isnan(trace)) {
          throw std::runtime_error("trace is NaN after swap update"+boost::lexical_cast<std::string>(itry));
        }
        if (my_isnan(sign)) {
          throw std::runtime_error("sign is NaN after swap update"+boost::lexical_cast<std::string>(itry));
        }
      } else {
        swap_acc_rate[iupdate].rejected();
      }
    }
  }

  //Shift operators to restore translational symmetry
  if (do_global_shift) {
    const bool accepted = global_shift(random, BETA, creation_operators,
                                       annihilation_operators,
                                       order_creation_flavor, order_annihilation_flavor,
                                       M, sign, trace, operators, sliding_window);
    if (accepted) {
      global_shift_acc_rate.accepted();
      if (my_isnan(trace)) {
        throw std::runtime_error("trace is NaN after global shift");
      }
      if (my_isnan(sign)) {
        throw std::runtime_error("sign is NaN after global shift");
      }
    } else {
      global_shift_acc_rate.rejected();
      if (p_model->translationally_invariant()) {
        std::cerr << "A global shift is rejected!" << std::endl;
        exit(-1);
      }
    }
    sanity_check();
  }

  const ITIME_AXIS_LEFT_OR_RIGHT new_move_direction = random()<0.5 ? ITIME_LEFT : ITIME_RIGHT;
  sliding_window.set_window_size(n_sliding_window_bak, operators,
                                 static_cast<int>((2*n_sliding_window_bak-1)*random()),
                                 new_move_direction);
}

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::update_MC_parameters() {
  assert(!is_thermalized());

  const int N_update_cutoff = par["N_UPDATE_CUTOFF"].template as<int>();
  const long interval_update_cutoff = static_cast<long>(std::max(static_cast<double>(par["THERMALIZATION"].template as<long>())/N_update_cutoff, 1.0));

  const double mag = std::pow(
    2.0,
    1.0/std::max(static_cast<double>(sweeps)/static_cast<double>(interval_update_cutoff),1.0)
  );

  std::vector<double> ratmp;
#ifdef ALPS_HAVE_MPI
  ratmp.push_back(weight_ins.update_cutoff(acc_rate_cutoff, mag, comm));
  ratmp.push_back(weight_rem.update_cutoff(acc_rate_cutoff, mag, comm));
#else
  ratmp.push_back(weight_ins.update_cutoff(acc_rate_cutoff, mag));
  ratmp.push_back(weight_rem.update_cutoff(acc_rate_cutoff, mag));
#endif
  max_distance_pair = *std::max_element(ratmp.begin(), ratmp.end());

#ifdef ALPS_HAVE_MPI
  ratmp.push_back(weight_shift.update_cutoff(acc_rate_cutoff, mag, comm));
#else
  ratmp.push_back(weight_shift.update_cutoff(acc_rate_cutoff, mag));
#endif
  //max_distance_shift = ratmp.back();
  const double max_distance = *std::max_element(ratmp.begin(), ratmp.end());

  //care about expansion order
  std::vector<double> expansion_order(1);
  std::vector<double> expansion_order_local;
  expansion_order_local.push_back(1.0*operators.size()/2);
#ifdef ALPS_HAVE_MPI
  my_all_reduce<double>(comm, expansion_order_local, expansion_order, std::plus<double>());
  expansion_order[0] /= comm.size();
#else
  expansion_order[0] = expansion_order_local[0];
#endif

  const std::size_t n_window_new = static_cast<std::size_t>(
    std::min(
      std::min(
        static_cast<int>(std::ceil(expansion_order[0]/FLAVORS)),
        std::max(
          1, static_cast<int>(BETA/(2.0*max_distance))
        )
      ),
      par["MAX_N_SLIDING_WINDOW"].template as<int>()
    )
  );
  //max_distance_pair = BETA/(2*n_window_new);

  if (n_window_new != sliding_window.get_n_window()) {
    const ITIME_AXIS_LEFT_OR_RIGHT new_move_direction = random()<0.5 ? ITIME_LEFT : ITIME_RIGHT;
    sliding_window.set_window_size(n_window_new, operators,
                                   static_cast<int>((2*n_window_new-1)*random()),
                                   new_move_direction);
  }
}

/////////////////////////////////////////////////
// Something to be done before measurement
/////////////////////////////////////////////////
template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::prepare_for_measurement() {
  g_meas_legendre.reset();
  weight_ins.reset();
  weight_rem.reset();
  weight_shift.reset();
  prob_valid_rem_move.reset();
  if (global_mpi_rank==0) {
    std::cout << "We're done with thermalization." << std::endl <<
    "The number of segments for sliding window update is " << sliding_window.get_n_window() << "." << std::endl <<
    std::endl;
    std::cout << "Perturbation orders (averaged over processes) are the following:" << std::endl;
  }
#ifdef ALPS_HAVE_MPI
  std::vector<int> tmp(FLAVORS, 0);
  my_all_reduce<int>(comm, order_creation_flavor, tmp, std::plus<int>());
  if (global_mpi_rank==0) {
    for (int flavor=0; flavor<FLAVORS; ++flavor) {
      std::cout << " flavor " << flavor << " " << tmp[flavor]/(1.*comm.size()) << std::endl;
    }
  }
#else
  for (int flavor=0; flavor<FLAVORS; ++flavor) {
    std::cout << " flavor " << flavor << " " << order_creation_flavor[flavor] << std::endl;
  }
  std::cout << std::endl;
#endif

  if (verbose && global_mpi_rank==0) {
    std::cout << "t_max for insertion/removal updates " << std::max(weight_ins.get_cutoff(0),weight_rem.get_cutoff(0)) << std::endl;
    std::cout << "t_max for shift update" << std::endl;
    for (int flavor = 0; flavor < FLAVORS; ++flavor) {
      std::cout << " flavor = " << flavor << " , t_max = "
      << weight_shift.get_cutoff(flavor)<<" "
      << std::endl;
    }
  }

  //N_meas
  const int N_meas_min = std::max(10, 4*sliding_window.get_n_window());//a sweep of the window takes 4*get_n_window()
  if (N_meas<N_meas_min) {
    N_meas = N_meas_min;
    if (global_mpi_rank==0) {
      std::cout << "Warning N_MEAS is too small: using N_MEAS = "  << N_meas << " instead." << std::endl;
    }
  }

  //N_meas_g
  const int N_meas_g_min = std::max(10, sliding_window.get_n_window());//a sweep of the window takes 4*get_n_window()
  if (N_meas_g<N_meas_g_min) {
    N_meas_g = N_meas_g_min;
    if (global_mpi_rank==0) {
      std::cout << "Warning N_MEAS_GREENS_FUNCTION is too small: using N_MEAS_GREENS_FUNCTION = "  << N_meas_g << " instead." << std::endl;
    }
  }

  //N_swap
  const int N_swap_min =std::max(10, 4*sliding_window.get_n_window());//a sweep of the window takes 4*get_n_window()
  if (N_swap<N_swap_min && swap_vector.size()>0) {
    N_swap = N_swap_min;
    if (global_mpi_rank==0) {
      std::cout << "Warning N_SWAP is too small: using N_SWAP = "  << N_swap << " instead." << std::endl;
    }
  }
}

template<typename MAT, typename MAT_COMPLEX, typename COMPLEX>
void
transform_G_back_to_original_basis(int FLAVORS, int SITES, int SPINS, int Np1, const MAT& rotmat_Delta, const MAT& inv_rotmat_Delta, std::vector<COMPLEX>& G) {
  assert(FLAVORS==SITES*SPINS);
  assert(G.size()==FLAVORS*FLAVORS*Np1);
  assert(SPINS==2);

  MAT_COMPLEX mattmp(FLAVORS,FLAVORS), mattmp2(FLAVORS,FLAVORS);
  for (int time=0; time<Np1; ++time) {
    for (int iflavor=0; iflavor<FLAVORS; ++iflavor) {
      for (int jflavor=0; jflavor<FLAVORS; ++jflavor) {
        mattmp(iflavor,jflavor) = G[(iflavor*FLAVORS+jflavor)*Np1+time];
      }
    }
    //G V^\dagger -> mattmp2
    //mattmp.matrix_right_multiply(inv_rotmat_Delta, mattmp2);
    //V G V^\dagger -> mattmp2
    //rotmat_Delta.matrix_right_multiply(mattmp2,mattmp);
    mattmp2 = rotmat_Delta*mattmp*inv_rotmat_Delta;
    for (int iflavor=0; iflavor<FLAVORS; ++iflavor) {
      for (int jflavor=0; jflavor<FLAVORS; ++jflavor) {
        G[(iflavor*FLAVORS+jflavor)*Np1+time] = mattmp2(iflavor,jflavor);
      }
    }
  }
}

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::sanity_check() const {
#ifndef NDEBUG
  alps::ResizableMatrix<SCALAR> M_new;

  // test consistency of operators
  {
    std::vector<int> order_creation_flavor_new(FLAVORS, 0), order_annihilation_flavor_new(FLAVORS, 0);
    operator_container_t c_ops_new, a_ops_new;
    int itmp = 0;
    for (operator_container_t::iterator it = operators.begin(); it!=operators.end(); ++it) {
      if (it->type()==0) {
        c_ops_new.insert(*it);
        ++order_creation_flavor_new[it->flavor()];
      } else {
        a_ops_new.insert(*it);
        ++order_annihilation_flavor_new[it->flavor()];
      }
    }
    assert(c_ops_new==creation_operators);
    assert(a_ops_new==annihilation_operators);
    assert(order_annihilation_flavor==order_annihilation_flavor_new);
    assert(order_creation_flavor==order_creation_flavor_new);
    if (c_ops_new!=creation_operators)
      throw std::logic_error("creation_operators is not consistent with operators.");
    if (a_ops_new!=annihilation_operators)
      throw std::logic_error("annihilation_operators is not consistent with operators.");
  }

  // compute determinants
  SCALAR sign_det = 1.0;
  if (creation_operators.size() > 0) {
    const std::vector<SCALAR>& det_vec = cal_det_as_vector(creation_operators, annihilation_operators, M_new, BETA, p_model->get_F());
    for (int i=0; i<creation_operators.size(); ++i) {
      sign_det *= mysign(det_vec[i]);
    }
  }

  // compute permuation sign
  SCALAR sign_new = 1.0;

  operator_container_t operators_new;
  operators_new.clear();
  operator_container_t::iterator it_c=creation_operators.begin();
  for (operator_container_t::iterator it_a = annihilation_operators.begin(); it_a != annihilation_operators.end(); it_a++) {

    for (operator_container_t::iterator ito = operators_new.begin(); ito != operators_new.end(); ito++)
      if (ito->time() > it_c->time()) {
        sign_new *= -1;
      }
    operators_new.insert(*it_c);       // creation operator

    for (operator_container_t::iterator ito = operators_new.begin(); ito != operators_new.end(); ito++)
      if (ito->time() > it_a->time()) {
        sign_new *= -1;
      }
    operators_new.insert(*it_a); // annihilation operator

    it_c++;
  }

  // compute trace
  EXTENDED_SCALAR trace_sw = sliding_window.compute_trace(operators_new);

  if (myabs(convert_to_scalar(static_cast<EXTENDED_SCALAR>(trace_sw/trace))-1.0)>1E-5) {
    throw std::runtime_error("trace != trace_new");
  }

  SCALAR sign_overall_new = dsign(sign_new)
                            *convert_to_scalar(dsign(trace))
                            *sign_det;
  if (std::abs(sign_overall_new/sign-1.0)>1E-5) {
    throw std::runtime_error("sign_overall_new != sign");
  }
#endif
}
