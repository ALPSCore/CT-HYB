#ifdef I_AM_USING_IDE
#include "impurity.hpp"
#endif

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::define_parameters(parameters_type & parameters) {
  Base::define_parameters(parameters);

  alps::define_convenience_parameters(parameters);
  parameters
    .description("Continous-time hybridization expansion impurity solver")
    .define<int>("SITES", "number of sites/orbitals")
    .define<int>("SPINS", "number of spins")
    .define<double>("BETA", "inverse temperature")
    .define<int>("N_TAU", "number of bins for G(tau), number of Matsubara frequencies for G(i omega_n)")
    .define<int>("N_LEGENDRE_MEASUREMENT", 100, "number of legendre coefficients for measuring G(tau)")
    .define<long>("SWEEPS", "number of sweeps for total run")
    .define<long>("THERMALIZATION", 500, "number of sweeps for thermalization")
    .define<int>("N_MEAS", 50, "Measurement is performed every N_MEAS updates.")
    .define<int>("N_SHIFT", 1, "how may shift moves attempted at each Monte Carlo step (N_SHIFT>0)")
    .define<int>("N_SWAP", 0, "Flavor-swap moves attempted every N_SWAP Monte Carlo steps.")
    .define<double>("ACCEPTANCE_RATE_CUTOFF", 0.1, "cutoff for acceptance rate in sliding window update")
    .define<int>("USE_SLIDING_WINDOW", 1, "Switch for sliding window update")
    .define<int>("N_SLIDING_WINDOW", 5, "Number of segments for sliding window update")
    .define<int>("N_UPDATE_CUTOFF", 50, "How many times N_SLIDING_WINDOW is updated during thermalization.")
    .define<int>("Tmin", 1, "The scheduler checks longer than every Tmin seconds if the simulation is finished.")
    .define<int>("Tmax", 60, "The scheduler checks shorter than every Tmax seconds if the simulation is finished.")
    .define<int>("N_ORDER", 100, "Histogram of expansion order for each flavor is measured up to an expansion order of N_ORDER")
    .define<int>("MAX_ORDER", 10000, "Sum of expansion orders of all flavors cannot go beyond this value")
    .define<int>("N_TAU_TWO_TIME_CORRELATION_FUNCTIONS", 0, "Number of tau points for which two-time correlation functions are measured (tau=0, ...., beta/2)")
    .define<std::string>("TWO_TIME_CORRELATION_FUNCTIONS", "", "Input file for definition of two-time correlation functions to be measured")
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
    comm(),
    thermalization_sweeps(parameters["THERMALIZATION"]),          //sweeps needed for thermalization
    total_sweeps(parameters["SWEEPS"]),                           //sweeps needed for total run
    N_meas(parameters["N_MEAS"]),
    N_shift(parameters["N_SHIFT"]),
    N_swap(parameters["N_SWAP"]),
    sweeps(0),                                                                 //sweeps done up to now
    M(0,0),
    sign(1),
    det(1),
    trace(std::numeric_limits<double>::max()),
    N_shift_flavor(FLAVORS, static_cast<int>(p["N_SHIFT"])),
    sliding_window(p_model.get(), BETA),
    max_dist_optimizer(N, 0.5*BETA, 3*FLAVORS),//ins, rem, shift
    weight_vs_distance(N, 0.5*BETA),
    weight_vs_distance_shift(N, 0.5*BETA),
    max_distance_pair(BETA*0.1),
    max_distance_shift(BETA*0.1),
    acc_rate_cutoff(static_cast<double>(p["ACCEPTANCE_RATE_CUTOFF"])),
    G_meas_new(FLAVORS*FLAVORS*(N+1)),
    g_meas_legendre(FLAVORS,p["N_LEGENDRE_MEASUREMENT"],N,BETA),
    p_meas_corr(0),
    global_shift_acc_rate()
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
  sliding_window.init_stacks(p["N_SLIDING_WINDOW"], operators);
  trace = sliding_window.compute_trace(operators);
  if (rank==0) {
    std::cout << "initial trace (sliding window) " << trace << " number of operators: "<<operators.size()<<"\n";
  }

  /////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////
  ////Check validity of parameters ////////////////////////////////////
  /////////////////////////////////////////////////////////////////////
  if (p["N_SHIFT"].template as<int>()==0) {
    throw std::runtime_error("N_shift=0 is a very bad idea!");
  }

  //Equal-time two-particle Green's function
  read_eq_time_two_particle_greens_meas();

  //Two-time correlation functions
  read_two_time_correlation_functions();
}


template<typename IMP_MODEL>
bool HybridizationSimulation<IMP_MODEL>::is_thermalized() const
{
  return (sweeps >= thermalization_sweeps);
}

template<typename IMP_MODEL>
double HybridizationSimulation<IMP_MODEL>::fraction_completed() const
{
  double work=(is_thermalized() ? (sweeps-thermalization_sweeps)/double(total_sweeps) : 0.);
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

  if (N_update_cutoff > thermalization_sweeps) {
    throw std::runtime_error("N_UPDATE_CUTOFF must be smaller than THERMALIZATION_SWEEPS.");
  }

  std::fill(G_meas_new.begin(),G_meas_new.end(),0.0);

  //////////////////////////////////
  // Monte Carlo updates
  //////////////////////////////////
  for (int imeas = 0; imeas < N_meas; imeas++) {    // accumulate measurements from N_meas updates before storing
    sweeps++;
    //std::cout << "sweeps " << sweeps << " operator.size = " << operators.size() << " window size " << sliding_window.get_n_window() << " rank " << comm.rank() << std::endl;

    /**** try to insert or remove a pair of operators with the same flavor ****/
    for (int flavor = 0; flavor < FLAVORS; flavor++) {
      const int flavor_target = (int) (random() * FLAVORS);
      boost::tuple<int, bool, double, SCALAR, bool> r =
        insert_remove_pair_flavor(random, flavor_target,flavor_target,det, BETA,
                                  order_creation_flavor,order_annihilation_flavor, creation_operators, annihilation_operators,
                                  M, sign, trace, operators, max_distance_pair, sliding_window, max_order);

      
      check_consistency_operators(operators, creation_operators, annihilation_operators);
      sanity_check();

      //measure the distance dependence of weight for insertion
      if (boost::get<0>(r) == 0 && boost::get<4>(r)) {//only for insertion
        const double op_distance = std::min(BETA - boost::get<2>(r), boost::get<2>(r));
        if (boost::get<1>(r)) {
          weight_vs_distance.add_sample(op_distance, 1.0);
        } else {
          weight_vs_distance.add_sample(op_distance, 0.0);
        }
      }
    }

    //std::cout << "sweeps " << sweeps << " checkpoint A rank " << comm.rank() << std::endl;

    /**** insert or remove a pair with random flavors ****/
    for (int flavor = 0; flavor < FLAVORS; flavor++) {
      int c_flavor = (int) (random() * FLAVORS);
      int a_flavor = (int) (random() * FLAVORS);
      insert_remove_pair_flavor(random, c_flavor,
                                a_flavor,
                                det, BETA,
                                order_creation_flavor,
                                order_annihilation_flavor,
                                creation_operators,
                                annihilation_operators, M,
                                sign, trace, operators,
                                max_distance_pair,
                                sliding_window, max_order);
      check_consistency_operators(operators, creation_operators, annihilation_operators);
      sanity_check();
    }

    /**** shift an operator ****/
    for (int ns = 0; ns < N_shift*FLAVORS; ns++) {
      sanity_check();
      boost::tuple<bool, double, bool, int> r = shift_lazy(random, det, BETA,
                                                      creation_operators, annihilation_operators, M, sign,
                                                      trace,
                                                      operators, max_distance_shift,
                                                      sliding_window);
      check_consistency_operators(operators, creation_operators, annihilation_operators);
      sanity_check();
      if (boost::get<2>(r)) {
        const double op_distance = std::min(BETA - boost::get<1>(r), boost::get<1>(r));
        if (boost::get<0>(r)) {
          weight_vs_distance_shift.add_sample(op_distance, 1.0);
        } else {
          weight_vs_distance_shift.add_sample(op_distance, 0.0);
        }
      }
    }

    //std::cout << "sweeps " << sweeps << " checkpoint B rank " << comm.rank() << std::endl;

    //Perform global updates which might cost O(beta)
    expensive_updates();

    check_consistency_operators(operators, creation_operators, annihilation_operators);
    sanity_check();

    //update parameters for MC moves and window size
    //std::cout << "Checking if update_MC_parameters rank is called " << comm.rank() << " sweeps " << sweeps << " " << !is_thermalized() << " " << static_cast<int>(sweeps%interval_update_cutoff==interval_update_cutoff-1) << std::endl;
    if (N_update_cutoff>0 && !is_thermalized() && sweeps%interval_update_cutoff==interval_update_cutoff-1) {
      //std::cout << "Calling update_MC_parameters rank " << comm.rank() << " sweeps " << sweeps << std::endl;
      update_MC_parameters();
      //std::cout << "Exiting update_MC_parameters rank " << comm.rank() << " sweeps " << sweeps << std::endl;
    }

    // move the window to the next position
    sliding_window.move_window_to_next_position(operators);

    // measure single-particle Green's function
    if (is_thermalized()) {
      g_meas_legendre.measure(M, operators, creation_operators, annihilation_operators, sign);
      {
        operator_container_t::iterator it1, it2;
        for (int k=0; k<M.size1(); k++) {
          (k==0 ? it1 = annihilation_operators.begin() : it1++);
          for (int l=0; l<M.size1(); l++) {
            (l==0 ? it2 = creation_operators.begin() : it2++);
            if (M(l,k)!=0.0) {
              double argument = it1->time()-it2->time();
              double bubble_sign=1;
              if (argument > 0) {
                bubble_sign = 1;
              } else {
                bubble_sign = -1;
                argument += BETA;
              }

              int flavor_a=it1->flavor();
              int flavor_c=it2->flavor();
              int index = (int)(argument/BETA*N+0.5);
              G_meas_new[(flavor_a*FLAVORS+flavor_c)*(N+1)+index] += M(l,k)*bubble_sign*sign;
            }
          }
        }
      }
    }

    //std::cout << "sweeps " << sweeps << " checkpoint C rank " << comm.rank() << " N_meas " << N_meas << "N_win " << sliding_window.get_state() << std::endl;
    sanity_check();
  }//loop up to N_meas
  //std::cout << "Exiting update() " << sweeps << " rank " << comm.rank() << std::endl;
}

//////////////////////////////////
// ALPS measurements
//////////////////////////////////
template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::measure() {
  assert(is_thermalized());
  //std::cout << "Call measurement rank " << comm.rank() << std::endl;

  // measure the perturbation order
  const int N_order = par["N_ORDER"].template as<int>();
  for (int flavor=0; flavor<FLAVORS; ++flavor) {
    std::vector<double> order_creation_meas(FLAVORS*N_order, 0.0);
    if (order_creation_flavor[flavor]<N_order) {
      order_creation_meas[flavor*N_order+order_creation_flavor[flavor]] = 1.0;
    }
    measurements["order"] << order_creation_meas;
  }

  // measure acceptance rate
  measurements["Insertion_attempted"] << to_std_vector(weight_vs_distance.get_counter());
  measurements["Shift_attempted"] << to_std_vector(weight_vs_distance_shift.get_counter());
  measurements["Insertion_accepted"] << to_std_vector(weight_vs_distance.get_sumval());
  measurements["Shift_accepted"] << to_std_vector(weight_vs_distance_shift.get_sumval());
  weight_vs_distance.reset();
  weight_vs_distance.reset();

  //measure acceptance rate of global shift
  if (global_shift_acc_rate.has_samples()) {
    measurements["Acceptance_rate_global_shift"] << global_shift_acc_rate.compute_acceptance_rate();
    global_shift_acc_rate.reset();
  }

  //Measure <n>
#ifndef NDEBUG
  const typename SlidingWindowManager<IMP_MODEL>::state_t state_bak = sliding_window.get_state();
#endif
  measure_n();
  assert(sliding_window.get_state()==state_bak);

  //Measure <n>
  measure_two_time_correlation_functions();

  //Measure Legendre coefficients of single-particle Green's function
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

  //Measure single-particle Green's function
  for (std::vector<COMPLEX>::iterator it=G_meas_new.begin(); it!=G_meas_new.end(); ++it) {
    *it *= (1.*N)/(BETA*BETA*N_meas);
  }
  measure_simple_vector_observable<COMPLEX>(measurements, "Greens_rotated", G_meas_new);
  const typename IMP_MODEL::matrix_t& rotmat_Delta = p_model->get_rotmat_Delta();
  const typename IMP_MODEL::matrix_t inv_rotmat_Delta = rotmat_Delta.adjoint();
  transform_G_back_to_original_basis<typename IMP_MODEL::matrix_t, typename IMP_MODEL::complex_matrix_t, std::complex<double> >
    (FLAVORS, SITES, SPINS, Np1, rotmat_Delta, inv_rotmat_Delta, G_meas_new);
  measure_simple_vector_observable<COMPLEX>(measurements, "Greens", G_meas_new);

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
}

//Measure the expectation values of density operators
template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::measure_n() {
  assert(is_thermalized());
  MeasStaticObs<SlidingWindowManager<IMP_MODEL>,CdagC> meas(sliding_window, operators);
  std::vector<CdagC> ops(FLAVORS);
  std::vector<SCALAR> result_meas(FLAVORS);
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
  for (int flavor=0; flavor<FLAVORS; ++flavor) {
    result_meas_Re[flavor] = get_real(result_meas[flavor]*sign/trace);
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

  boost::multi_array<std::complex<double>,2> result;
  p_meas_corr->perform_meas(sliding_window, operators, result);
  //measure_simple_multiarray_observable<SCALAR,2>(measurements, "Two_time_correlation_functions", result);
  measure_simple_vector_observable<COMPLEX>(measurements, "Two_time_correlation_functions", to_std_vector(result));
}

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::expensive_updates() {
  const int N_swap = par["N_SWAP"].template as<int>();
  const bool do_swap = (N_swap != 0 && sweeps%N_swap == 0 && swap_vector.size() > 0);
  const bool do_global_shift = (sliding_window.get_position_right_edge()==0);

  if (!do_swap && !do_global_shift) {
    return;
  }

  const std::size_t n_sliding_window_bak = sliding_window.get_n_window();
  sliding_window.set_window_size(1, operators);

  //Swap flavors
  if (do_swap) {
    //std::cout << "doing swap update " << std::endl;
    for (int i = 0; i < swap_vector.size(); i += 2) {
      int j1 = swap_vector[i];
      int j2 = swap_vector[i + 1];
      swap_flavors(random, det, BETA, creation_operators,
                   annihilation_operators,
                   order_creation_flavor, order_annihilation_flavor,
                   M, sign, trace, operators, j1, j2,
                   sliding_window);
    }
  }

  //Shift operators to restore translational symmetry
  if (do_global_shift) {
    const bool accepted = global_shift(random, det, BETA, creation_operators,
                                       annihilation_operators,
                                       order_creation_flavor, order_annihilation_flavor,
                                       M, sign, trace, operators, sliding_window);
    if (accepted) {
      global_shift_acc_rate.accepted();
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

  //const long interval = std::max(par["THERMALIZATION"]/par["N_UPDATE_CUTOFF"],1);
  const int N_update_cutoff = par["N_UPDATE_CUTOFF"].template as<int>();
  const long interval_update_cutoff = static_cast<long>(std::max(static_cast<double>(par["THERMALIZATION"].template as<long>())/N_update_cutoff, 1.0));

  const double mag = std::pow(
    2.0,
    1.0/std::max(static_cast<double>(sweeps)/static_cast<double>(interval_update_cutoff),1.0)
  );

  boost::tuple<bool,double> r_pair = weight_vs_distance.update_cutoff(acc_rate_cutoff, max_distance_pair, mag, comm);
  max_distance_pair = boost::get<1>(r_pair);
  max_distance_pair = std::min(0.5*BETA, max_distance_pair);

  //std::cout << "Done max_distance_pair rank " << comm.rank() << " sweeps " << sweeps << std::endl;

  boost::tuple<bool,double> r_shift = weight_vs_distance_shift.update_cutoff(acc_rate_cutoff, max_distance_shift, mag, comm);
  max_distance_shift = boost::get<1>(r_shift);
  max_distance_shift = std::min(0.5*BETA, max_distance_shift);

  //std::cout << "Done max_distance_shift rank " << comm.rank() << " sweeps " << sweeps << std::endl;

  const double max_distance = std::max(max_distance_pair,max_distance_shift);
  const std::size_t n_window_new = static_cast<std::size_t>(std::max(1,static_cast<int>(BETA/(2.0*max_distance))));

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
  max_dist_optimizer.reset();
  weight_vs_distance.reset();
  weight_vs_distance_shift.reset();
  //std::cout << "Call prepare_for_measurement rank " << comm.rank() << std::endl;
  if (comm.rank()==0) {
    std::cout << "We're done with thermalization." << std::endl << "The number of segments for sliding window update is " << sliding_window.get_n_window() << "." << std::endl << std::endl;
  }

  //N_meas
  const int N_meas_min = std::max(10, 4*sliding_window.get_n_window());//a sweep of the window takes 4*get_n_window()
  if (N_meas<N_meas_min) {
    N_meas = N_meas_min;
    if (comm.rank()==0) {
      std::cout << "Warning N_MEAS is too small: using N_MEAS = "  << N_meas << " instead." << std::endl;
    }
  }

  //N_swap
  const int N_swap_min =std::max(10, 4*sliding_window.get_n_window());//a sweep of the window takes 4*get_n_window()
  if (N_swap<N_swap_min && N_swap>0) {
    N_swap = N_swap_min;
    if (comm.rank()==0) {
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
    operator_container_t c_ops_new, a_ops_new;
    int itmp = 0;
    for (operator_container_t::iterator it = operators.begin(); it!=operators.end(); ++it) {
      if (it->type()==0) {
        c_ops_new.insert(*it);
      } else {
        a_ops_new.insert(*it);
      }
    }
    assert(c_ops_new==creation_operators);
    assert(a_ops_new==annihilation_operators);
    if (c_ops_new!=creation_operators)
      throw std::logic_error("creation_operators is not consistent with operators.");
    if (a_ops_new!=annihilation_operators)
      throw std::logic_error("annihilation_operators is not consistent with operators.");
  }

  // compute determinants
  SCALAR det_new = cal_det(creation_operators, annihilation_operators, M_new, BETA, p_model->get_F());

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
  SCALAR trace_sw = sliding_window.compute_trace(operators_new);

  if (std::abs(trace_sw/trace-1.0)>1E-5) {
    throw std::runtime_error("trace != trace_new");
  }

  if (std::abs(det/det_new-1.0)>1E-5) {
    throw std::runtime_error("det_new != det");
  }

  SCALAR sign_overall_new = dsign(sign_new)*dsign(trace)*dsign(det_new);
  //std::cout << "debug sign "<< sign_overall_new << " " << sign << std::endl;
  if (std::abs(sign_overall_new/sign-1.0)>1E-5) {
    throw std::runtime_error("sign_overall_new != sign");
  }
#endif
}
