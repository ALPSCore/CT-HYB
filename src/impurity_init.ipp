template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::create_observables() {
  // create measurement objects
  create_observable<COMPLEX,SimpleRealVectorObservable>(measurements, "Greens_rotated");
  create_observable<COMPLEX,SimpleRealVectorObservable>(measurements, "Greens");
  create_observable<COMPLEX,SimpleRealVectorObservable>(measurements, "Greens_legendre");
  create_observable<COMPLEX,SimpleRealVectorObservable>(measurements, "Greens_legendre_rotated");
  create_observable<COMPLEX,SimpleRealVectorObservable>(measurements, "Two_time_correlation_functions");

  measurements << alps::accumulators::LogBinningAccumulator<std::vector<double> >("n");
  measurements << alps::accumulators::LogBinningAccumulator<double>("Sign");
  measurements << alps::accumulators::NoBinningAccumulator<std::vector<double> >("order");

  measurements << alps::accumulators::NoBinningAccumulator<std::vector<double> >("Insertion_attempted");
  measurements << alps::accumulators::NoBinningAccumulator<std::vector<double> >("Insertion_accepted");
  measurements << alps::accumulators::NoBinningAccumulator<std::vector<double> >("Shift_attempted");
  measurements << alps::accumulators::NoBinningAccumulator<std::vector<double> >("Shift_accepted");

  measurements << alps::accumulators::NoBinningAccumulator<double>("Acceptance_rate_global_shift");
}

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::resize_vectors() {

  order_creation_flavor.resize(FLAVORS);
  order_annihilation_flavor.resize(FLAVORS);

  M.clear();

  swap_vector.clear();
  if (par.defined("SWAP_VECTOR")) {
    std::stringstream swapstream(par["SWAP_VECTOR"].template as<std::string>());
    int f;
    while (swapstream >> f) {
      swap_vector.push_back(f);
    }
    if (f >= FLAVORS) {
      std::cerr << "Swap vector: trying to swap orbital with orbital " << f << "which is > FLAVOR " << FLAVORS <<
      " exiting." << std::endl;
      abort();
    }
  }
  if (swap_vector.size() % 2 != 0) {
    std::cerr << "illegal SWAP_VECTOR! " << std::endl;
    exit(1);
  }

  //////////////////INITIALIZE SHIFT PROB FOR FLAVORS//////////////////
  if (par.defined("N_SHIFT_FLAVOR")) {
    std::string nsf(par["N_SHIFT_FLAVOR"].template as<std::string>());
    std::stringstream nsf_strstream(nsf);
    for (int i = 0; i < FLAVORS; ++i) {
      std::cout << "adjusting shift probability for flavor: " << i << " ";
      nsf_strstream >> N_shift_flavor[i];
      std::cout << N_shift_flavor[i] << std::endl;
    }
  }
}

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::read_eq_time_two_particle_greens_meas() {
  const int GF_RANK=2;

  const std::string fname_key = "EQUAL_TIME_TWO_PARTICLE_GREEENS_FUNCTION";
  const bool verbose = (comm.rank()==0);

  if (!par.defined(fname_key)) {
    return;
  }

  std::ifstream infile_f(boost::lexical_cast<std::string>(par[fname_key]).c_str());
  if(!infile_f.is_open()) {
    std::cerr<<"We cannot open "<<par[fname_key]<<"!"<<std::endl;
    exit(1);
  }

  int num_elem;
  infile_f >> num_elem;
  if (num_elem<0) {
    std::runtime_error("The number of Green's functions in "+fname_key+" cannot be negative!");
  }
  if (verbose) {
    std::cout << "The number of Green's functions is " << num_elem << std::endl;
  }

  eq_time_two_particle_greens_meas.reserve(num_elem);
  for (int i_elem=0; i_elem<num_elem; ++i_elem) {
    int line;
    infile_f >> line;
    if (line != i_elem) {
      throw std::runtime_error(boost::str(boost::format("First column of line %1% is incorrect.") % i_elem));
    }

    boost::array<int,2*GF_RANK> flavors_array;
    for (int iflavor=0; iflavor<2*GF_RANK; ++iflavor) {
      int flavor_tmp;
      infile_f >> flavor_tmp;
      if (flavor_tmp < 0 || flavor_tmp >= FLAVORS) {
        throw std::runtime_error(boost::str(boost::format("Column %1% of line %2% is incorrect.")%(iflavor+2)%i_elem));
      }
      flavors_array[iflavor] = flavor_tmp;
    }
    eq_time_two_particle_greens_meas.push_back(EqualTimeOperator<2>(flavors_array));
  }
}

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::read_two_time_correlation_functions() {
  const std::string fname_key = "TWO_TIME_CORRELATION_FUNCTIONS";
  const bool verbose = (comm.rank()==0);

  if (!par.defined("N_TAU_TWO_TIME_CORRELATION_FUNCTIONS")) {
    return;
  }
  const int num_tau_points = par["N_TAU_TWO_TIME_CORRELATION_FUNCTIONS"];
  if (num_tau_points<2 || !par.defined(fname_key)) {
    return;
  }

  std::ifstream infile_f(boost::lexical_cast<std::string>(par[fname_key]).c_str());
  if(!infile_f.is_open()) {
    std::cerr<<"We cannot open "<<par[fname_key]<<"!"<<std::endl;
    exit(1);
  }

  int num_elem;
  infile_f >> num_elem;
  if (num_elem<0) {
    std::runtime_error("The number of two-time correlation functions in "+fname_key+" cannot be negative!");
  }
  if (verbose) {
    std::cout << "The number of two-time correlation functions is " << num_elem << std::endl;
  }

  std::vector<std::pair<EqualTimeOperator<1>,EqualTimeOperator<1> > > corr_meas;
  corr_meas.reserve(num_elem);
  for (int i_elem=0; i_elem<num_elem; ++i_elem) {
    int line;
    infile_f >> line;
    if (line != i_elem) {
      throw std::runtime_error(boost::str(boost::format("First column of line %1% is incorrect.") % i_elem));
    }

    boost::array<int,4> flavors_array;
    for (int iflavor=0; iflavor<4; ++iflavor) {
      int flavor_tmp;
      infile_f >> flavor_tmp;
      if (flavor_tmp < 0 || flavor_tmp >= FLAVORS) {
        throw std::runtime_error(boost::str(boost::format("Column %1% of line %2% is incorrect.")%(iflavor+2)%i_elem));
      }
      flavors_array[iflavor] = flavor_tmp;
    }
    corr_meas.push_back(
      std::make_pair(
        EqualTimeOperator<1>(flavors_array.data()),
        EqualTimeOperator<1>((flavors_array.data()+2))
      )
    );
  }
  assert(corr_meas.size()==num_elem);

  p_meas_corr.reset(
    new MeasCorrelation<SW_TYPE,EqualTimeOperator<1> >(
      corr_meas, num_tau_points
    )
  );
}
