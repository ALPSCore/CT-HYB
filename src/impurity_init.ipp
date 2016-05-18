#include "impurity.hpp"

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
  measurements << alps::accumulators::NoBinningAccumulator<std::vector<double> >("Acceptance_rate_swap");

  measurements << alps::accumulators::NoBinningAccumulator<std::vector<double> >("Timings");
}

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::resize_vectors() {

  order_creation_flavor.resize(FLAVORS);
  order_annihilation_flavor.resize(FLAVORS);

  M.clear();

  swap_vector.resize(0);
  if (par.exists("SWAP_VECTOR")) {
    std::vector<int> flavors_vector;
    std::stringstream swapstream(par["SWAP_VECTOR"].template as<std::string>());
    int f;
    while (swapstream >> f) {
      if (f >= FLAVORS || f < 0) {
        std::cerr << "Out of range in SWAP_VECTOR:  << " << f << std::endl;
        abort();
      }
      flavors_vector.push_back(f);
    }

    if (flavors_vector.size() % FLAVORS != 0) {
      std::cerr << "The number of elements in SWAP_VECTOR is wrong! " << std::endl;
      exit(1);
    }

    const int num_templates = flavors_vector.size()/FLAVORS;
    std::vector<int>::iterator it = flavors_vector.begin();
    std::set<std::vector<int> > updates_set;//no duplication
    std::map<std::vector<int>, int> source_templates;
    for (int itemplate = 0; itemplate < num_templates; ++itemplate) {
      if (std::set<int>(it, it+FLAVORS).size()<FLAVORS) {
        std::cerr << "Duplicate elements in the definition of the " << itemplate+1 << "-th update in SWAP_VECTOR! " << std::endl;
        exit(1);
      }

      std::vector<int> tmp_vec(it, it+FLAVORS);
      std::vector<int> tmp_vec_rev(FLAVORS);
      for (int flavor=0; flavor<FLAVORS; ++flavor) {
        tmp_vec_rev[tmp_vec[flavor]] = flavor;
      }
      updates_set.insert(tmp_vec);
      updates_set.insert(tmp_vec_rev);//reverse process
      source_templates[tmp_vec] = itemplate;
      source_templates[tmp_vec_rev] = itemplate;

      it += FLAVORS;
    }

    for (std::set<std::vector<int> >::iterator it = updates_set.begin(); it != updates_set.end(); ++it) {
      swap_vector.push_back(std::make_pair(*it, source_templates[*it]));
    }
    swap_acc_rate.resize(num_templates);

    if (comm.rank() == 0) {
      std::cout << "The following swap updates will be performed." << std::endl;
      for (int i=0; i<swap_vector.size(); ++i) {
        std::cout << "Update #" << i << " generated from template #" << swap_vector[i].second << std::endl;
        for (int j=0; j<swap_vector[i].first.size(); ++j) {
          std::cout << "flavor " << j << " to flavor " << swap_vector[i].first[j] << std::endl;
        }
      }
    }
  }

  //////////////////INITIALIZE SHIFT PROB FOR FLAVORS//////////////////
  if (par.exists("N_SHIFT_FLAVOR")) {
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

  const std::string fname_key = "EQUAL_TIME_TWO_PARTICLE_GREENS_FUNCTION";
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
