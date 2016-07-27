#include "impurity.hpp"

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::create_observables() {
  // create measurement objects
  //create_observable<COMPLEX, SimpleRealVectorObservable>(measurements, "Greens_rotated");
  //create_observable<COMPLEX, SimpleRealVectorObservable>(measurements, "Greens");
  create_observable<COMPLEX, SimpleRealVectorObservable>(measurements, "Greens_legendre");
  create_observable<COMPLEX, SimpleRealVectorObservable>(measurements, "Greens_legendre_rotated");
  create_observable<COMPLEX, SimpleRealVectorObservable>(measurements, "Two_time_correlation_functions");

  measurements << alps::accumulators::LogBinningAccumulator<std::vector<double> >("n");
  measurements << alps::accumulators::LogBinningAccumulator<double>("Sign");
  measurements << alps::accumulators::NoBinningAccumulator<std::vector<double> >("order");
  measurements << alps::accumulators::NoBinningAccumulator<std::vector<double> >("PerturbationOrderFlavors");

  for (int k = 1; k < par["RANK_INSERTION_REMOVAL_UPDATE"].template as<int>() + 1; ++k) {
    ins_rem_diagonal_updater[k - 1]->create_measurement_acc_rate(measurements);
  }
  single_op_shift_updater.create_measurement_acc_rate(measurements);
  operator_pair_flavor_updater.create_measurement_acc_rate(measurements);

  measurements << alps::accumulators::NoBinningAccumulator<double>("Acceptance_rate_global_shift");
  measurements << alps::accumulators::NoBinningAccumulator<std::vector<double> >("Acceptance_rate_swap");

  measurements << alps::accumulators::NoBinningAccumulator<double>("Z_function_space_volume");
  measurements << alps::accumulators::NoBinningAccumulator<double>("Z_function_space_num_steps");
  for (int w = 0; w < worm_names.size(); ++w) {
    measurements << alps::accumulators::NoBinningAccumulator<double>("worm_space_volume_" + worm_names[w]);
    measurements << alps::accumulators::NoBinningAccumulator<double>("worm_space_num_steps_" + worm_names[w]);
  }

  if (par["N_LEGENDRE_N2_MEASUREMENT"] > 0) {
    create_observable<COMPLEX, SimpleRealVectorObservable>(measurements, "N2_correlation_function");
  }
  create_observable<COMPLEX, SimpleRealVectorObservable>(measurements, "G1");

  if (par["N_MEASURE_EQUAL_TIME_G2"] > 0) {
    create_observable<COMPLEX, SimpleRealVectorObservable>(measurements, "Equal_time_G2");
  }

  //fidelity susceptibility
  create_observable<SCALAR, SimpleRealObservable>(measurements, "kLkR");
  create_observable<SCALAR, SimpleRealObservable>(measurements, "k");

  measurements << alps::accumulators::NoBinningAccumulator<double>("Pert_order_start");
  measurements << alps::accumulators::NoBinningAccumulator<double>("Pert_order_end");

#ifdef MEASURE_TIMING
  measurements << alps::accumulators::NoBinningAccumulator<std::vector<double> >("TimingsSecPerNMEAS");
#endif
}


template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::create_worm_updaters() {
  /*
   * Single-particle Green's function
   */
  worm_names.push_back("G1");
  worm_movers.push_back(
      boost::shared_ptr<WormMoverType>(
          new WormMoverType("G1", BETA, FLAVORS, 0.0, BETA)
      )
  );
  worm_insertion_removers.push_back(
      boost::shared_ptr<WormInsertionRemoverType>(
          new WormInsertionRemoverType(
              "G1", BETA, FLAVORS, 0.0, BETA,
              boost::shared_ptr<Worm>(new GWorm<1>("G1"))
          )
      )
  );
  p_G1_meas.reset(
      new GMeasurement<SCALAR, 1>(FLAVORS, par["N_LEGENDRE_MEASUREMENT"], BETA)
  );
  p_g1_worm_insertion_remover.reset(
      new G1WormInsertionRemoverType(
          "G1", BETA, FLAVORS, boost::shared_ptr<Worm>(new GWorm<1>("G1"))
      )
  );

  /*
   * Generalized spin-spin correlations
   */
  if (par["N_LEGENDRE_N2_MEASUREMENT"] > 0) {
    worm_names.push_back("N2_correlation");
    worm_movers.push_back(
        boost::shared_ptr<WormMoverType>(
            new WormMoverType("N2_correlation", BETA, FLAVORS, 0.0, BETA)
        )
    );
    worm_insertion_removers.push_back(
        boost::shared_ptr<WormInsertionRemoverType>(
            new WormInsertionRemoverType(
                "N2_correlation", BETA, FLAVORS, 0.0, BETA,
                boost::shared_ptr<Worm>(new CorrelationWorm<2>("N2_correlation"))
            )
        )
    );
    p_N2_meas.reset(
        new N2CorrelationFunctionMeasurement<SCALAR>(FLAVORS, par["N_LEGENDRE_N2_MEASUREMENT"], BETA)
    );
  }

  /*
   * Equal-time two-particle Green's function
   */
  if (par["N_MEASURE_EQUAL_TIME_G2"] > 0) {
    const std::string name("Equal_time_G2");
    worm_names.push_back(name);
    worm_movers.push_back(
        boost::shared_ptr<WormMoverType>(
            new WormMoverType(name, BETA, FLAVORS, 0.0, BETA)
        )
    );
    worm_insertion_removers.push_back(
        boost::shared_ptr<WormInsertionRemoverType>(
            new WormInsertionRemoverType(
                name, BETA, FLAVORS, 0.0, BETA,
                boost::shared_ptr<Worm>(new EqualTimeGWorm<2>(name))
            )
        )
    );
    p_equal_time_G2_meas.reset(
        new EqualTimeGMeasurement<SCALAR, 2>(FLAVORS)
    );
  }

  //Proposal probability of worm insertion is smaller than that of removal by the number of active worm spaces.
  //We correct this here.
  for (int w = 0; w < worm_names.size(); ++w) {
    worm_insertion_removers[w]->set_relative_insertion_proposal_rate(1.0 / worm_names.size());
  }

  //if we have active worm spaces, we activate flat histogram algorithm.
  if (worm_names.size() > 0) {
    p_flat_histogram_config_space.reset(new FlatHistogram(worm_names.size()));
  }
  config_space_extra_weight.resize(0);
  config_space_extra_weight.resize(worm_names.size() + 1, 1.0);
  for (int w = 0; w < worm_names.size(); ++w) {
    worm_insertion_removers[w]->set_worm_space_weight(config_space_extra_weight[w + 1] / config_space_extra_weight[0]);
  }
  num_steps_in_config_space.resize(0);
  num_steps_in_config_space.resize(worm_names.size() + 1);
}

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::resize_vectors() {
  {
    swap_vector.resize(0);
    std::string input_str(par["SWAP_VECTOR"].template as<std::string>());
    //When SPINS==2, a global spin flip is pre-defined
    if (SPINS == 2) {
      for (int site = 0; site < SITES; ++site) {
        input_str += " " + boost::lexical_cast<std::string>(2 * site + 1);
        input_str += " " + boost::lexical_cast<std::string>(2 * site);
      }
    }
    std::vector<int> flavors_vector;
    std::stringstream swapstream(input_str);
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

    const int num_templates = flavors_vector.size() / FLAVORS;
    std::vector<int>::iterator it = flavors_vector.begin();
    std::set<std::vector<int> > updates_set;//no duplication
    std::map<std::vector<int>, int> source_templates;
    for (int itemplate = 0; itemplate < num_templates; ++itemplate) {
      if (std::set<int>(it, it + FLAVORS).size() < FLAVORS) {
        std::cerr << "Duplicate elements in the definition of the " << itemplate + 1 << "-th update in SWAP_VECTOR! "
            << std::endl;
        exit(1);
      }

      std::vector<int> tmp_vec(it, it + FLAVORS);
      std::vector<int> tmp_vec_rev(FLAVORS);
      for (int flavor = 0; flavor < FLAVORS; ++flavor) {
        tmp_vec_rev[tmp_vec[flavor]] = flavor;
      }
      updates_set.insert(tmp_vec);
      updates_set.insert(tmp_vec_rev);//reverse process
      source_templates[tmp_vec] = itemplate;
      source_templates[tmp_vec_rev] = itemplate;

      it += FLAVORS;
    }

    //remove the update which does nothing
    {
      std::vector<int> identity;
      for (int flavor = 0; flavor < FLAVORS; ++flavor) {
        identity.push_back(flavor);
      }
      updates_set.erase(identity);
    }

    for (std::set<std::vector<int> >::iterator it = updates_set.begin(); it != updates_set.end(); ++it) {
      swap_vector.push_back(std::make_pair(*it, source_templates[*it]));
    }
    swap_acc_rate.resize(swap_vector.size());

    if (global_mpi_rank == 0) {
      std::cout << "The following swap updates will be performed." << std::endl;
      for (int i = 0; i < swap_vector.size(); ++i) {
        std::cout << "Update #" << i << " generated from template #" << swap_vector[i].second << std::endl;
        for (int j = 0; j < swap_vector[i].first.size(); ++j) {
          std::cout << "flavor " << j << " to flavor " << swap_vector[i].first[j] << std::endl;
        }
      }
    }
  }
}

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::read_eq_time_two_particle_greens_meas() {
  const int GF_RANK = 2;

  const std::string fname_key = "EQUAL_TIME_TWO_PARTICLE_GREENS_FUNCTION";
  const bool verbose = (global_mpi_rank == 0);

  if (!par.defined(fname_key)) {
    return;
  }

  std::ifstream infile_f(boost::lexical_cast<std::string>(par[fname_key]).c_str());
  if (!infile_f.is_open()) {
    std::cerr << "We cannot open " << par[fname_key] << "!" << std::endl;
    exit(1);
  }

  int num_elem;
  infile_f >> num_elem;
  if (num_elem < 0) {
    std::runtime_error("The number of Green's functions in " + fname_key + " cannot be negative!");
  }
  if (verbose) {
    std::cout << "The number of Green's functions is " << num_elem << std::endl;
  }

  eq_time_two_particle_greens_meas.reserve(num_elem);
  for (int i_elem = 0; i_elem < num_elem; ++i_elem) {
    int line;
    infile_f >> line;
    if (line != i_elem) {
      throw std::runtime_error(boost::str(boost::format("First column of line %1% is incorrect.") % i_elem));
    }

    boost::array<int, 2 * GF_RANK> flavors_array;
    for (int iflavor = 0; iflavor < 2 * GF_RANK; ++iflavor) {
      int flavor_tmp;
      infile_f >> flavor_tmp;
      if (flavor_tmp < 0 || flavor_tmp >= FLAVORS) {
        throw std::runtime_error(boost::str(
            boost::format("Column %1% of line %2% is incorrect.") % (iflavor + 2) % i_elem));
      }
      flavors_array[iflavor] = flavor_tmp;
    }
    eq_time_two_particle_greens_meas.push_back(EqualTimeOperator<2>(flavors_array));
  }
}

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::read_two_time_correlation_functions() {
  const std::string fname_key = "TWO_TIME_CORRELATION_FUNCTIONS";
  const bool verbose = (global_mpi_rank == 0);

  if (!par.defined("N_TAU_TWO_TIME_CORRELATION_FUNCTIONS")) {
    return;
  }
  const int num_tau_points = par["N_TAU_TWO_TIME_CORRELATION_FUNCTIONS"];
  if (num_tau_points < 2 || !par.defined(fname_key)) {
    return;
  }

  if (par[fname_key].template as<std::string>() == "") {
    return;
  }

  std::ifstream infile_f(boost::lexical_cast<std::string>(par[fname_key]).c_str());
  if (!infile_f.is_open()) {
    std::cerr << "We cannot open " << par[fname_key] << "!" << std::endl;
    exit(1);
  }

  int num_elem;
  infile_f >> num_elem;
  if (num_elem < 0) {
    std::runtime_error("The number of two-time correlation functions in " + fname_key + " cannot be negative!");
  }
  if (verbose) {
    std::cout << "The number of two-time correlation functions is " << num_elem << std::endl;
  }

  std::vector<std::pair<EqualTimeOperator<1>, EqualTimeOperator<1> > > corr_meas;
  corr_meas.reserve(num_elem);
  for (int i_elem = 0; i_elem < num_elem; ++i_elem) {
    int line;
    infile_f >> line;
    if (line != i_elem) {
      throw std::runtime_error(boost::str(boost::format("First column of line %1% is incorrect.") % i_elem));
    }

    boost::array<int, 4> flavors_array;
    for (int iflavor = 0; iflavor < 4; ++iflavor) {
      int flavor_tmp;
      infile_f >> flavor_tmp;
      if (flavor_tmp < 0 || flavor_tmp >= FLAVORS) {
        throw std::runtime_error(boost::str(
            boost::format("Column %1% of line %2% is incorrect.") % (iflavor + 2) % i_elem));
      }
      flavors_array[iflavor] = flavor_tmp;
    }
    corr_meas.push_back(
        std::make_pair(
            EqualTimeOperator<1>(flavors_array.data()),
            EqualTimeOperator<1>((flavors_array.data() + 2))
        )
    );
  }
  assert(corr_meas.size() == num_elem);

  p_meas_corr.reset(
      new MeasCorrelation<SW_TYPE, EqualTimeOperator<1> >(
          corr_meas, num_tau_points
      )
  );
}
