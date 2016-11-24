#include "impurity.hpp"

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::create_observables() {
  // create measurement objects
  create_observable<COMPLEX, SimpleRealVectorObservable>(measurements, "Greens_legendre");
  create_observable<COMPLEX, SimpleRealVectorObservable>(measurements, "Greens_legendre_rotated");
  create_observable<COMPLEX, SimpleRealVectorObservable>(measurements, "Two_time_correlation_functions");

  measurements << alps::accumulators::LogBinningAccumulator<std::vector<double> >("n");
  measurements << alps::accumulators::LogBinningAccumulator<double>("Sign");
  measurements << alps::accumulators::NoBinningAccumulator<std::vector<double> >("order");
  measurements << alps::accumulators::NoBinningAccumulator<std::vector<double> >("PerturbationOrderFlavors");

  for (int k = 1; k < par["update.multi_pair_ins_rem"].template as<int>() + 1; ++k) {
    ins_rem_updater[k - 1]->create_measurement_acc_rate(measurements);
  }
  single_op_shift_updater.create_measurement_acc_rate(measurements);
  operator_pair_flavor_updater.create_measurement_acc_rate(measurements);

  measurements << alps::accumulators::NoBinningAccumulator<double>("Acceptance_rate_global_shift");
  measurements << alps::accumulators::NoBinningAccumulator<std::vector<double> >("Acceptance_rate_swap");

  measurements << alps::accumulators::NoBinningAccumulator<double>("Z_function_space_volume");
  measurements << alps::accumulators::NoBinningAccumulator<double>("Z_function_space_num_steps");
  for (int w = 0; w < worm_types.size(); ++w) {
    measurements << alps::accumulators::NoBinningAccumulator<double>(
        "worm_space_volume_" + get_config_space_name(worm_types[w]));
    measurements << alps::accumulators::NoBinningAccumulator<double>(
        "worm_space_num_steps_" + get_config_space_name(worm_types[w]));
  }

  if (par["measurement.two_time_G2.on"] != 0) {
    create_observable<COMPLEX, SimpleRealVectorObservable>(measurements, "Two_time_G2");
  }
  if (p_G1_meas) {
    p_G1_meas->create_alps_observable(measurements);
  }
  if (p_G2_meas) {
    p_G2_meas->create_alps_observable(measurements);
  }

  create_observable<COMPLEX, SimpleRealVectorObservable>(measurements, "Equal_time_G1");

  if (par["measurement.equal_time_G2.on"] != 0) {
    create_observable<COMPLEX, SimpleRealVectorObservable>(measurements, "Equal_time_G2");
  }

  //Fidelity susceptibility
  create_observable<SCALAR, SimpleRealObservable>(measurements, "kLkR");
  create_observable<SCALAR, SimpleRealObservable>(measurements, "k");

  measurements << alps::accumulators::NoBinningAccumulator<double>("Pert_order_start");
  measurements << alps::accumulators::NoBinningAccumulator<double>("Pert_order_end");

  //Acceptance rate of worm updates
  for (int i = 0; i < worm_insertion_removers.size(); ++i) {
    worm_insertion_removers[i]->create_measurement_acc_rate(measurements);
  }
  for (typename worm_updater_map_t::iterator it = worm_movers.begin(); it != worm_movers.end();
       ++it) {
    it->second->create_measurement_acc_rate(measurements);
  }
  for (typename std::map<std::string, boost::shared_ptr<LocalUpdaterType> >::iterator
           it = specialized_updaters.begin(); it != specialized_updaters.end(); ++it) {
    it->second->create_measurement_acc_rate(measurements);
  }

#ifdef MEASURE_TIMING
  measurements << alps::accumulators::NoBinningAccumulator<std::vector<double> >("TimingsSecPerNMEAS");
#endif
}

template<typename IMP_MODEL>
template<typename W>
void HybridizationSimulation<IMP_MODEL>::add_worm_mover(ConfigSpace config_space,
                    const std::string &updater_name
) {
  worm_movers.insert(
      std::make_pair(
          config_space,
          boost::shared_ptr<WormUpdaterType>(
              new W(updater_name, BETA, FLAVORS, 0.0, BETA)
          )
      )
  );
}

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::create_worm_updaters() {
  /*
   * G1
   */
  worm_types.push_back(G1);
  add_worm_mover<WormMoverType>(G1, "G1_mover");
  add_worm_mover<WormFlavorChangerType>(G1, "G1_flavor_changer");
  worm_insertion_removers.push_back(
      boost::shared_ptr<WormInsertionRemoverType>(
          new WormInsertionRemoverType(
              "G1_ins_rem", BETA, FLAVORS, 0.0, BETA,
              boost::shared_ptr<Worm>(new GWorm<1>())
          )
      )
  );
  //Via connecting or cutting hybridization lines
  specialized_updaters["G1_ins_rem_hyb"] =
      boost::shared_ptr<LocalUpdaterType>(
          new G1WormInsertionRemoverType(
              "G1_ins_rem_hyb", BETA, FLAVORS, boost::shared_ptr<Worm>(new GWorm<1>())
          )
      );
  p_G1_meas.reset(
      new GMeasurement<SCALAR, 1>(FLAVORS, par["measurement.G1.n_legendre"], 0, BETA,
                                  par["measurement.G1.max_num_data_accumulated"])
  );

  /*
   * G2
   */
  if (par["measurement.G2.on"] != 0) {
    worm_types.push_back(G2);
    add_worm_mover<WormMoverType>(G2, "G2_mover");
    add_worm_mover<WormFlavorChangerType>(G2, "G2_flavor_changer");
    worm_insertion_removers.push_back(
        boost::shared_ptr<WormInsertionRemoverType>(
            new WormInsertionRemoverType(
                "G2_ins_rem", BETA, FLAVORS, 0.0, BETA,
                boost::shared_ptr<Worm>(new GWorm<2>())
            )
        )
    );
    p_G2_meas.reset(
        new GMeasurement<SCALAR, 2>(FLAVORS,
                                    par["measurement.G2.n_legendre"], par["measurement.G2.n_bosonic_freq"], BETA,
                                    par["measurement.G2.max_num_data_accumulated"]
        )
    );
    specialized_updaters["G2_ins_rem_hyb"] =
        boost::shared_ptr<LocalUpdaterType>(
            new G2WormInsertionRemoverType(
                "G2_ins_rem_hyb", BETA, FLAVORS, boost::shared_ptr<Worm>(new GWorm<2>())
            )
        );
  }

  /*
   * Two-time G2
   */
  if (par["measurement.two_time_G2.on"] != 0) {
    worm_types.push_back(Two_time_G2);
    add_worm_mover<WormMoverType>(Two_time_G2, "Two_time_G2_mover");
    add_worm_mover<WormFlavorChangerType>(Two_time_G2, "Two_time_G2_flavor_changer");
    worm_insertion_removers.push_back(
        boost::shared_ptr<WormInsertionRemoverType>(
            new WormInsertionRemoverType(
                "Two_time_G2_ins_rem", BETA, FLAVORS, 0.0, BETA,
                boost::shared_ptr<Worm>(new CorrelationWorm<2>())
            )
        )
    );
    p_two_time_G2_meas.reset(
        new TwoTimeG2Measurement<SCALAR>(FLAVORS, par["measurement.two_time_G2.n_legendre"], BETA)
    );
  }

  /*
   * Equal-time G1
   */
  {
    const std::string name("Equal_time_G1");
    worm_types.push_back(Equal_time_G1);
    add_worm_mover<WormMoverType>(Equal_time_G1, name + "_mover");
    add_worm_mover<WormFlavorChangerType>(Equal_time_G1, name + "_flavor_changer");
    worm_insertion_removers.push_back(
        boost::shared_ptr<WormInsertionRemoverType>(
            new WormInsertionRemoverType(
                name + "_ins_rem", BETA, FLAVORS, 0.0, BETA,
                boost::shared_ptr<Worm>(new EqualTimeGWorm<1>())
            )
        )
    );
    p_equal_time_G1_meas.reset(
        new EqualTimeGMeasurement<SCALAR, 1>(FLAVORS)
    );
  }

  /*
   * Equal-time G2
   */
  if (par["measurement.equal_time_G2.on"] != 0) {
    const std::string name("Equal_time_G2");
    worm_types.push_back(Equal_time_G2);
    add_worm_mover<WormMoverType>(Equal_time_G2, name + "_mover");
    add_worm_mover<WormFlavorChangerType>(Equal_time_G2, name + "_flavor_changer");
    worm_insertion_removers.push_back(
        boost::shared_ptr<WormInsertionRemoverType>(
            new WormInsertionRemoverType(
                name + "_ins_rem", BETA, FLAVORS, 0.0, BETA,
                boost::shared_ptr<Worm>(new EqualTimeGWorm<2>())
            )
        )
    );
    p_equal_time_G2_meas.reset(
        new EqualTimeGMeasurement<SCALAR, 2>(FLAVORS)
    );
  }

  /*
   * Connect Equal_time_G1 and Two_time_G2 spaces
   */
  if (std::find(worm_types.begin(), worm_types.end(), Equal_time_G1) != worm_types.end() &&
      std::find(worm_types.begin(), worm_types.end(), Two_time_G2) != worm_types.end()) {
    specialized_updaters["Connect_Equal_time_G1_and_Two_time_G2"] =
        boost::shared_ptr<LocalUpdaterType>(
            new EqualTimeG1_TwoTimeG2_Connector<SCALAR, EXTENDED_SCALAR, SW_TYPE>(
                "Connect_Equal_time_G1_and_Two_time_G2", BETA, FLAVORS
            )
        );
  }

  if (std::find(worm_types.begin(), worm_types.end(), G1) != worm_types.end()) {
    specialized_updaters["G1_shifter_hyb"] =
        boost::shared_ptr<LocalUpdaterType>(
            new GWormShifter<SCALAR, 1, EXTENDED_SCALAR, SW_TYPE>(
                "G1_shifter_hyb", BETA, FLAVORS,
                boost::shared_ptr<Worm>(new GWorm<1>())
            )
        );
  }

  if (std::find(worm_types.begin(), worm_types.end(), G2) != worm_types.end()) {
    specialized_updaters["G2_shifter_hyb"] =
        boost::shared_ptr<LocalUpdaterType>(
            new GWormShifter<SCALAR, 2, EXTENDED_SCALAR, SW_TYPE>(
                "G2_shifter_hyb", BETA, FLAVORS,
                boost::shared_ptr<Worm>(new GWorm<2>())
            )
        );
  }

  //Proposal probability of worm insertion is smaller than that of removal by the number of active worm spaces.
  //We correct this here.
  for (int w = 0; w < worm_types.size(); ++w) {
    worm_insertion_removers[w]->set_relative_insertion_proposal_rate(1.0 / worm_types.size());
  }

  //if we have active worm spaces, we activate flat histogram algorithm.
  if (worm_types.size() > 0) {
    std::vector<double> target_fractions;
    target_fractions.push_back(1.0);//partition function space
    for (int w = 0; w < worm_types.size(); ++w) {
      //we should not spend too many Monte Carlo steps for measuring equal-time observables
      target_fractions.push_back(
          (worm_types[w] == Equal_time_G1 || worm_types[w] == Equal_time_G2) ? 0.1 : 1.0
      );
    }
    p_flat_histogram_config_space.reset(new FlatHistogram(worm_types.size(), target_fractions));
  }
  config_space_extra_weight.resize(0);
  config_space_extra_weight.resize(worm_types.size() + 1, 1.0);
  for (int w = 0; w < worm_types.size(); ++w) {
    worm_space_extra_weight_map[worm_types[w]] = 1.0;
  }
  num_steps_in_config_space.resize(0);
  num_steps_in_config_space.resize(worm_types.size() + 1);

  config_spaces_visited_in_measurement_steps.resize(0);
  config_spaces_visited_in_measurement_steps.resize(worm_types.size() + 1, false);
}

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::resize_vectors() {
  {
    swap_vector.resize(0);
    std::string input_str(par["update.swap_vector"].template as<std::string>());
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

    if (comm.rank() == 0) {
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
  const bool verbose = (comm.rank() == 0);

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
  const std::string fname_key = "measurement.nn_corr.def";
  const bool verbose = (comm.rank() == 0);

  if (!par.defined("measurement.nn_corr.n_tau")) {
    return;
  }
  const int num_tau_points = par["measurement.nn_corr.n_tau"];
  const int num_def = par["measurement.nn_corr.n_def"];
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
    throw std::runtime_error("The number of density-density correlation functions in " + fname_key + " cannot be negative!");
  }
  if (num_elem != num_def) {
    throw std::runtime_error("Inconsistent numbers of density-density correlation functions between the input file and the parameter");
  }
  if (verbose) {
    std::cout << "The number of density-density correlation functions is " << num_elem << std::endl;
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
    for (int iflavor = 0; iflavor < 2; ++iflavor) {
      int flavor_tmp;
      infile_f >> flavor_tmp;
      if (flavor_tmp < 0 || flavor_tmp >= FLAVORS) {
        throw std::runtime_error(boost::str(
            boost::format("Column %1% of line %2% is incorrect.") % (iflavor + 2) % i_elem));
      }
      flavors_array[2 * iflavor] = flavors_array[2 * iflavor + 1] = flavor_tmp;
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
