#include "impurity.hpp"


template<typename IMP_MODEL>
template<typename WORM_T>
void HybridizationSimulation<IMP_MODEL>::add_worm_space() {
  auto p_worm = std::shared_ptr<Worm>(new WORM_T());
  auto w = p_worm->get_config_space();
  check_true(
    std::find(worm_types.begin(), worm_types.end(), w) == worm_types.end(),
    "Duplicate worm space!"
  );
  worm_types.push_back(w);
  auto name = ConfigSpaceEnum::to_string(w);
  add_worm_mover<WormMoverType>(w, name + "_mover");
  add_worm_mover<WormFlavorChangerType>(w, name + "_flavor_changer");
  worm_insertion_removers.push_back(
      std::shared_ptr<WormInsertionRemoverType>(
          new WormInsertionRemoverType(
              name + "_ins_rem", BETA, FLAVORS, 0.0, BETA, p_worm
          )
      )
  );
}


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
        "worm_space_volume_" + ConfigSpaceEnum::to_string(worm_types[w]));
    measurements << alps::accumulators::NoBinningAccumulator<double>(
        "worm_space_num_steps_" + ConfigSpaceEnum::to_string(worm_types[w]));
  }

  if (par["measurement.two_time_G2.on"] != 0) {
    create_observable<COMPLEX, SimpleRealVectorObservable>(measurements, "Two_time_G2");
  }
  if (p_G1_legendre_meas) {
    p_G1_legendre_meas->create_alps_observable(measurements);
  }
  if (p_G2_legendre_meas) {
    p_G2_legendre_meas->create_alps_observable(measurements);
  }

  for (auto& w: worm_meas) {
    for (auto& elem: w.second) {
      elem.second->create_alps_observable(measurements);
    }
  }

  //Fidelity susceptibility
  create_observable<SCALAR, SimpleRealObservable>(measurements, "kLkR");
  create_observable<SCALAR, SimpleRealObservable>(measurements, "k");

  measurements << alps::accumulators::NoBinningAccumulator<double>("Pert_order_start");
  measurements << alps::accumulators::NoBinningAccumulator<double>("Pert_order_end");

  //Acceptance rate of worm updates
  for (auto i = 0; i < worm_insertion_removers.size(); ++i) {
    worm_insertion_removers[i]->create_measurement_acc_rate(measurements);
  }
  for (auto it = worm_movers.begin(); it != worm_movers.end();
       ++it) {
    it->second->create_measurement_acc_rate(measurements);
  }
  for (auto it = specialized_updaters.begin(); it != specialized_updaters.end(); ++it) {
    it->second->create_measurement_acc_rate(measurements);
  }

  measurements << alps::accumulators::NoBinningAccumulator<std::vector<double> >("TimingsSecPerNMEAS");
}

template<typename IMP_MODEL>
template<typename W>
void HybridizationSimulation<IMP_MODEL>::add_worm_mover(ConfigSpaceEnum::Type config_space,
                    const std::string &updater_name
) {
  worm_movers.insert(
      std::make_pair(
          config_space,
          std::shared_ptr<WormUpdaterType>(
              new W(updater_name, BETA, FLAVORS, 0.0, BETA)
          )
      )
  );
}


template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::create_worm_spaces() {
  auto target_worm_space = get_defined_worm_spaces(par)[par["target_worm_space"].template as<int>()];
  switch (target_worm_space)
  {
  case ConfigSpaceEnum::G1:
    add_worm_space<GWorm<1>>();
    break;

  case ConfigSpaceEnum::Equal_time_G1:
    add_worm_space<EqualTimeGWorm<1>>();
    break;

  case ConfigSpaceEnum::G2:
    add_worm_space<GWorm<2>>();
    break;
  
  case ConfigSpaceEnum::lambda:
    add_worm_space<TwoPointCorrWorm<PH_CHANNEL>>();
    break;

  case ConfigSpaceEnum::varphi:
    add_worm_space<TwoPointCorrWorm<PP_CHANNEL>>();
    break;

  case ConfigSpaceEnum::eta:
    add_worm_space<ThreePointCorrWorm<PH_CHANNEL,true>>();
    break;

  case ConfigSpaceEnum::gamma:
    add_worm_space<ThreePointCorrWorm<PP_CHANNEL,true>>();
    break;

  case ConfigSpaceEnum::vartheta:
    add_worm_space<GWorm<1,true>>();
    break;

  case ConfigSpaceEnum::h:
    add_worm_space<GWorm<2,true>>();
    break;

  default:
    throw std::logic_error("create_worm_spaces does not implement worm space " + ConfigSpaceEnum::to_string(target_worm_space)+"!");
  }
}


template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::create_custom_worm_updaters() {
  /*
   * G1
   */
  if (is_worm_space_active(ConfigSpaceEnum::G1)) {
    specialized_updaters["G1_ins_rem_hyb"] =
      std::shared_ptr<LocalUpdaterType>(
          new G1WormInsertionRemoverType(
              "G1_ins_rem_hyb", BETA, FLAVORS, std::shared_ptr<Worm>(new GWorm<1>())
          )
      );
    p_G1_legendre_meas.reset(
      new GMeasurement<SCALAR, 1>(FLAVORS, par["measurement.G1.n_legendre"], 0, BETA,
                                  par["measurement.G1.max_num_data_accumulated"])
    );
  }

  /*
   * G2
   */
  if (is_worm_space_active(ConfigSpaceEnum::G2)) {
    if (par["measurement.G2.legendre.on"].template as<int>() != 0) {
      /*
      p_G2_legendre_meas.reset(
          new GMeasurement<SCALAR, 2>(FLAVORS,
                                      par["measurement.G2.legendre.n_legendre"],
                                      par["measurement.G2.legendre.n_bosonic_freq"], BETA,
                                      par["measurement.G2.legendre.max_num_data_accumulated"]
          )
      );
      */
    }
    specialized_updaters["G2_ins_rem_hyb"] =
        std::shared_ptr<LocalUpdaterType>(
            new G2WormInsertionRemoverType(
                "G2_ins_rem_hyb", BETA, FLAVORS, std::shared_ptr<Worm>(new GWorm<2>())
            )
        );
  }

  //Proposal probability of worm insertion is smaller than that of removal by the number of active worm spaces.
  //We correct this here.
  for (int w = 0; w < worm_types.size(); ++w) {
    worm_insertion_removers[w]->set_relative_insertion_proposal_rate(1.0 / worm_types.size());
  }

  // Activate flat histogram algorithm
  {
    std::vector<double> target_fractions(worm_types.size()+1, 1.0);
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
void HybridizationSimulation<IMP_MODEL>::create_worm_meas() {
  for (auto w : ConfigSpaceEnum::AllWormSpaces) {
    if (!is_worm_space_active(w)) {
      continue;
    }
    worm_meas[w] = std::unordered_map<std::string,std::unique_ptr<WORM_MEAS_TYPE>>();
  }

  auto register_worm_meas = [this]
    (ConfigSpaceEnum::Type w, const std::string& meas_name, WormMeas<SCALAR,SW_TYPE> *p_meas) {
    if (this->worm_meas[w].find(meas_name) != this->worm_meas[w].end()) {
      throw std::logic_error("Duplicated worm_meas!");
    }
    this->worm_meas[w][meas_name] = std::unique_ptr<WORM_MEAS_TYPE>(p_meas);
  };

  // G1
  if (is_worm_space_active(ConfigSpaceEnum::G1)) {
    register_worm_meas(ConfigSpaceEnum::G1, "g1",
      new G1Meas<SCALAR,SW_TYPE>(&random, BETA, FLAVORS, get_wsample_f())
    );
  }
  
  // vartheta
  if (is_worm_space_active(ConfigSpaceEnum::vartheta)) {
    register_worm_meas(ConfigSpaceEnum::vartheta, "vartheta_legendre",
        new VarThetaLegendreMeas<SCALAR,SW_TYPE>(&random, BETA, FLAVORS, 400)
    );
  }

  //Equal_time_G1
  if (is_worm_space_active(ConfigSpaceEnum::Equal_time_G1)) {
    register_worm_meas(
      ConfigSpaceEnum::Equal_time_G1,
      "Equal_time_G1",
      new EqualTimeG1Meas<SCALAR,SW_TYPE>(&random, BETA, FLAVORS,
        par["measurement.equal_time_G1.num_ins"]
      )
    );
  }

  //Two_point_PH (lambda)
  if (is_worm_space_active(ConfigSpaceEnum::lambda)) {
    register_worm_meas(
      ConfigSpaceEnum::lambda,
      "lambda_legendre",
      new TwoPointCorrMeas<SCALAR,SW_TYPE,PH_CHANNEL>(&random, BETA, FLAVORS, 400)
    );
  }

  //Two_point_PP (varphi)
  if (is_worm_space_active(ConfigSpaceEnum::varphi)) {
    register_worm_meas(
      ConfigSpaceEnum::varphi,
      "varphi_legendre",
      new TwoPointCorrMeas<SCALAR,SW_TYPE,PP_CHANNEL>(&random, BETA, FLAVORS, 400)
    );
  }

  // eta
  if (is_worm_space_active(ConfigSpaceEnum::eta)) {
    register_worm_meas(
      ConfigSpaceEnum::eta, "eta",
      new ThreePointCorrMeas<SCALAR,SW_TYPE,PH_CHANNEL>(&random, BETA, FLAVORS)
    );
  }

  // gamma
  if (is_worm_space_active(ConfigSpaceEnum::gamma)) {
    register_worm_meas(
      ConfigSpaceEnum::gamma, "gamma",
      new ThreePointCorrMeas<SCALAR,SW_TYPE,PP_CHANNEL>(&random, BETA, FLAVORS)
    );
  }

  // G2
  if (is_worm_space_active(ConfigSpaceEnum::G2)) {
    if (par["measurement.G2.SIE.on"] != 0) {
      register_worm_meas(
        ConfigSpaceEnum::G2,
        "h_corr",
        new HCorrMeas<SCALAR,SW_TYPE>(&random, BETA, FLAVORS)
      );
    }
  }

  // h
  if (is_worm_space_active(ConfigSpaceEnum::h)) {
    register_worm_meas(
      ConfigSpaceEnum::h, "h",
      new HCorrMeas<SCALAR,SW_TYPE>(&random, BETA, FLAVORS)
    );
  }
}

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::resize_vectors() {
  {
    swap_vector.resize(0);
    std::string input_str(par["update.swap_vector"].template as<std::string>());

    //When SPINS==2, a global spin flip is pre-defined
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

    // Predefined spin flips
    if (FLAVORS%2 == 0) {
      auto SITES = FLAVORS/2;
      for (int site = 0; site < SITES; ++site) {
        flavors_vector.push_back(2 * site + 1);
        flavors_vector.push_back(2 * site);
      }
      for (int site = 0; site < SITES; ++site) {
        flavors_vector.push_back(site + SITES);
      }
      for (int site = 0; site < SITES; ++site) {
        flavors_vector.push_back(site);
      }
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
      logger_out << "The following swap updates will be performed." << std::endl;
      for (int i = 0; i < swap_vector.size(); ++i) {
        logger_out << "Update #" << i << " generated from template #" << swap_vector[i].second << std::endl;
        for (int j = 0; j < swap_vector[i].first.size(); ++j) {
          logger_out << "flavor " << j << " to flavor " << swap_vector[i].first[j] << std::endl;
        }
      }
    }
  }
}