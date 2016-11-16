#include "impurity.hpp"

inline void print_acc_rate(const alps::accumulators::result_set &results, const std::string &name, std::ostream &os) {
  os << " " << name + " : "
     << results[name + "_accepted_scalar"].mean<double>()
         / results[name + "_valid_move_scalar"].mean<double>()
     << std::endl;
}

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::show_statistics(const alps::accumulators::result_set &results) {
#ifdef MEASURE_TIMING
  const std::vector<double> timings = results["TimingsSecPerNMEAS"].template mean<std::vector<double> >();
  std::cout << std::endl << "==== Timings analysis ====" << std::endl;
  std::cout << " The following are the timings per window sweep (in units of second): " << std::endl;
  std::cout << " Local updates (insertion/removal/shift of operators/worm: " << timings[0] << std::endl;
  std::cout << " Global updates (global shift etc.): " << timings[1] << std::endl;
  std::cout << " Worm measurement: " << timings[2] << std::endl;
  std::cout << " Non worm measurement: " << timings[3] << std::endl;
#endif

  std::cout << std::endl << "==== Thermalization analysis ====" << std::endl;
  std::cout << boost::format("Perturbation orders just before and after measurement steps are %1% and %2%.") %
      results["Pert_order_start"].template mean<double>() %
      results["Pert_order_end"].template mean<double>() << std::endl;

  std::cout << std::endl << "==== Number of Monte Carlo steps spent in configuration spaces ====" << std::endl;
  std::cout << "Z function" << " : " << results["Z_function_space_num_steps"].template mean<double>() << std::endl;
  for (int w = 0; w < worm_types.size(); ++w) {
    std::cout << get_config_space_name(worm_types[w]) << " : "
              << results["worm_space_num_steps_" + get_config_space_name(worm_types[w])].template mean<double>()
              << std::endl;
  }

  std::cout << std::endl << "==== Acceptance updates of operators hybridized with bath ====" << std::endl;
  for (int k = 1; k < par["update.multi_pair_ins_rem"].template as<int>() + 1; ++k) {
    print_acc_rate(results, ins_rem_updater[k-1]->get_name(), std::cout);
  }
  print_acc_rate(results, single_op_shift_updater.get_name(), std::cout);
  print_acc_rate(results, operator_pair_flavor_updater.get_name(), std::cout);

  std::cout << std::endl << "==== Acceptance rates of worm updates ====" << std::endl;
  std::vector<std::string> active_worm_updaters = get_active_worm_updaters();
  for (int iu = 0; iu < active_worm_updaters.size(); ++iu) {
    print_acc_rate(results, active_worm_updaters[iu], std::cout);
  }
}
