#include "impurity.hpp"

inline void print_acc_rate(const alps::accumulators::result_set &results, const std::string &name, std::ostream &os) {
  os << " " << name + " : "
     << results[name + "_accepted_scalar"].mean<double>()
         / results[name + "_valid_move_scalar"].mean<double>()
     << std::endl;
}

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::show_statistics(const alps::accumulators::result_set &results) {
  const std::vector<double> timings = results["TimingsSecPerNMEAS"].template mean<std::vector<double> >();
  logger_out << std::endl << "==== Timings analysis ====" << std::endl;
  logger_out << " The following are the timings per n_non_worm_meas window sweeps (in units of second): " << std::endl;
  logger_out << " Local updates (insertion/removal/shift of operators/worm: " << timings[0] * 1e-9 << std::endl;
  logger_out << " Global updates (global shift etc.): " << timings[1] * 1e-9 << std::endl;
  logger_out << " Worm measurement: " << timings[2] * 1e-9 << std::endl;
  logger_out << " Rebuild inverse of determinat matrix : " << timings[3] * 1e-9 << std::endl;
  logger_out << " Non worm measurement: " << timings[4] * 1e-9 << std::endl;

  logger_out << std::endl << "==== Thermalization analysis ====" << std::endl;
  logger_out << boost::format("Perturbation orders just before and after measurement steps are %1% and %2%.") %
      results["Pert_order_start"].template mean<double>() %
      results["Pert_order_end"].template mean<double>() << std::endl;

  logger_out << std::endl << "==== Number of Monte Carlo steps spent in configuration spaces ====" << std::endl;
  logger_out << "Z function" << " : " << results["Z_function_space_num_steps"].template mean<double>() << std::endl;
  for (int w = 0; w < worm_types.size(); ++w) {
    logger_out << ConfigSpaceEnum::to_string(worm_types[w]) << " : "
              << results["worm_space_num_steps_" + ConfigSpaceEnum::to_string(worm_types[w])].template mean<double>()
              << std::endl;
  }

  logger_out << std::endl << "==== Acceptance updates of operators hybridized with bath ====" << std::endl;
  for (int k = 1; k < par["update.multi_pair_ins_rem"].template as<int>() + 1; ++k) {
    print_acc_rate(results, ins_rem_updater[k-1]->get_name(), logger_out);
  }
  print_acc_rate(results, single_op_shift_updater.get_name(), logger_out);
  print_acc_rate(results, operator_pair_flavor_updater.get_name(), logger_out);

  logger_out << std::endl << "==== Acceptance rates of worm updates ====" << std::endl;
  std::vector<std::string> active_worm_updaters = get_active_worm_updaters();
  for (int iu = 0; iu < active_worm_updaters.size(); ++iu) {
    print_acc_rate(results, active_worm_updaters[iu], logger_out);
  }
}
