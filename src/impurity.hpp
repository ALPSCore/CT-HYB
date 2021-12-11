/************************************************************************************
 *
 * Hybridization expansion code for multi-orbital systems with general interactions
 *
 * Copyright (C) 2016 by Hiroshi Shinaoka <h.shinaoka@gmail.com>,
 *                                    Emanuel Gull <egull@umich.edu>,
 *                                    Philipp Werner <philipp.werner@unifr.ch>
 *
 *
 * This software is published under the GNU General Public License version 2.
 * See the file LICENSE.txt.
 *
 *************************************************************************************/
#pragma once

#include <vector>
#include <iostream>
#include <memory>
#include <algorithm>
#include <cmath>
#include <valarray>
#include <time.h>
#include <unordered_map>

#include <boost/assert.hpp>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/sequenced_index.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/identity.hpp>
#include <boost/range/algorithm/for_each.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>
#include <boost/array.hpp>

//Eigen3
#include<Eigen/Dense>

//ALPSCore
#include <alps/params.hpp>
#include <alps/accumulators.hpp>
#include <alps/mc/api.hpp>
#include <alps/mc/mcbase.hpp>
#ifdef ALPS_HAVE_MPI
#include <alps/mc/mpiadapter.hpp>
#endif
#include <alps/mc/stop_callback.hpp>
#include <alps/params/convenience_params.hpp>

#include "common/wide_scalar.hpp"
#include "common/logger.hpp"
#include "model/operator.hpp"
#include "model/atomic_model.hpp"

#include "sliding_window/sliding_window.hpp"

#include "moves/moves.hpp"
#include "moves/mc_config.hpp"

#include "measurement/all.hpp"

#include "accumulator.hpp"
#include "update_histogram.hpp"
#include "wang_landau.hpp"


template<typename IMP_MODEL>
class HybridizationSimulation: public alps::mcbase {
 public:
  HybridizationSimulation(parameters_type const &params, const alps::mpi::communicator &comm); //constructor

  //TYPES
  typedef alps::mcbase Base;
  typedef typename model_traits<IMP_MODEL>::SCALAR_T SCALAR;
  typedef std::complex<double> COMPLEX;
  typedef SlidingWindowManager<IMP_MODEL> SW_TYPE;
  typedef typename ExtendedScalar<SCALAR>::value_type EXTENDED_SCALAR;

  static void define_parameters(parameters_type &parameters);
  static std::vector<ConfigSpaceEnum::Type> get_defined_worm_spaces(const parameters_type &parameters);

  void update(); //the main monte carlo step
  void measure_every_step();//measure every step, which is called in update()
  void measure(); //the top level of the measurement
  void measure_Z_function_space(); //the main monte carlo step
  void prepare_for_measurement(); //called once after thermalization is reached
  void finish_measurement(); //called once after thermalization is done
  virtual double fraction_completed() const;

  void resize_vectors(); //early initialization stuff

  //update thermalization status
  void update_thermalization_status() {
    if (time(NULL) - start_time > thermalization_time) {
      thermalized = true;
    }
  }

  bool is_thermalized() const;
  static void print_copyright(std::ostream &os) {
    os << "Matrix code based on the hybridization expansion method of PRB 74, 155107 (2006)" << std::endl
        << "This program is licensed under GPLv2.";
  }

  std::vector<std::string> get_active_worm_updaters() const {
    std::vector<std::string> names;
    for (int i = 0; i < worm_insertion_removers.size(); ++i) {
      names.push_back(worm_insertion_removers[i]->get_name());
    }
    for (typename worm_updater_map_t::const_iterator it = worm_movers.begin(); it != worm_movers.end(); ++it) {
      names.push_back(it->second->get_name());
    }
    for (typename std::map<std::string, std::shared_ptr<LocalUpdaterType> >::const_iterator
             it = specialized_updaters.begin(); it != specialized_updaters.end(); ++ it) {
      names.push_back(it->second->get_name());
    }
    return names;
  }

  //for postprocess
  void show_statistics(const alps::accumulators::result_set &results);

 private:
  //for set up
  void create_worm_spaces();
  void create_custom_worm_updaters();
  void create_worm_meas();
  void create_observables(); //build ALPS observables

  template<typename W>
  void add_worm_mover(ConfigSpaceEnum::Type config_space,
                      const std::string &updater_name);
  
  template<typename WORM_T>
  void add_worm_space();

  void do_one_sweep(); // one sweep of the window
  void transition_between_config_spaces();
  void global_updates(); //expensive updates
  void update_MC_parameters(); //update parameters for MC moves during thermalization steps
  void adjust_worm_space_weight();

  int get_config_space_position(ConfigSpaceEnum::Type config_space) const {
    if (config_space == ConfigSpaceEnum::Z_FUNCTION) {
      return 0;
    } else {
      std::vector<ConfigSpaceEnum::Type>::const_iterator
          it = std::find(worm_types.begin(), worm_types.end(), config_space);
      if (it == worm_types.end()) {
        return -1;
      } else {
        return std::distance(worm_types.begin(), it) + 1;
      }
    }
  }

  bool is_worm_space_active(ConfigSpaceEnum::Type w) const {
    return std::find(worm_types.begin(), worm_types.end(), w) != worm_types.end();
  }

  //Definition of system parameters constant during simulation
  const parameters_type par;
  const double BETA;
  const int FLAVORS;
  const int N;
  const int N_non_worm_meas;
  double thermalization_time;
  const time_t start_time;

  //Model object
  std::shared_ptr<IMP_MODEL> p_model;

  std::shared_ptr<HybridizationFunction<SCALAR> > F;

  //ALPS MPI communicator
  alps::mpi::communicator comm;

  //nearly equal to the average perturbation order (must be kept fixed during measurement steps)
  int N_win_standard;

  //Monte Calro configuration
  long sweeps;                          // sweeps done
  MonteCarloConfiguration<SCALAR> mc_config;

  //for Z_function space, and active worm spaces.
  //The active worm spaces are in the same order as they are initialized in the constructor.
  //Use get_worm_position to find the actual position of a given worm space.
  std::vector<double> config_space_extra_weight;

  //std::map version which contains the same information as config_space_extra_weight
  std::map<ConfigSpaceEnum::Type, double> worm_space_extra_weight_map;

  /* Monte Carlo updater */
  //insertion/removal updater for single pair update, double pair update, triple pair update, etc.
  std::vector<std::shared_ptr<InsertionRemovalUpdater<SCALAR, EXTENDED_SCALAR, SW_TYPE> > >
      ins_rem_updater;

  //change the flavor of a pair of operators
  OperatorPairFlavorUpdater<SCALAR, EXTENDED_SCALAR, SW_TYPE> operator_pair_flavor_updater;

  //change the time of an operator (hybrized with the bath)
  SingleOperatorShiftUpdater<SCALAR, EXTENDED_SCALAR, SW_TYPE> single_op_shift_updater;

  //swap-flavor update
  std::vector<std::pair<std::vector<int>, int> >
      swap_vector;        // contains the flavors f1 f2 f3 f4 ...   Flavors 1 ... N will be relabeled as f1 f2 ... fN.

  //N2Worm updater: worm for computing <c^\dagger_i(tau) c_j(tau) c^\dagger_k(0) c_l(0)>
  typedef LocalUpdater<SCALAR, EXTENDED_SCALAR, SW_TYPE> LocalUpdaterType;
  typedef WormUpdater<SCALAR, EXTENDED_SCALAR, SW_TYPE> WormUpdaterType;
  typedef WormMover<SCALAR, EXTENDED_SCALAR, SW_TYPE> WormMoverType;
  typedef WormFlavorChanger<SCALAR, EXTENDED_SCALAR, SW_TYPE> WormFlavorChangerType;
  typedef WormInsertionRemover<SCALAR, EXTENDED_SCALAR, SW_TYPE> WormInsertionRemoverType;
  typedef GWormInsertionRemover<SCALAR, 1, EXTENDED_SCALAR, SW_TYPE> G1WormInsertionRemoverType;
  typedef GWormInsertionRemover<SCALAR, 2, EXTENDED_SCALAR, SW_TYPE> G2WormInsertionRemoverType;

  //a list of active worm spaces
  std::vector<ConfigSpaceEnum::Type> worm_types;

  //update the status of the worm (time, flavor), move head and tail
  typedef std::multimap<ConfigSpaceEnum::Type, std::shared_ptr<WormUpdaterType> > worm_updater_map_t;
  worm_updater_map_t worm_movers;

  //insertion and removal of a worm by evaluating the trace (worm space <=> Z function space)
  std::vector<std::shared_ptr<WormInsertionRemoverType> > worm_insertion_removers;

  //specialized version with improved efficiency
  std::map<std::string, std::shared_ptr<LocalUpdaterType> > specialized_updaters;

  std::shared_ptr<FlatHistogram> p_flat_histogram_config_space;

  //sliding window for computing trace
  SW_TYPE sliding_window;

  //Measurement of single-particle Green's functions by worm sampling
  //std::shared_ptr<GMeasurement<SCALAR, 1> > p_G1_legendre_meas;

  //Measurement of two-particle Green's functions by worm sampling (Legendre basis)
  //std::shared_ptr<GMeasurement<SCALAR, 2> > p_G2_legendre_meas;

  using WORM_MEAS_TYPE = WormMeas<SCALAR,SW_TYPE>;
  std::unordered_map<ConfigSpaceEnum::Type,std::unordered_map<std::string,std::unique_ptr<WORM_MEAS_TYPE>>> worm_meas;

  //Acceptance rate of global shift and swap updates
  AcceptanceRateMeasurement global_shift_acc_rate;
  std::vector<AcceptanceRateMeasurement> swap_acc_rate;

  //for measuring the volume of configuration spaces
  std::vector<double> num_steps_in_config_space;

  //timings (msec/N_MEAS steps)
  //0 : local update
  //1 : global update
  //2 : measuring single-particle green's function and N2 correlation function
  std::vector<double> timings;

  bool verbose;

  bool thermalized;

  PertOrderRecorder pert_order_recorder;

  std::vector<bool> config_spaces_visited_in_measurement_steps;

  //SamplingFreqsG2SIE smpl_freqs_SIE;

  void sanity_check();

};