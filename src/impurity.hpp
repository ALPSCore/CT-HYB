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
#include <algorithm>
#include <cmath>
#include <valarray>

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
#ifdef MEASURE_TIMING
  #include <boost/timer/timer.hpp>
#endif

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

#include "wide_scalar.hpp"
#include "operator.hpp"
#include "model.hpp"
#include "moves.hpp"
#include "sliding_window.hpp"
#include "update_histogram.hpp"
#include "accumulator.hpp"
#include "resizable_matrix.hpp"
#include "measurement.hpp"


template<typename IMP_MODEL>
class HybridizationSimulation : public alps::mcbase
{
public:
  HybridizationSimulation(parameters_type const & params, int rank); //constructor

  //TYPES
  typedef alps::mcbase Base;
  typedef typename model_traits<IMP_MODEL>::SCALAR_T SCALAR;
  typedef alps::ResizableMatrix<SCALAR> matrix_t;
  typedef std::complex<double> COMPLEX;
  typedef SlidingWindowManager<IMP_MODEL> SW_TYPE;
  typedef typename ExtendedScalar<SCALAR>::value_type EXTENDED_SCALAR;

  static void define_parameters(parameters_type & parameters);
  void create_observables(); //build ALPS observables

  void update(); //the main monte carlo step
  void measure(); //the main monte carlo step
  void prepare_for_measurement(); //called once after thermalization is reached
  virtual double fraction_completed() const;

  void resize_vectors(); //early initialization stuff

  bool is_thermalized() const;
  static void print_copyright(std::ostream & os) {
                         os << "Matrix code based on the hybridization expansion method of PRB 74, 155107 (2006)" << std::endl
													<< "This program is licensed under GPLv2.";}

private:
  //for set up
  void read_eq_time_two_particle_greens_meas();
  void read_two_time_correlation_functions();

  void expensive_updates(); //expensive updates
  void update_MC_parameters(); //update parameters for MC moves during thermalization steps
  void measure_n();
  void measure_two_time_correlation_functions();

  //Definition of system constant during simulation
  const parameters_type par;
  const double BETA;
  const int SITES;
  const int SPINS;
  const int FLAVORS;
  const int N;
  const int Np1;

  //Model object
  boost::scoped_ptr<IMP_MODEL> p_model;

  //ALPS MPI communicator
#ifdef ALPS_HAVE_MPI
  alps::mpi::communicator comm;
#endif

  //Constant simulation parameters
  ThermalizationChecker thermalization_checker;
  const long total_sweeps;                    // sweeps to be done after equilibration

  //Simulation parameters that may be modified after/during thermalization
  int N_meas;
  int N_meas_g;
  const int N_shift;
  int N_swap;

  //Monte Calro status
  long sweeps;                          // sweeps done
  alps::ResizableMatrix<SCALAR> M;
  SCALAR sign;							// the sign of w=Z_k_up*Z_k'_down*trace
  EXTENDED_SCALAR trace;							// matrix trace

  typedef typename std::iterator_traits<std::vector<int>::iterator>::value_type mytpe;

  operator_container_t operators;	// contains times and types (site, flavor) of the operators
  operator_container_t creation_operators;
  operator_container_t annihilation_operators;
  std::vector<int> order_creation_flavor;//deprecated
  std::vector<int> order_annihilation_flavor;

  std::vector<int> N_shift_flavor;

  //swap-flavor update
  std::vector<std::pair<std::vector<int>, int> > swap_vector;		// contains the flavors f1 f2 f3 f4 ...   Flavors 1 ... N will be relabeled as f1 f2 ... fN.
  //the second elements of std::pair denote from which entries in input acual updates are generated.

  //sliding window
  SW_TYPE sliding_window;

  double                    max_distance_pair; //cutoff for insert/removal of pair
  double                    acc_rate_cutoff;
  scalar_histogram_flavors  weight_ins;
  scalar_histogram_flavors  weight_rem;
  scalar_histogram_flavors  weight_shift;

  //for measuring Green's function
  GreensFunctionLegendreMeasurement<SCALAR> g_meas_legendre;

  //For measuring equal-time two-particle Green's function
  std::vector<EqualTimeOperator<2> > eq_time_two_particle_greens_meas;

  //For measuring two-time correlation functions <c^dagger(tau) c(tau) c^dagger(0) c(0)>
  boost::scoped_ptr<MeasCorrelation<SW_TYPE, EqualTimeOperator<1> > > p_meas_corr;

  //Acceptance rate of global shift and swap updates
  AcceptanceRateMeasurement global_shift_acc_rate;
  std::vector<AcceptanceRateMeasurement> swap_acc_rate;

  AcceptanceRateMeasurement prob_valid_rem_move;

  //timings (msec/N_MEAS steps)
  //0 : local update
  //1 : global update
  //2 : move of sliding window
  //3 : measuring single-particle green's function
  //4 : rest of measurement
  std::vector<double> timings;

  bool verbose;

  void sanity_check() const;
};

template<typename MAT, typename MAT_COMPLEX, typename COMPLEX>
void
transform_G_back_to_original_basis(int FLAVORS, int SITES, int SPINS, int Np1, const MAT& rotmat_Delta, const MAT& inv_rotmat_Delta, std::vector<COMPLEX>& G);

#include "./impurity.ipp"
#include "./impurity_init.ipp"
