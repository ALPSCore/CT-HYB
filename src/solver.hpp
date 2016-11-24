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

#include <alps/utilities/signal.hpp>
#include <alps/mc/api.hpp>
#include <alps/mc/mcbase.hpp>
#include <alps/mc/stop_callback.hpp>
#include <alps/gf/gf.hpp>
#include <alps/hdf5.hpp>
#ifdef ALPS_HAVE_MPI
#include <alps/utilities/mpi.hpp>
#endif

#ifndef ALPS_HAVE_MPI
MPI environment is required!
#endif

template<typename T> class ImpurityModelEigenBasis;
template<typename T> class HybridizationSimulation;
template<typename T> class mymcmpiadapter;

namespace alps {
namespace cthyb {

class Solver {
 public:
  Solver(const alps::params &parameters) : comm_(alps::mpi::communicator()), parameters_(parameters) {}

  virtual int solve() = 0;

  /** Get a reference to a collection of results */
  virtual const std::map<std::string,boost::any>& get_results() const = 0;

  /** Get a reference to accumulated raw Monte Carlo results */
  virtual const alps::accumulators::result_set& get_accumulated_results() const = 0;

  typedef alps::gf::three_index_gf<std::complex<double>, alps::gf::itime_mesh,
                                   alps::gf::index_mesh,
                                   alps::gf::index_mesh
  > G1_tau_t;

  typedef alps::gf::three_index_gf<std::complex<double>, alps::gf::matsubara_positive_mesh,
                                   alps::gf::index_mesh,
                                   alps::gf::index_mesh
  > G1_omega_t;

 protected:
  alps::mpi::communicator comm_;
  alps::params parameters_;
};

template<typename Scalar>
class MatrixSolver : public Solver {
 private:
  typedef Solver Base;
  typedef HybridizationSimulation<ImpurityModelEigenBasis<Scalar> > SOLVER_TYPE;
  typedef mymcmpiadapter<SOLVER_TYPE> sim_type;

 public:
  MatrixSolver(const alps::params &parameters);

  static void define_parameters(alps::params &parameters);

  int solve();

  /** Get a reference to a collection of results */
  const std::map<std::string,boost::any>& get_results() const;

  /** Get a reference to accumulated raw Monte Carlo results */
  const alps::accumulators::result_set& get_accumulated_results() const;

 private:

  alps::accumulators::result_set mc_results_;

  std::map<std::string,boost::any> results_;
};

}
}
