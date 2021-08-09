/************************************************************************************
 *
 * Hybridization expansion code for multi-orbital systems with general interactions
 *
 * Copyright (C) 2018 by Hiroshi Shinaoka <h.shinaoka@gmail.com>,
 *                                    Emanuel Gull <egull@umich.edu>,
 *                                    Philipp Werner <philipp.werner@unifr.ch>
 *
 *
 * This software is published under the GNU General Public License version 3 or later.
 * See the file LICENSE.txt.
 *
 *************************************************************************************/
#pragma once

#include <boost/any.hpp>
#include <alps/utilities/signal.hpp>
#include <alps/mc/api.hpp>
#include <alps/mc/mcbase.hpp>
#include <alps/mc/stop_callback.hpp>
#include <alps/gf/gf.hpp>
#include <alps/hdf5.hpp>
#include <alps/utilities/mpi.hpp>

#ifndef ALPS_HAVE_MPI
#error MPI environment is required!
#endif

#include "mc/mympiadapter.hpp"
#include "moves/worm.hpp"
#include "model/atomic_model.hpp"
#include "impurity.hpp"

//template<typename T> class AtomicModelEigenBasis;
//template<typename T> class HybridizationSimulation;
//template<typename T> class mymcmpiadapter;

namespace alps {
namespace cthyb {

class Solver {
 public:
  Solver(const alps::params &parameters, const alps::mpi::communicator& comm)
    : comm_(comm), parameters_(parameters) {}

  virtual int solve(const std::string& dump_file = "") = 0;

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
  typedef HybridizationSimulation<AtomicModelEigenBasis<Scalar> > SOLVER_TYPE;
  typedef mymcmpiadapter<SOLVER_TYPE> sim_type;

 public:
  MatrixSolver(const alps::params &parameters, const alps::mpi::communicator &comm);

  static void define_parameters(alps::params &parameters);
  static std::vector<ConfigSpaceEnum::Type> get_defined_worm_spaces(alps::params &parameters);

  int solve(const std::string& dump_file = "");

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
