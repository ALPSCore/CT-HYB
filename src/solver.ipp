#pragma once

#include "common/util.hpp"

#include <boost/any.hpp>

#include <alps/utilities/signal.hpp>
#include <alps/mc/api.hpp>
#include <alps/mc/mcbase.hpp>
#include <alps/mc/stop_callback.hpp>
#include <alps/utilities/mpi.hpp>

#include "impurity.hpp"
#include "mc/mympiadapter.hpp"
#include "postprocess.hpp"
#include "hdf5/boost_any.hpp"
#include "measurement/all.hpp"
#include "common/logger.hpp"

namespace alps {
namespace cthyb {

template<typename Scalar>
MatrixSolver<Scalar>::MatrixSolver(const alps::params &parameters,const alps::mpi::communicator &comm):
  Base(parameters, comm), mc_results_(), results_() {

}

template<typename Scalar>
void MatrixSolver<Scalar>::define_parameters(alps::params &parameters) {
  SOLVER_TYPE::define_parameters(parameters);
}

template<typename Scalar>
std::vector<ConfigSpaceEnum::Type> MatrixSolver<Scalar>::get_defined_worm_spaces(alps::params &parameters) {
  return sim_type::get_defined_worm_spaces(parameters);
}

  static std::vector<ConfigSpaceEnum::Type> get_defined_worm_spaces(alps::params &parameters);

template<typename Scalar>
int MatrixSolver<Scalar>::solve(const std::string& dump_file) {
  const int rank = comm_.rank();
  const int verbose = Base::parameters_["verbose"];
  if (rank == 0 && verbose > 0) {
    logger_out << "Creating simulation with " << std::to_string(comm_.size()) << " processes..." << std::endl;
  }

  sim_type sim(Base::parameters_, comm_);
  const boost::function<bool()> cb = alps::stop_callback(comm_, size_t(Base::parameters_["timelimit"]));

  std::pair<bool, bool> r = sim.run(cb);

  logger_out.flush();

  if (comm_.rank() == 0) {
    logger_out.flush();
    if (!r.second) {
      throw std::runtime_error("Master process is not thermalized yet. Increase simulation time!");
    }
    logger_out.flush();
    mc_results_ = alps::collect_results(sim);
    if (comm_.rank() == 0) {
      sim.show_statistics(mc_results_);
    }
    logger_out.flush();
  } else {
    logger_out.flush();
    if (r.second) {
      logger_out.flush();
      alps::collect_results(sim);
      logger_out.flush();
    } else {
      throw std::runtime_error((boost::format(
          "Warning: MPI process %1% is not thermalized yet. Increase simulation time!") % rank).str());
    }
      logger_out.flush();
  }

  return 0;
}

template<typename Scalar>
const std::map<std::string,boost::any>& MatrixSolver<Scalar>::get_results() const {
  if (Base::comm_.rank() != 0) {
    throw std::runtime_error("Cannot be called at slave MPI process");
  }
  return results_;
}

template<typename Scalar>
const alps::accumulators::result_set& MatrixSolver<Scalar>::get_accumulated_results() const {
  if (Base::comm_.rank() != 0) {
    throw std::runtime_error("Cannot be called at slave MPI process");
  }
  return mc_results_;
}

}
}
