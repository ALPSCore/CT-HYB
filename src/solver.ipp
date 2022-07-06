#pragma once

#include "util.hpp"

#include <boost/any.hpp>

#include <alps/utilities/signal.hpp>
#include <alps/mc/api.hpp>
#include <alps/mc/mcbase.hpp>
#include <alps/mc/stop_callback.hpp>
#include <alps/utilities/mpi.hpp>
#include <alps/hdf5.hpp>
#include <alps/numeric/tensors.hpp>

#include "impurity.hpp"
#include "mc/mympiadapter.hpp"
#include "postprocess.hpp"
#include "hdf5/boost_any.hpp"

namespace alps {
namespace cthyb {

template<typename Scalar>
MatrixSolver<Scalar>::MatrixSolver(const alps::params &parameters) : Base(parameters), mc_results_(), results_() {}

template<typename Scalar>
void MatrixSolver<Scalar>::define_parameters(alps::params &parameters) {
  SOLVER_TYPE::define_parameters(parameters);
}

template<typename Scalar>
int MatrixSolver<Scalar>::solve(const std::string& dump_file) {
  try {
    alps::mpi::communicator c;
    const int my_rank = c.rank();
    const int verbose = Base::parameters_["verbose"];
    if (my_rank == 0 && verbose > 0) {
      std::cout << "Creating simulation..." << std::endl;
    }

    sim_type sim(Base::parameters_, c);
    const boost::function<bool()> cb = alps::stop_callback(c, size_t(Base::parameters_["timelimit"]));

    std::pair<bool, bool> r = sim.run(cb);

    if (c.rank() == 0) {
      if (!r.second) {
        throw std::runtime_error("Master process is not thermalized yet. Increase simulation time!");
      }
      mc_results_ = alps::collect_results(sim);

      //post process basis transformation etc.
      {
        //Average sign
        results_["Sign"] = mc_results_["Sign"].template mean<double>();

        //Single-particle Green's function
        compute_G1<SOLVER_TYPE>(mc_results_, Base::parameters_, sim.get_rotmat_Delta(), results_);
        if (Base::parameters_["measurement.equal_time_G1.on"] != 0) {
          compute_euqal_time_G1<SOLVER_TYPE>(mc_results_, Base::parameters_, sim.get_rotmat_Delta(), results_);
        }

        //Two-particle Green's function
        if (Base::parameters_["measurement.G2.on"] != 0) {
          compute_G2<SOLVER_TYPE>(mc_results_, Base::parameters_, sim.get_rotmat_Delta(), results_);
        }
        if (Base::parameters_["measurement.two_time_G2.on"] != 0) {
          compute_two_time_G2<SOLVER_TYPE>(mc_results_, Base::parameters_, sim.get_rotmat_Delta(), results_);
        }
        if (Base::parameters_["measurement.equal_time_G2.on"] != 0) {
          compute_euqal_time_G2<SOLVER_TYPE>(mc_results_, Base::parameters_, sim.get_rotmat_Delta(), results_);
        }

        //Density-density correlation
        if (Base::parameters_["measurement.nn_corr.n_def"] != 0) {
          compute_nn_corr<SOLVER_TYPE>(mc_results_, Base::parameters_, results_);
        }

        //Fidelity susceptibility
        compute_fidelity_susceptibility<SOLVER_TYPE>(mc_results_, Base::parameters_, results_);

        if (my_rank == 0) {
          sim.show_statistics(mc_results_);
        }
      }

    } else {
      if (r.second) {
        alps::collect_results(sim);
      } else {
        throw std::runtime_error((boost::format(
            "Warning: MPI process %1% is not thermalized yet. Increase simulation time!") % my_rank).str());
      }
    }

    //c.barrier();
    // Dump data for debug
    if (c.rank() == 0 && dump_file != "") {
      alps::hdf5::archive ar(dump_file, "w");
      ar["/model/num_sectos"] << sim.p_model->num_sectors();
      ar["/model/sector_of_state"] << sim.p_model->get_sector_of_state();
      const auto& eigenvals_sector = sim.p_model->get_eigenvals_sector();
      for (auto s=0; s<sim.p_model->num_sectors(); ++s) {
        ar["/model/sector"+std::to_string(s)+"/eigenvals"] << eigenvals_sector[s];
        const auto& ham = sim.p_model->get_dense_ham_sector(s);
        const auto& evecs = sim.p_model->get_evecs_sector(s);
        auto dim = ham.rows();
        alps::numerics::tensor<std::complex<double>,2> ham_tmp(dim, dim), evecs_tmp(dim, dim);
        for (auto i=0; i<dim; ++i) {
          for (auto j=0; j<dim; ++j) {
            ham_tmp(i, j) = ham(i, j);
            evecs_tmp(i, j) = evecs(i, j);
          }
        }
        ar["/model/sector"+std::to_string(s)+"/ham"] << ham_tmp;
        ar["/model/sector"+std::to_string(s)+"/evecs"] << evecs_tmp;
      }
    }

  } catch (const std::exception& e) {
      std::cerr << "Thrown exception " << e.what() << std::endl;
      return 1;
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
