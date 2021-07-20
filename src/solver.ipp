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
  //try {
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

      //post process 
      {
        //Average sign
        auto sign     = mc_results_["Sign"].template mean<double>(); 
        results_["Sign"] = sign;
        auto nsites   = Base::parameters_["model.sites"].template as<int>();
        auto nspins   = Base::parameters_["model.spins"].template as<int>();
        auto beta     = Base::parameters_["model.beta"].template as<double>();
        auto nflavors = nsites * nspins;

        auto G1_vol = mc_results_["worm_space_volume_G1"].template mean<double>();
        auto equal_time_G1_vol = mc_results_["worm_space_volume_Equal_time_G1"].template mean<double>();
        auto Z_vol =  mc_results_["Z_function_space_volume"].template mean<double>();

        //Single-particle Green's function
        std::cout << "Postprocessing G1..." << std::endl;
        compute_G1<SOLVER_TYPE>(mc_results_, Base::parameters_, results_);

        //Two-particle Green's function
        if (Base::parameters_["measurement.G2.matsubara.on"] != 0) {
          std::cout << "Postprocessing G2 (matsubara)..." << std::endl;
          compute_G2_matsubara<SOLVER_TYPE>(mc_results_, Base::parameters_);
        }
        if (Base::parameters_["measurement.G2.legendre.on"] != 0) {
          std::cout << "Postprocessing G2 (legendre)..." << std::endl;
          compute_G2<SOLVER_TYPE>(mc_results_, Base::parameters_, results_);
        }

        compute_equal_time_G1(mc_results_, nflavors, beta, sign, equal_time_G1_vol/Z_vol, results_);
        compute_vartheta(mc_results_, nflavors, beta, sign, G1_vol/Z_vol, results_);

        /**
        if (Base::parameters_["measurement.two_time_G2.on"] != 0) {
          std::cout << "Postprocessing two_time_G2..." << std::endl;
          compute_two_time_G2<SOLVER_TYPE>(mc_results_, Base::parameters_, results_);
        }
        if (Base::parameters_["measurement.equal_time_G2.on"] != 0) {
          std::cout << "Postprocessing equal_time_G2..." << std::endl;
          compute_equal_time_G2<SOLVER_TYPE>(mc_results_, Base::parameters_, results_);
        }

        //Density-density correlation
        if (Base::parameters_["measurement.nn_corr.n_def"] != 0) {
          std::cout << "Postprocessing nn_corr..." << std::endl;
          compute_nn_corr<SOLVER_TYPE>(mc_results_, Base::parameters_, results_);
        }
        **/

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

    c.barrier();

    // Dump data for debug
    if (c.rank() == 0 && dump_file != "") {
      alps::hdf5::archive ar(dump_file, "w");
      ar["/parameters"] << Base::parameters_;
      ar["/simulation/results"] << this->get_accumulated_results();
      {
        const std::map<std::string,boost::any> &results = this->get_results();
        for (std::map<std::string,boost::any>::const_iterator it = results.begin(); it != results.end(); ++it) {
          ar["/" + it->first] << it->second;
        }
      }
    }

  //} catch (const std::exception& e) {
      //std::cerr << "Thrown exception " << e.what() << std::endl;
      //return 1;
  //}

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
