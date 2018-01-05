#pragma once

#include <boost/any.hpp>

#include <alps/gf/gf.hpp>
#include <alps/hdf5/archive.hpp>
#include <alps/utilities/cast.hpp>

namespace alps {
namespace hdf5 {

namespace detail {
  typedef alps::gf::three_index_gf<std::complex<double>, alps::gf::itime_mesh,
                                 alps::gf::index_mesh,
                                 alps::gf::index_mesh
  > G1_tau_t;

  typedef alps::gf::three_index_gf<std::complex<double>, alps::gf::matsubara_positive_mesh,
                                 alps::gf::index_mesh,
                                 alps::gf::index_mesh
  > G1_omega_t;

  template<class T>
  struct save_if_match {
    static bool perform(
      archive & ar
      , std::string const & path
      , boost::any const & value
      , std::vector<std::size_t> size = std::vector<std::size_t>()
      , std::vector<std::size_t> chunk = std::vector<std::size_t>()
      , std::vector<std::size_t> offset = std::vector<std::size_t>()
    ) {
      if (value.type() == typeid(T)) {
        save(ar, ar.complete_path(path), boost::any_cast<const T&>(value));
        return true;
      } else {
        return false;
      }
    }
  };
}

inline void save(
    archive & ar
    , std::string const & path
    , boost::any const & value
    , std::vector<std::size_t> size = std::vector<std::size_t>()
    , std::vector<std::size_t> chunk = std::vector<std::size_t>()
    , std::vector<std::size_t> offset = std::vector<std::size_t>()
) {
  typedef std::complex<double> dcomplex_;

  int cnt = 0;

  cnt += detail::save_if_match<double>::perform(ar, path, value, size, chunk, offset) ? 1 : 0;
  cnt += detail::save_if_match<dcomplex_>::perform(ar, path, value, size, chunk, offset) ? 1 : 0;

  cnt += detail::save_if_match<boost::multi_array<double,1 > >::perform(ar, path, value, size, chunk, offset) ? 1 : 0;
  cnt += detail::save_if_match<boost::multi_array<double,2 > >::perform(ar, path, value, size, chunk, offset) ? 1 : 0;
  cnt += detail::save_if_match<boost::multi_array<double,3 > >::perform(ar, path, value, size, chunk, offset) ? 1 : 0;
  cnt += detail::save_if_match<boost::multi_array<double,4 > >::perform(ar, path, value, size, chunk, offset) ? 1 : 0;
  cnt += detail::save_if_match<boost::multi_array<double,5 > >::perform(ar, path, value, size, chunk, offset) ? 1 : 0;
  cnt += detail::save_if_match<boost::multi_array<double,6 > >::perform(ar, path, value, size, chunk, offset) ? 1 : 0;
  cnt += detail::save_if_match<boost::multi_array<double,7 > >::perform(ar, path, value, size, chunk, offset) ? 1 : 0;
  cnt += detail::save_if_match<boost::multi_array<double,8 > >::perform(ar, path, value, size, chunk, offset) ? 1 : 0;

  cnt += detail::save_if_match<boost::multi_array<dcomplex_,1 > >::perform(ar, path, value, size, chunk, offset) ? 1 : 0;
  cnt += detail::save_if_match<boost::multi_array<dcomplex_,2 > >::perform(ar, path, value, size, chunk, offset) ? 1 : 0;
  cnt += detail::save_if_match<boost::multi_array<dcomplex_,3 > >::perform(ar, path, value, size, chunk, offset) ? 1 : 0;
  cnt += detail::save_if_match<boost::multi_array<dcomplex_,4 > >::perform(ar, path, value, size, chunk, offset) ? 1 : 0;
  cnt += detail::save_if_match<boost::multi_array<dcomplex_,5 > >::perform(ar, path, value, size, chunk, offset) ? 1 : 0;
  cnt += detail::save_if_match<boost::multi_array<dcomplex_,6 > >::perform(ar, path, value, size, chunk, offset) ? 1 : 0;
  cnt += detail::save_if_match<boost::multi_array<dcomplex_,7 > >::perform(ar, path, value, size, chunk, offset) ? 1 : 0;
  cnt += detail::save_if_match<boost::multi_array<dcomplex_,8 > >::perform(ar, path, value, size, chunk, offset) ? 1 : 0;

  //r = detail::save_if_match<detail::G1_omega_t>::perform(ar, path, value, size, chunk, offset);
  //r = detail::save_if_match<detail::G1_tau_t>::perform(ar, path, value, size, chunk, offset);

  if (value.type() == typeid(detail::G1_omega_t)) {
    boost::any_cast<const detail::G1_omega_t&>(value).save(ar, ar.complete_path(path));
    ++ cnt;
  }
  if (value.type() == typeid(detail::G1_tau_t)) {
    boost::any_cast<const detail::G1_tau_t&>(value).save(ar, ar.complete_path(path));
    ++ cnt;
  }

  if (cnt == 0) {
    throw std::runtime_error("No matching rule for saving the given object to a HDF5 file!");
  }
}

inline void load(
    archive & ar
    , std::string const & path
    , boost::any & value
    , std::vector<std::size_t> chunk = std::vector<std::size_t>()
    , std::vector<std::size_t> offset = std::vector<std::size_t>()
) {
  throw std::runtime_error("A boost::any object cannot be loaded from a HDF file.");
}
}
}
