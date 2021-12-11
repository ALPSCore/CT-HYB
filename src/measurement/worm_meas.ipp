#include "worm_meas.hpp"

template<typename TIME_T, typename VAL_REAL_T>
void
WormConfigRecord<TIME_T,VAL_REAL_T>::add(const Worm &worm, const std::complex<double> &val) {
  for (auto t=0; t<num_time_idx_; ++t) {
    taus[t].push_back(worm.get_time(t));
  }
  for (auto f=0; f<num_flavor_idx_; ++f) {
    flavors[f].push_back(worm.get_flavor(f));
  }
  vals_real.push_back(val.real());
  vals_imag.push_back(val.imag());
}

template<typename TIME_T, typename VAL_REAL_T>
void
WormConfigRecord<TIME_T,VAL_REAL_T>::save(alps::hdf5::archive &oar, const std::string &path) const {
  for (auto t=0; t<num_time_idx_; ++t) {
    oar[path + "/taus/" + std::to_string(t)] = taus[t];
  }
  for (auto f=0; f<num_flavor_idx_; ++f) {
    oar[path + "/flavors/" + std::to_string(f)] = flavors[f];
  }
  oar[path + "/vals_real"] = vals_real;
  oar[path + "/vals_imag"] = vals_imag;
}

template<typename TIME_T, typename VAL_REAL_T>
void
WormConfigRecord<TIME_T,VAL_REAL_T>::load(alps::hdf5::archive &iar, const std::string &path) {
  for (auto t=0; t<num_time_idx_; ++t) {
    iar[path + "/taus/" + std::to_string(t)] >> taus[t];
  }
  for (auto f=0; f<num_flavor_idx_; ++f) {
    iar[path + "/flavors/" + std::to_string(f)] >> flavors[f];
  }
  iar[path + "/vals_real"] >> vals_real;
  iar[path + "/vals_imag"] >> vals_imag;
}