#pragma once

#include <algorithm>
#include <string>
#include <iostream>

#include <boost/multi_array.hpp>

#include <alps/accumulators.hpp>

#include "util.hpp"

typedef alps::accumulators::accumulator_set observableset_t;

typedef alps::accumulators::LogBinningAccumulator<double> RealObservable;
typedef alps::accumulators::NoBinningAccumulator<double> SimpleRealObservable;
typedef alps::accumulators::LogBinningAccumulator<std::vector<double> > RealVectorObservable;
typedef alps::accumulators::NoBinningAccumulator<std::vector<double> > SimpleRealVectorObservable;
typedef alps::accumulators::NoBinningAccumulator<boost::multi_array<double, 2> > SimpleRealMultiArray2Observable;

template<class SCALAR, class OBS>
struct create_observable_helper {
  void static operate(observableset_t &meas, const char *obs_name) {
    //do nothing
    throw std::runtime_error("This should not be called.");
  };
};

template<class SCALAR, class OBS, class DATA>
struct measure_observable_helper {
  void static operate(observableset_t &meas, const char *obs_name, const DATA &data) {
    //do nothing
    std::cerr << " obs_name " << obs_name << std::endl;
    throw std::runtime_error("This should not be called.");
  }
};

//Helper for measuring a boost::multi_array<SCALAR,DIMENSION>
template<class SCALAR, int DIMENSION, class OBS>
struct measure_multiarray_helper {
  void static operate(observableset_t &meas, const char *obs_name, const boost::multi_array<SCALAR, DIMENSION> &data) {
    meas[std::string(obs_name) + "_Re"] << get_real_parts(data);
    meas[std::string(obs_name) + "_Im"] << get_imag_parts(data);
  }
};

/* ****
 * SPECIALIZATION FOR DOUBLE
 * ****/
template<class OBS>
struct create_observable_helper<double, OBS> {
  void static operate(observableset_t &meas, const char *obs_name) {
    meas << OBS(std::string(obs_name) + "_Re");
    meas << OBS(std::string(obs_name) + "_Im");
  };
};

//measure a scalar
template<class OBS>
struct measure_observable_helper<double, OBS, double> {
  void static operate(observableset_t &meas, const char *obs_name, const double &data) {
    meas[std::string(obs_name) + "_Re"] << data;
    meas[std::string(obs_name) + "_Im"] << 0.0;
  };
};

//measure a vector
template<class OBS>
struct measure_observable_helper<double, OBS, std::vector<double> > {
  void static operate(observableset_t &meas, const char *obs_name, const std::vector<double> &data) {
    meas[std::string(obs_name) + "_Re"] << data;
    std::vector<double> zeros(0.0, data.size());
    meas[std::string(obs_name) + "_Im"] << zeros;
  };
};

//measure a boost::multi_array<double,*>
//template<class OBS>
//struct measure_observable_helper<double, OBS, boost::multi_array<double,2> > {
//void static operate(observableset_t& meas, const char* obs_name, const boost::multi_array<double,2> & data) {
//meas[std::string(obs_name)+"_Re"] << data;
//boost::multi_array<double,2> zeros(data.shape());
//std::fill(zeros.origin(), zeros.origin()+zeros.num_elements(), 0.0);
//meas[std::string(obs_name)+"_Im"] << zeros;
//};
//};

/* ****
 * SPECIALIZATION FOR COMPLEX
 * ****/
template<class OBS>
struct create_observable_helper<std::complex<double>, OBS> {
  void static operate(observableset_t &meas, const char *obs_name) {
    meas << OBS(std::string(obs_name) + "_Re");
    meas << OBS(std::string(obs_name) + "_Im");
  };
};

//measure a scalar
template<class OBS>
struct measure_observable_helper<std::complex<double>, OBS, std::complex<double> > {
  void static operate(observableset_t &meas, const char *obs_name, const std::complex<double> &data) {
    meas[std::string(obs_name) + "_Re"] << data.real();
    meas[std::string(obs_name) + "_Im"] << data.imag();
  };
};

//measure a vector
template<class OBS>
struct measure_observable_helper<std::complex<double>, OBS, std::vector<std::complex<double> > > {
  void static operate(observableset_t &meas, const char *obs_name, const std::vector<std::complex<double> > &data) {
    std::vector<double> tmparray(data.size());
    for (int i = 0; i < data.size(); ++i) {
      tmparray[i] = data[i].real();
    }
    meas[std::string(obs_name) + "_Re"] << tmparray;
    for (int i = 0; i < data.size(); ++i) {
      tmparray[i] = data[i].imag();
    }
    meas[std::string(obs_name) + "_Im"] << tmparray;
  };
};

/* ****
 * WRAPPER FUNCIONS
 * ****/
template<class SCALAR, class OBS>
void
create_observable(observableset_t &meas, const char *obs_name) {
  create_observable_helper<SCALAR, OBS>::operate(meas, obs_name);
}

template<class SCALAR>
void
measure_scalar_observable(observableset_t &meas, const char *obs_name, const SCALAR &data) {
  measure_observable_helper<SCALAR, RealObservable, SCALAR>::operate(meas, obs_name, data);
}

template<class SCALAR>
void
measure_vector_observable(observableset_t &meas, const char *obs_name, const std::vector<SCALAR> &data) {
  measure_observable_helper<SCALAR, RealVectorObservable, std::vector<SCALAR> >::operate(meas, obs_name, data);
}

template<class SCALAR>
void
measure_simple_vector_observable(observableset_t &meas, const char *obs_name, const std::vector<SCALAR> &data) {
  measure_observable_helper<SCALAR, SimpleRealVectorObservable, std::vector<SCALAR> >::operate(meas, obs_name, data);
}

template<class SCALAR, int DIMENSION>
void
measure_simple_multiarray_observable(observableset_t &meas,
                                     const char *obs_name,
                                     const boost::multi_array<SCALAR, DIMENSION> &data) {
  measure_multiarray_helper<SCALAR, DIMENSION, SimpleRealVectorObservable>::operate(meas, obs_name, data);
}

/* ****
 * UTILITIES
 * ****/
template<typename T>
std::vector<T> to_std_vector(const std::valarray<T> &array) {
  std::vector<T> std_vec(array.size());
  for (int i = 0; i < array.size(); ++i) {
    std_vec[i] = array[i];
  }
  return std_vec;
}

template<typename T, unsigned long N>
std::vector<T> to_std_vector(const boost::multi_array<T, N> &array) {
  std::vector<T> std_vec(array.num_elements());
  const T *it = array.origin();
  for (int count = 0; count < array.num_elements(); ++count) {
    std_vec[count] = *it;
    ++it;
  }
  return std_vec;
}

//multiprecision version
template<typename T, unsigned long N>
std::vector<double> to_double_std_vector(const boost::multi_array<T, N> &array) {
  std::vector<double> std_vec(array.num_elements());
  const T *it = array.origin();
  for (int count = 0; count < array.num_elements(); ++count) {
    std_vec[count] = it->template convert_to<double>();
    ++it;
  }
  return std_vec;
}

//multiprecision version
template<typename T, unsigned long N>
std::vector<std::complex<double> > to_complex_double_std_vector(const boost::multi_array<T, N> &array) {
  std::vector<std::complex<double> > std_vec(array.num_elements());
  const T *it = array.origin();
  for (int count = 0; count < array.num_elements(); ++count) {
    std_vec[count] = convert_to_complex(*it);//->template convert_to<std::complex<double> >();
    ++it;
  }
  return std_vec;
}

template<typename T, typename S>
void operator/=(std::vector<T> &vec, const S& val) {
  for (typename std::vector<T>::iterator it = vec.begin(); it != vec.end(); ++it) {
    *it /= val;
  }
}

template<typename T1, typename T2>
void operator/=(std::vector<T1> &vec1, const std::vector<T2> &vec2) {
  assert(vec1.size() == vec2.size());
  typename std::vector<T2>::const_iterator it2 = vec2.begin();
  for (typename std::vector<T1>::iterator it = vec1.begin(); it != vec1.end(); ++it, ++it2) {
    *it /= *it2;
  }
}

