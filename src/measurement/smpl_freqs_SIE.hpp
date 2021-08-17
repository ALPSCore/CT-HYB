#pragma once

#include <vector>
#include <fstream>
#include "../common/util.hpp"


class SamplingFreqsG2SIE {
  public:
  SamplingFreqsG2SIE(
    const std::vector<int> &v,
    const std::vector<int> &vp,
    const std::vector<int> &w
  ) : n_smpl_freqs(v.size()),
      v(v),
      vp(vp),
      w(w)
  {
    check_true(v.size() == vp.size() && vp.size() == w.size());
    check_true(is_fermionic(v));
    check_true(is_fermionic(vp));
    check_true(is_bosonic(w));

    v1.resize(n_smpl_freqs);
    v2.resize(n_smpl_freqs);
    v3.resize(n_smpl_freqs);
    v4.resize(n_smpl_freqs);

    // To four-fermion convention
    for (int f=0; f<n_smpl_freqs; ++f) {
      v1[f] = v[f] + w[f];
      v2[f] = v[f];
      v3[f] = vp[f];
      v4[f] = vp[f] + w[f];
      check_true(v1[f]-v2[f]+v3[f]-v4[f] == 0);
    }

    // Sampling frequencies for lambda and varphi
    v_lambda.clear();
    v_varphi.clear();
    for (int f=0; f<n_smpl_freqs; ++f) {
      v_lambda.push_back(v1[f] - v2[f]);
      v_lambda.push_back(v1[f] - v4[f]);
      v_varphi.push_back(v1[f] + v3[f]);
    }
    v_lambda = unique(v_lambda);
    v_varphi = unique(v_varphi);

    // Sampling frequencies for eta
    auto vw_eta = std::vector<std::pair<int,int>>();
    for (int f=0; f<n_smpl_freqs; ++f) {
      vw_eta.push_back({v1[f], v1[f]-v2[f]});
      vw_eta.push_back({v3[f], v2[f]-v1[f]});
      vw_eta.push_back({v1[f], v1[f]-v4[f]});
      vw_eta.push_back({v3[f], v4[f]-v1[f]});
    }
    vw_eta = unique(vw_eta);
    v_eta.clear();
    w_eta.clear();
    for (auto vw: vw_eta) {
      v_eta.push_back(std::get<0>(vw));
      w_eta.push_back(std::get<1>(vw));
    }

    // Sampling frequencies for gamma
    auto vw_gamma = std::vector<std::pair<int,int>>();
    for (int f=0; f<n_smpl_freqs; ++f) {
      vw_gamma.push_back({ v1[f],  v1[f]+v3[f]});
      vw_gamma.push_back({-v4[f], -v2[f]-v4[f]});
    }
    vw_gamma = unique(vw_gamma);
    v_gamma.clear();
    w_gamma.clear();
    for (auto vw: vw_gamma) {
      v_gamma.push_back(std::get<0>(vw));
      w_gamma.push_back(std::get<1>(vw));
    }

  };

  int n_smpl_freqs;
  std::vector<int> v, vp, w; //ph convention
  std::vector<int> v1, v2, v3, v4; //four-fermion convention
  std::vector<int> v_lambda; //lambda
  std::vector<int> v_varphi; //phi
  std::vector<int> v_eta; //eta (fermion)
  std::vector<int> w_eta; //eta (boson)
  std::vector<int> v_gamma; //gamma (fermion)
  std::vector<int> w_gamma; //gamma (boson)
};

inline
SamplingFreqsG2SIE load_smpl_freqs_SIE(const std::string& file) {
  std::ifstream f(file);
  std::vector<int> v;
  std::vector<int> vp;
  std::vector<int> w;

  if (!f.is_open()) {
    throw std::runtime_error("File at " + file + ", which should contain the list of Matsubara frequencies for G2 measurement cannot be read or does not exit.");
  }

  int num_freqs;
  f >> num_freqs;

  for (int i=0; i<num_freqs; ++i) {
    int j, v_, vp_, w_;
    f >> j >> w_ >> vp_ >> w_;
    if (i != j) {
      throw std::runtime_error("The first column has a wrong value in " + file + ".");
    }
    v.push_back(v_);
    vp.push_back(vp_);
    w.push_back(w_);
  }

  return SamplingFreqsG2SIE(v, vp, w);
}