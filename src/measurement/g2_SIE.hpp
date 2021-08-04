#include <vector>
#include "../common/util.hpp"


class SamplingFreqsG2SIE {
  SamplingFreqsG2SIE(
    const std::vector<int> &v,
    const std::vector<int> &vp,
    const std::vector<int> &w
  ) : n_smpl_freqs(v.size()),
      this->v(v),
      this->vp(vp),
      this->w(w)
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

  };

  public:
    int n_smpl_freqs;
    std::vector<int> v, vp, w;//ph convention
    std::vector<int> v1, v2, v3, v4;//four-fermion convention
    std::vector<int> v_lambda;//lambda
    std::vector<int> v_varphi;//phi
};