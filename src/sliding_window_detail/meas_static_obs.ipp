#include "../sliding_window.hpp"

template<typename SW>
MeasStaticObs<SW>::MeasStaticObs(SW& sw, const typename SW::operators_t &operators)
  : num_brakets(sw.get_num_brakets()),
    state_bak(sw.get_state()),
    sw_(sw),
    ops_(operators)
{
  const int edge_pos = sw.get_n_window();
  sw.move_left_edge_to(operators, edge_pos);
  sw.move_right_edge_to(operators, edge_pos);
}

template<typename SW>
template<typename OBS, typename SCALAR>
void
MeasStaticObs<SW>::perform_meas(const std::vector<OBS>& obs, std::vector<SCALAR>& result) const {
  const int num_obs = obs.size();
  result.resize(num_obs);
  std::fill(result.begin(),result.end(),0.0);
  for (int braket=0; braket<num_brakets; ++braket) {
    if (sw_.get_bra(braket).invalid() || sw_.get_ket(braket).invalid()) {
      continue;
    }
    for (int i_obs=0; i_obs<num_obs; ++i_obs) {
      typename SW::BRAKET_TYPE ket(sw_.get_ket(braket));
      sw_.get_p_model()->apply_op_ket(obs[i_obs], ket);
      result[i_obs] += mycast<SCALAR>(sw_.get_p_model()->product(sw_.get_bra(braket), ket));
    }
  }
}

template<typename SW>
MeasStaticObs<SW>::~MeasStaticObs() {
  sw_.restore_state(ops_, state_bak);
}
