template<typename SCALAR>
SCALAR mycast(std::complex<double>);

template<typename SW, typename OBS>
MeasStaticObs<SW, OBS>::MeasStaticObs(SW &sw, const operator_container_t &operators)
    : num_brakets(sw.get_num_brakets()),
      state_bak(sw.get_state()),
      sw_(sw),
      ops_(operators) {
  const int edge_pos = sw.get_n_window();
  sw.move_left_edge_to(operators, edge_pos);
  sw.move_right_edge_to(operators, edge_pos);
}

template<typename SW, typename OBS>
void
MeasStaticObs<SW, OBS>::perform_meas(const std::vector<OBS> &obs, std::vector<EXTENDED_COMPLEX> &result) const {
  const int num_obs = obs.size();
  result.resize(num_obs);
  std::fill(result.begin(), result.end(), 0.0);
  for (int braket = 0; braket < num_brakets; ++braket) {
    if (sw_.get_bra(braket).invalid() || sw_.get_ket(braket).invalid()) {
      continue;
    }
    for (int i_obs = 0; i_obs < num_obs; ++i_obs) {
      typename SW::BRAKET_TYPE ket(sw_.get_ket(braket));
      sw_.get_p_model()->apply_op_ket(obs[i_obs], ket);
      result[i_obs] += static_cast<EXTENDED_COMPLEX>(sw_.get_p_model()->product(sw_.get_bra(braket), ket));
    }
  }
}

template<typename SW, typename OBS>
MeasStaticObs<SW, OBS>::~MeasStaticObs() {
  sw_.restore_state(ops_, state_bak);
}
