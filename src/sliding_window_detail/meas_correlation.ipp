#include "../sliding_window.hpp"

template<typename SW, typename OBS>
MeasCorrelation<SW,OBS>::MeasCorrelation(const std::vector<std::pair<OBS,OBS> >& correlators, int num_tau_points)
  : num_correlators_(correlators.size()),
    num_tau_points_(num_tau_points),
    num_win_(2*(num_tau_points_-1)),
    right_edge_pos_(num_win_/2),
    left_edge_pos_(num_win_/2+2*(num_tau_points-1))
{
  std::set<OBS> left_obs_set, right_obs_set;
  for (typename std::vector<std::pair<OBS,OBS> >::const_iterator it=correlators.begin(); it!=correlators.end(); ++it) {
    left_obs_set.insert(it->first);
    right_obs_set.insert(it->second);
  }

  obs_pos_in_unique_set.resize(correlators.size());
  int icorr = 0;
  for (typename std::vector<std::pair<OBS,OBS> >::const_iterator it=correlators.begin(); it!=correlators.end(); ++it, ++icorr) {
    obs_pos_in_unique_set[icorr].first  = std::distance(left_obs_set.begin(),  left_obs_set.find(it->first));
    obs_pos_in_unique_set[icorr].second = std::distance(right_obs_set.begin(), right_obs_set.find(it->second));
  }

  left_obs_unique_list = std::vector<OBS>(left_obs_set.begin(), left_obs_set.end());
  right_obs_unique_list = std::vector<OBS>(right_obs_set.begin(), right_obs_set.end());
}

template<typename SW, typename OBS>
void
MeasCorrelation<SW,OBS>::perform_meas(SW& sw, const operator_container_t& operators, boost::multi_array<EXTENDED_COMPLEX,2>& result) const {
  namespace bll = boost::lambda;

  typedef typename SW::BRAKET_TYPE BRAKET_TYPE;
  typedef typename operator_container_t::iterator OP_IT_TYPE;

  const int num_braket = sw.get_num_brakets();

  //place the window somewhere in the middle of [0,beta]
  typename SW::state_t state_bak = sw.get_state();
  sw.set_window_size(num_win_, operators, num_win_/2);
  sw.move_left_edge_to(operators, left_edge_pos_);
  assert(left_edge_pos_-right_edge_pos_==num_tau_points_-1);

  //make a list of tau points
  std::vector<double> tau_points(num_tau_points_);
  for (int itau=0; itau<num_tau_points_; ++itau) {
    tau_points[itau] = sw.get_tau_edge(right_edge_pos_+2*itau);
  }

  //Find out operators in imaginary time segments
  //Note: look at the differences between "<" and "<="
  //This avoids a double-counting problem of operators for the case there is (accidentaly) a creation/annihilation operator on the right top of a tau point.
  std::vector<std::pair<OP_IT_TYPE,OP_IT_TYPE> > ops_ranges_from_left(num_tau_points_-1),
    ops_ranges_from_right(num_tau_points_-1);
  for (int itau=0; itau<num_tau_points_-1; ++itau) {
    ops_ranges_from_right[itau] = operators.range(tau_points[itau]<=bll::_1, bll::_1<tau_points[itau+1]);
    ops_ranges_from_left[itau] = operators.range(tau_points[itau]<=bll::_1, bll::_1<=tau_points[itau+1]);
  }

  boost::multi_array<BRAKET_TYPE,2> bra_sector(boost::extents[left_obs_unique_list.size()][num_tau_points_]);
  boost::multi_array<BRAKET_TYPE,2> ket_sector(boost::extents[right_obs_unique_list.size()][num_tau_points_]);

  result.resize(boost::extents[num_correlators_][num_tau_points_]);
  std::fill(result.origin(),result.origin()+result.num_elements(),0.0);

  for (int ibraket=0; ibraket<num_braket; ++ibraket) {
    //evolve a ket from right-hand side with placing the right operator at the right most tau point.
    for (int i_obs=0; i_obs<right_obs_unique_list.size(); ++i_obs) {
      BRAKET_TYPE ket(sw.get_ket(ibraket));
      sw.get_p_model()->apply_op_ket(right_obs_unique_list[i_obs], ket);

      ket_sector[i_obs][0] = ket;
      for (int itau=1; itau<num_tau_points_; ++itau) {
        SW::evolve_ket(*sw.get_p_model(), ket, ops_ranges_from_right[itau-1], tau_points[itau-1], tau_points[itau]);
        ket_sector[i_obs][itau] = ket;
      }
    }

    //evolve a bra from left-hand side
    //At each tau point, we apply the left operator of the correlator on the bra.
    BRAKET_TYPE bra(sw.get_bra(ibraket));
    for (int itau=num_tau_points_-1; itau>=0; --itau) {
      if (itau != num_tau_points_-1) {
        SW::evolve_bra(*sw.get_p_model(), bra, ops_ranges_from_left[itau], tau_points[itau+1], tau_points[itau]);
      }
      for (int i_obs=0; i_obs<left_obs_unique_list.size(); ++i_obs) {
        bra_sector[i_obs][itau] = bra;
        sw.get_p_model()->apply_op_bra(left_obs_unique_list[i_obs], bra_sector[i_obs][itau]);
      }
    }

    //Now compute correlators
    for (int icorr=0; icorr<num_correlators_; ++icorr) {
      for (int itau=0; itau<num_tau_points_; ++itau) {
        result[icorr][itau] += sw.get_p_model()->product(
          bra_sector[obs_pos_in_unique_set[icorr].first][itau],
          ket_sector[obs_pos_in_unique_set[icorr].second][itau]
        );
      }
    }
  }

  sw.restore_state(operators, state_bak);
}

template<typename SW, typename OBS>
MeasCorrelation<SW,OBS>::~MeasCorrelation() {
}

