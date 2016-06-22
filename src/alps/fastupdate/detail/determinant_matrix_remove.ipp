#include "../determinant_matrix.hpp"

namespace alps {
  namespace fastupdate {

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    template<typename CdaggCIterator>
    Scalar
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::try_remove(
      CdaggCIterator cdagg_c_rem_first,
      CdaggCIterator cdagg_c_rem_last
    ) {
      check_state(waiting);
      state_ = try_rem_called;

      const int nop_rem = std::distance(cdagg_c_rem_first, cdagg_c_rem_last);
      const int nop = inv_matrix_.size1();

      if (!removal_possible(cdagg_c_rem_first, cdagg_c_rem_last)) {
        return 0.0;
      }

      //move all rows and cols to be removed to the last
      rem_cols_.resize(nop_rem);
      rem_rows_.resize(nop_rem);
      CdaggCIterator it = cdagg_c_rem_first;
      for (int iop=0; iop<nop_rem; ++iop) {
        rem_cols_[iop] = find_cdagg(it->first);
        rem_rows_[iop] = find_c(it->second);
        ++it;
      }
      std::sort(rem_cols_.begin(), rem_cols_.end());
      std::sort(rem_rows_.begin(), rem_rows_.end());

      for (int swap=0; swap<nop_rem; ++swap) {
        swap_cdagg_op(rem_cols_[nop_rem-1-swap], nop-1-swap);
        swap_c_op(rem_rows_[nop_rem-1-swap], nop-1-swap);
      }

      //remember what operators are removed
      removed_op_pairs_.resize(0);
      removed_op_pairs_.reserve(nop_rem);
      for (int iop=0; iop<nop_rem; ++iop) {
        removed_op_pairs_.push_back(
          std::make_pair(
            cdagg_ops_[nop-1-iop],
            c_ops_[nop-1-iop]
          )
        );
        cdagg_ops_set_.erase(cdagg_ops_[nop-1-iop]);
        c_ops_set_.erase(c_ops_[nop-1-iop]);
      }

      //Remove the last operators and add new operators
      perm_rat_ = remove_last_operators(nop_rem);

      return static_cast<double>(perm_rat_)*compute_det_ratio_down(nop_rem, inv_matrix_);
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    void
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::perform_remove() {
      check_state(try_rem_called);
      state_ = waiting;

      const int nop_rem = removed_op_pairs_.size();
      permutation_row_col_ *= perm_rat_;
      compute_inverse_matrix_down(nop_rem, inv_matrix_);
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    void
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::reject_remove() {
      check_state(try_rem_called);
      state_ = waiting;

      //insert the removed operators back
      add_new_operators(removed_op_pairs_.rbegin(), removed_op_pairs_.rend());
    }

  }
}
