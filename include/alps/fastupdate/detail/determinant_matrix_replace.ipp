#include "../determinant_matrix.hpp"

namespace alps {
  namespace fastupdate {

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    Scalar
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::try_replace_cdagg(
            const CdaggerOp& old_cdagg,
            const CdaggerOp& new_cdagg
    ) {
      check_state(waiting);
      state_ = try_replace_cdagg_called;
      const int nop = inv_matrix_.size1();
      new_cdagg_ = new_cdagg;
      old_cdagg_ = old_cdagg;

      //move the target row to the end
      const int pos = find_cdagg(old_cdagg);
      if (pos != nop-1) {
        swap_cdagg_op(pos, nop-1);
      }

      //compute the values of new elements
      G_j_n_.resize(nop, 1);
      for(int i=0; i<nop; ++i) {
        G_j_n_(i, 0) = p_gf_->operator()(c_ops_[i], new_cdagg);
      }

      //permutation sign
      const int diff =
        std::abs(
          std::distance(cdagg_op_pos_.lower_bound(operator_time(old_cdagg_)), cdagg_op_pos_.end())-
          std::distance(cdagg_op_pos_.lower_bound(operator_time(new_cdagg_)), cdagg_op_pos_.end())
        );
      perm_rat_ = (diff%2==0 ? 1 : -1);
      if (operator_time(new_cdagg_) > operator_time(old_cdagg_)) {
        perm_rat_ *= -1;
      }

      //std::cout << "computing " << det_rat_ << std::endl;
      //std::cout << "inv_matrix " << inv_matrix_ << std::endl;
      //std::cout << "G_j_n " << G_j_n_ << std::endl;
      det_rat_ = compute_det_ratio_relace_last_col(inv_matrix_, G_j_n_);
      return (1.*perm_rat_)*det_rat_;
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    void
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::perform_replace_cdagg() {
      check_state(try_replace_cdagg_called);
      state_ = waiting;

      const int nop = inv_matrix_.size1();

      compute_inverse_matrix_replace_last_col(inv_matrix_, G_j_n_, det_rat_);
      permutation_row_col_ *= perm_rat_;
      cdagg_ops_[nop-1] = new_cdagg_;
      cdagg_op_pos_.erase(operator_time(old_cdagg_));
      cdagg_op_pos_.insert(std::make_pair(operator_time(new_cdagg_), nop-1));

      cdagg_ops_set_.erase(old_cdagg_);
      cdagg_ops_set_.insert(new_cdagg_);
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    void
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::reject_replace_cdagg() {
      //do nothing
      check_state(try_replace_cdagg_called);
      state_ = waiting;
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    Scalar
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::try_replace_c(
      const COp& old_c,
      const COp& new_c
    ) {
      check_state(waiting);
      state_ = try_replace_c_called;
      const int nop = inv_matrix_.size1();
      new_c_ = new_c;
      old_c_ = old_c;

      //move the target row to the end
      const int pos = find_c(old_c);
      if (pos != nop-1) {
        swap_c_op(pos, nop-1);
      }

      //compute the values of new elements
      G_n_j_.resize(1, nop);
      for(int i=0; i<nop; ++i) {
        G_n_j_(0, i) = p_gf_->operator()(new_c, cdagg_ops_[i]);
      }

      //permutation sign
      const int diff =
        std::abs(
          std::distance(cop_pos_.lower_bound(operator_time(old_c_)), cop_pos_.end())-
          std::distance(cop_pos_.lower_bound(operator_time(new_c_)), cop_pos_.end())
        );
      perm_rat_ = (diff%2==0 ? 1 : -1);
      if (operator_time(new_c_) > operator_time(old_c_)) {
        perm_rat_ *= -1;
      }

      det_rat_ = compute_det_ratio_relace_last_row(inv_matrix_, G_n_j_);
      return (1.*perm_rat_)*det_rat_;
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    void
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::perform_replace_c() {
      check_state(try_replace_c_called);
      state_ = waiting;

      const int nop = inv_matrix_.size1();

      compute_inverse_matrix_replace_last_row(inv_matrix_, G_n_j_, det_rat_);
      permutation_row_col_ *= perm_rat_;
      c_ops_[nop-1] = new_c_;
      cop_pos_.erase(operator_time(old_c_));
      cop_pos_.insert(std::make_pair(operator_time(new_c_), nop-1));

      c_ops_set_.erase(old_c_);
      c_ops_set_.insert(new_c_);
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    void
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::reject_replace_c() {
      //do nothing
      check_state(try_replace_c_called);
      state_ = waiting;
    }

  }
}
