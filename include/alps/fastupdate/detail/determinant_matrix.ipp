#include "../determinant_matrix.hpp"

namespace alps {
  namespace fastupdate {

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::DeterminantMatrix(
        boost::shared_ptr<GreensFunction> p_gf
    )
      : Base(p_gf),
        state_(waiting),
        inv_matrix_(0,0),
        permutation_row_col_(1),
        p_gf_(p_gf)
    {
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    template<typename CdaggCIterator>
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::DeterminantMatrix(
      boost::shared_ptr<GreensFunction> p_gf,
      CdaggCIterator first,
      CdaggCIterator last
    )
      : Base(p_gf),
        state_(waiting),
        inv_matrix_(0,0),
        permutation_row_col_(1),
        p_gf_(p_gf)
    {
      try_add(first, last);
      perform_add();
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    template<typename CdaggCIterator>
    bool
      DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::insertion_possible(
      CdaggCIterator first,
      CdaggCIterator last) {

      int iop = 0;
      std::vector<itime_t> times_add(2*std::distance(first, last));
      for (CdaggCIterator it=first; it!=last; ++it) {
        const itime_t t1 = times_add[2*iop] = operator_time(it->first);
        const itime_t t2 = times_add[2*iop+1] = operator_time(it->second);

        if (exist(t1) || exist(t2)) {
          return false;
        }
        ++iop;
      }
      std::sort(times_add.begin(), times_add.end());
      if (boost::adjacent_find(times_add) != times_add.end()) {
        return false;
      }
      return true;
    };

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    template<typename CdaggCIterator>
    bool
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::removal_insertion_possible(
      CdaggCIterator first_removal,
      CdaggCIterator last_removal,
      CdaggCIterator first_insertion,
      CdaggCIterator last_insertion
    ) {
      //check if removal is possible
      const int nop_rem = std::distance(first_removal, last_removal);
      std::vector<itime_t> times_rem(2*nop_rem);
      int iop = 0;
      for (CdaggCIterator it=first_removal; it!=last_removal; ++it) {
        bool tmp = exist_cdagg(it->first) && exist_c(it->second);
        times_rem[2*iop] = operator_time(it->first);
        times_rem[2*iop+1] = operator_time(it->second);
        if (!tmp) {
          throw std::runtime_error("Error in removal_insertion_possible: some operator to be removed is missing!");
        }
        ++iop;
      }

      //check if insertion is possible.
      const int nop_add = std::distance(first_insertion, last_insertion);
      std::vector<itime_t> times_add(2*nop_add);
      iop = 0;
      for (CdaggCIterator it=first_insertion; it!=last_insertion; ++it) {
        const itime_t t1 = times_add[2*iop] = operator_time(it->first);
        const itime_t t2 = times_add[2*iop+1] = operator_time(it->second);
        bool tmp =
          (
            !exist(t1) || (std::find(times_rem.begin(), times_rem.end(), t1)!=times_rem.end())
          ) &&
          (
            !exist(t2) || (std::find(times_rem.begin(), times_rem.end(), t2)!=times_rem.end())
          );
        if (!tmp) {
          return false;
        }
        ++iop;
      }

      //check if there is no duplicate
      std::sort(times_rem.begin(), times_rem.end());
      std::sort(times_add.begin(), times_add.end());
      if (
        boost::adjacent_find(times_add) != times_add.end() ||
        boost::adjacent_find(times_rem) != times_rem.end()
        ) {
        return false;
      }

      return true;
    };

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    template<typename CdaggCIterator>
    bool
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::removal_possible(
      CdaggCIterator first_removal,
      CdaggCIterator last_removal
    ) const {
      //check if removal is possible
      const int nop_rem = std::distance(first_removal, last_removal);
      std::vector<itime_t> times_rem(2*nop_rem);
      bool possible = true;
      int iop = 0;
      for (CdaggCIterator it=first_removal; it!=last_removal; ++it) {
        itime_t t1 = operator_time(it->first);
        itime_t t2 = operator_time(it->second);
        bool tmp = exist(t1) && exist(t2);
        times_rem[2*iop] = t1;
        times_rem[2*iop+1] = t2;
        if (!tmp) {
          throw std::runtime_error("Error in removal_possible: some operator to be removed is missing!");
          possible = false;
          break;
        }
        ++iop;
      }
      if (!possible) {
        return false;
      }

      //check if there is no duplicate
      std::sort(times_rem.begin(), times_rem.end());
      if (boost::adjacent_find(times_rem)!=times_rem.end()) {
        return false;
      }

      return possible;
    };

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    void
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::swap_cdagg_op(int col1, int col2) {
      using std::swap;
      if (col1==col2) return;

      const itime_t t1 = operator_time(cdagg_ops_[col1]);
      const itime_t t2 = operator_time(cdagg_ops_[col2]);
      cdagg_op_pos_[t1] = col2;
      cdagg_op_pos_[t2] = col1;

      //Note we need to swap ROWS of the inverse matrix (not columns)
      inv_matrix_.swap_row(col1, col2);
      swap(cdagg_ops_[col1], cdagg_ops_[col2]);
      permutation_row_col_ *= -1;
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    void
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::swap_c_op(int row1, int row2) {
      using std::swap;
      if (row1==row2) return;

      const itime_t t1 = operator_time(c_ops_[row1]);
      const itime_t t2 = operator_time(c_ops_[row2]);
      cop_pos_[t1] = row2;
      cop_pos_[t2] = row1;

      //Note we need to swap COLS of the inverse matrix (not rows)
      inv_matrix_.swap_col(row1, row2);
      swap(c_ops_[row1], c_ops_[row2]);
      permutation_row_col_ *= -1;
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    template<
      typename Iterator
    >
    int
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::add_new_operators(
      Iterator first,
      Iterator last
    ) {
      std::pair<typename std::map<itime_t,int>::iterator,bool> ret;
      itime_t time_new;
      int perm_diff = 0;
      for (Iterator it=first; it!=last; ++it) {
        const int pos = cdagg_ops_.size();

        cdagg_ops_.push_back(it->first);
        time_new = operator_time(it->first);
        perm_diff += std::distance(cdagg_op_pos_.lower_bound(time_new), cdagg_op_pos_.end());
        ret = cdagg_op_pos_.insert(std::make_pair(operator_time(it->first), pos));
        cdagg_ops_set_.insert(it->first);
        if(ret.second==false) {
          throw std::runtime_error("Something went wrong: cdagg already exists");
        }

        c_ops_.push_back(it->second);
        time_new = operator_time(it->second);
        perm_diff += std::distance(cop_pos_.lower_bound(time_new), cop_pos_.end());
        ret = cop_pos_.insert(std::make_pair(operator_time(it->second), pos));
        c_ops_set_.insert(it->second);
        if(ret.second==false) {
          throw std::runtime_error("Something went wrong: c operator already exists");
        }
      }

      //sanity_check();
      return perm_diff%2==0 ? 1 : -1;
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    int
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::remove_last_operators(int num_operators_remove) {
      const int num_ops_remain = cdagg_ops_.size() - num_operators_remove;
      assert(num_ops_remain >= 0);
      if (num_ops_remain < 0) {
        throw std::logic_error("num_ops_remain < 0");
      }

      //Remove the last operators one by one
      int perm_diff = 0;
      for (int iop=0; iop<num_operators_remove; ++iop) {
        const itime_t t1 = operator_time(c_ops_.back());
        perm_diff += std::distance(cop_pos_.lower_bound(t1), cop_pos_.end());
        cop_pos_.erase(t1);
        c_ops_set_.erase(c_ops_.back());

        const itime_t t2 = operator_time(cdagg_ops_.back());
        perm_diff += std::distance(cdagg_op_pos_.lower_bound(t2), cdagg_op_pos_.end());
        cdagg_op_pos_.erase(t2);
        cdagg_ops_set_.erase(cdagg_ops_.back());

        c_ops_.pop_back();
        cdagg_ops_.pop_back();
      }

      return perm_diff%2==0 ? 1 : -1;
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    void
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::remove_excess_operators() {
      const int nop_rem = cdagg_op_pos_.size() - inv_matrix_.size1(); //number of excess operators to be removed
      const int offset = inv_matrix_.size1();

      //remove operators from std::map<operator_time,int>
      for (int iop=0; iop<nop_rem; ++iop) {
        cop_pos_.erase(operator_time(c_ops_[iop+offset]));
        c_ops_set_.erase(c_ops_[iop+offset]);
        cdagg_op_pos_.erase(operator_time(cdagg_ops_[iop+offset]));
        cdagg_ops_set_.erase(cdagg_ops_[iop+offset]);
      }

      cdagg_ops_.resize(offset);
      c_ops_.resize(offset);

      sanity_check();
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    void
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::sanity_check() const {
#ifndef NDEBUG
      check_state(waiting);

      //print_operators();

      const int num_ops = cdagg_ops_.size();
      const int mat_rank = inv_matrix_.size1();

      assert(mat_rank==num_ops);
      assert(cdagg_ops_.size()==c_ops_.size());
      assert(cop_pos_.size()==num_ops);
      assert(cdagg_op_pos_.size()==num_ops);

      for (typename std::map<itime_t,int>::const_iterator it=cop_pos_.begin(); it!=cop_pos_.end(); ++it) {
        assert(it->second<num_ops);
        assert(operator_time(c_ops_[it->second])==it->first);
      }

      for (typename std::map<itime_t,int>::const_iterator it=cdagg_op_pos_.begin(); it!=cdagg_op_pos_.end(); ++it) {
        assert(it->second<num_ops);
        assert(operator_time(cdagg_ops_[it->second])==it->first);
      }

      assert(permutation_row_col_ ==
               detail::permutation(c_ops_.begin(), c_ops_.begin()+mat_rank)*
               detail::permutation(cdagg_ops_.begin(), cdagg_ops_.begin()+mat_rank)
      );

      for (int iop=0; iop<mat_rank; ++iop) {
        assert(find_cdagg(cdagg_ops_[iop])<mat_rank);
        assert(find_c(c_ops_[iop])<mat_rank);
        for (int iop2=0; iop2<mat_rank; ++iop2) {
          assert(!detail::my_isnan(inv_matrix_(iop,iop2)));
        }
      }

      assert(cdagg_ops_set_ == cdagg_set_t(cdagg_ops_.begin(), cdagg_ops_.end()));
      assert(c_ops_set_ == c_set_t(c_ops_.begin(), c_ops_.end()));

#endif
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    void
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::rebuild_inverse_matrix() {
      check_state(waiting);

      const int pert_order = cdagg_ops_.size();
      assert(size()==pert_order);

      inv_matrix_.destructive_resize(pert_order, pert_order);
      for (int j=0; j<pert_order; ++j) {
        for (int i=0; i<pert_order; ++i) {
          inv_matrix_(i,j) = p_gf_->operator()(c_ops_[i], cdagg_ops_[j]);
        }
      }
      //std::cout << "matrix " << inv_matrix_ << std::endl;
      //std::cout << "det matrix " << inv_matrix_.safe_determinant() << std::endl;
      inv_matrix_.invert();
      //std::cout << "inv_matrix " << inv_matrix_ << std::endl;

      sanity_check();
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::compute_G_matrix() const {
      check_state(waiting);

      const int pert_order = cdagg_ops_.size();
      assert(size()==pert_order);

      eigen_matrix_t matrix(pert_order, pert_order);
      for (int j=0; j<pert_order; ++j) {
        for (int i = 0; i < pert_order; ++i) {
          matrix(i, j) = p_gf_->operator()(c_ops_[i], cdagg_ops_[j]);
        }
      }
      return matrix;
    }

    /*
    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::compute_inverse_matrix_time_ordered() {
      check_state(waiting);

      const int N = size();
      eigen_matrix_t inv_mat_ordered(N, N);

      //Look up the time-ordered set
      int col = 0;
      for (typename operator_map_t::const_iterator it_c=cop_pos_.begin(); it_c!=cop_pos_.end(); ++it_c) {
        int row = 0;
        for (typename operator_map_t::const_iterator it_cdagg=cdagg_op_pos_.begin(); it_cdagg!=cdagg_op_pos_.end(); ++it_cdagg) {
          inv_mat_ordered(row, col) = inv_matrix_(it_cdagg->second, it_c->second);
          ++row;
        }
        ++col;
      }

      return inv_mat_ordered;
    }
    */

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    void
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::print_operators() const {
      const int N = size();

      for (int iop=0; iop<cdagg_ops_.size(); ++iop) {
        std::cout << "operator at row/col " << iop << " " << operator_time(c_ops_[iop]) << " " << operator_time(cdagg_ops_[iop]) << std::endl;
      }
    }

    template<typename CdaggerOp, typename COp, typename IteratorCdaggerOp, typename IteratorCOp>
    std::vector<std::pair<CdaggerOp,COp> >
    to_operator_pairs(IteratorCdaggerOp cdagg_first, IteratorCOp c_first, int nop) {
      std::vector<std::pair<CdaggerOp,COp> > ops;
      ops.reserve(nop);
      IteratorCdaggerOp it_cdagg = cdagg_first;
      IteratorCOp       it_c     = c_first;
      for (int iop=0; iop<nop; ++iop) {
        ops.push_back(std::make_pair(*it_cdagg, *it_c));
        ++it_c;
        ++it_cdagg;
      }
      return ops;
    };


    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    template<typename CdaggIterator, typename CIterator, typename CdaggIterator2, typename CIterator2>
    Scalar
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::try_update(
      CdaggIterator  cdagg_rem_first,  CdaggIterator  cdagg_rem_last,
      CIterator      c_rem_first,      CIterator      c_rem_last,
      CdaggIterator2 cdagg_add_first,  CdaggIterator2 cdagg_add_last,
      CIterator2     c_add_first,      CIterator2     c_add_last
    ) {
      sanity_check();

      const int n_cdagg_add = std::distance(cdagg_add_first, cdagg_add_last);
      const int n_cdagg_rem = std::distance(cdagg_rem_first, cdagg_rem_last);
      const int n_c_add = std::distance(c_add_first, c_add_last);
      const int n_c_rem = std::distance(c_rem_first, c_rem_last);

      if (n_cdagg_add-n_cdagg_rem != n_c_add-n_c_rem) {
        update_mode_ = invalid_operation;
        return 0.0;
      }

      if (n_cdagg_add==0 && n_cdagg_rem==0 && n_c_add==0 && n_c_rem==0) {
        update_mode_ = do_nothing;
        return 1.0;
      }

      Scalar det_rat;
      if (n_cdagg_add==1 && n_cdagg_rem==1 && n_c_add==0 && n_c_rem==0) {
        update_mode_ = replace_cdagg;
        det_rat = try_replace_cdagg(*cdagg_rem_first, *cdagg_add_first);

      } else if (n_cdagg_add==0 && n_cdagg_rem==0 && n_c_add==1 && n_c_rem==1) {
        update_mode_ = replace_c;
        det_rat = try_replace_c(*c_rem_first, *c_add_first);

      } else if (n_cdagg_add >0 && n_cdagg_rem==0 && n_c_add >0 && n_c_rem==0) {

        update_mode_ = add;
        const std::vector<std::pair<CdaggerOp,COp> >& ops_add =
          to_operator_pairs<CdaggerOp,COp>(cdagg_add_first, c_add_first, n_cdagg_add);
        det_rat = try_add(ops_add.begin(), ops_add.end());

      } else if (n_cdagg_add==0 && n_cdagg_rem >0 && n_c_add==0 && n_c_rem >0) {

        update_mode_ = rem;
        const std::vector<std::pair<CdaggerOp,COp> >& ops_rem =
          to_operator_pairs<CdaggerOp,COp>(cdagg_rem_first, c_rem_first, n_cdagg_rem);
        det_rat = try_remove(ops_rem.begin(), ops_rem.end());

      } else {

        update_mode_ = rem_add;
        const std::vector<std::pair<CdaggerOp,COp> >& ops_add =
          to_operator_pairs<CdaggerOp,COp>(cdagg_add_first, c_add_first, n_cdagg_add);
        const std::vector<std::pair<CdaggerOp,COp> >& ops_rem =
          to_operator_pairs<CdaggerOp,COp>(cdagg_rem_first, c_rem_first, n_cdagg_rem);
        det_rat = try_remove_add(
          ops_rem.begin(), ops_rem.end(),
          ops_add.begin(), ops_add.end()
        );
      }

      assert(!detail::my_isnan(det_rat));
      return det_rat;
    };

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    void
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::perform_update() {
      switch (update_mode_) {
        case do_nothing:
          break;

        case replace_cdagg:
          perform_replace_cdagg();
          break;

        case replace_c:
          perform_replace_c();
          break;

        case add:
          perform_add();
          break;

        case rem:
          perform_remove();
          break;

        case rem_add:
          perform_remove_add();
          break;

        case invalid_operation:
          break;
      }

    };

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    void
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::reject_update() {
      switch (update_mode_) {
        case do_nothing:
          break;

        case replace_cdagg:
          reject_replace_cdagg();
          break;

        case replace_c:
          reject_replace_c();
          break;

        case add:
          reject_add();
          break;

        case rem:
          reject_remove();
          break;

        case rem_add:
          reject_remove_add();
          break;

        case invalid_operation:
          break;
      }
    };

  }
}
