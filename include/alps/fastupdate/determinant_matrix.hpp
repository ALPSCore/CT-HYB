/**
 * Copyright (C) 2016 by Hiroshi Shinaoka <h.shinaoka@gmail.com>
 */
#pragma once

#include <map>
#include <algorithm>
#include <boost/range/algorithm/adjacent_find.hpp>

#include <Eigen/Dense>

#include "fastupdate_formula.hpp"
#include "./detail/util.hpp"

#include "determinant_matrix_base.hpp"

namespace alps {
  namespace fastupdate {

    /**
     * CdaggerOp and COp must have the following functionalities
     *   CdaggerOp::itime_type, COp::itime_type the type of time
     *
     *  Function itime_type operator_time(const CdaggerOp&) and operator_time(const COp&)
     *  Function int operator_flavor(const CdaggerOp&) and operator_flavor(const COp&)
     */
    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    class DeterminantMatrix
            : public DeterminantMatrixBase<
                    Scalar,
                    GreensFunction,
                    CdaggerOp,
                    COp,
                    DeterminantMatrix<Scalar, GreensFunction, CdaggerOp, COp>
                  >
    {
    private:
      typedef DeterminantMatrixBase<
        Scalar,
        GreensFunction,
        CdaggerOp,
        COp,
        DeterminantMatrix<Scalar, GreensFunction, CdaggerOp, COp>
      > Base;
      typedef typename CdaggerOp::itime_type itime_t;
      typedef std::vector<CdaggerOp> cdagg_container_t;
      typedef std::vector<COp> c_container_t;
      typedef std::map<itime_t,int> operator_map_t;
      typedef boost::multi_index::multi_index_container<CdaggerOp> cdagg_set_t;
      typedef boost::multi_index::multi_index_container<COp> c_set_t;

      typedef typename cdagg_container_t::iterator cdagg_it;
      typedef typename c_container_t::iterator c_it;

      typedef Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> eigen_matrix_t;

      enum DeterminantMatrixState {
        waiting = 0,
        try_add_called = 1,
        try_rem_called = 2,
        try_rem_add_called = 3,
        try_replace_cdagg_called = 4,
        try_replace_c_called = 5
      };

      enum UpdateMode {
        do_nothing,
        add,
        rem,
        rem_add,
        replace_cdagg,
        replace_c,
        invalid_operation//e.g., try to change the num of rows and cols differently
      };

    public:
      DeterminantMatrix(
        boost::shared_ptr<GreensFunction> p_gf
      );

      template<typename CdaggCIterator>
      DeterminantMatrix(
        boost::shared_ptr<GreensFunction> p_gf,
        CdaggCIterator first,
        CdaggCIterator last
      );

      //size of matrix
      inline int size() const {return inv_matrix_.size1();}

      //Getter
      inline const cdagg_container_t& get_cdagg_ops() const { return cdagg_ops_; }
      inline const c_container_t& get_c_ops() const { return c_ops_; }

      /**
       * Return a reference to a set of time-ordered creation operators
       */
      const cdagg_set_t& get_cdagg_ops_set() const {
        return cdagg_ops_set_;
      }

      /**
       * Return a reference to a set of time-ordered creation operators for a given block
       */
      const cdagg_set_t& get_cdagg_ops_set(int block) const {
        assert(block==0);
        return get_cdagg_ops_set();
      }

      /**
       * Return a reference to a set of time-ordered annihilation operators
       */
      const c_set_t& get_c_ops_set() const {
        return c_ops_set_;
      }

      /**
       * Return a reference to a set of time-ordered annihilation operators for a given block
       */
      const c_set_t& get_c_ops_set(int block) const {
        assert(block==0);
        return get_c_ops_set();
      }

      /**
       * Compute determinant. This may suffer from overflow
       */
      inline Scalar compute_determinant() const {
        return (1.*permutation_row_col_)/inv_matrix_.determinant();
      }

      std::vector<Scalar> compute_determinant_as_product() const {
        if (inv_matrix_.size1() == 0) {
          std::vector<Scalar> r(1);
          r[0] = 1.0;
          return r;
        } else {
          const std::vector<Scalar>& vec = detail::lu_product<Scalar>(inv_matrix_.block());
          std::vector<Scalar> r(vec.size());
          std::transform(
              vec.begin(), vec.end(), r.begin(),
              std::bind1st(
                  std::divides<Scalar>(), (Scalar) 1.0
              )
          );
          if (r.size()>0) {
            r[0] *= 1.*permutation_row_col_;
          }
          std::sort(r.begin(), r.end(), detail::lesser_by_abs<Scalar>);
          return r;
        }
      }

      /**
       * Compute inverse matrix. The rows and cols may not be time-ordered.
       */
      eigen_matrix_t compute_inverse_matrix() const {
        return eigen_matrix_t(inv_matrix_.block());
      }

      /**
       * Remove some operators and add new operators: no acutual update, just compute determinant ratio
       */
      template<typename CdaggIterator, typename CIterator, typename CdaggIterator2, typename CIterator2>
      Scalar try_update(
        CdaggIterator  cdagg_rem_first,  CdaggIterator  cdagg_rem_last,
        CIterator      c_rem_first,      CIterator      c_rem_last,
        CdaggIterator2 cdagg_add_first,  CdaggIterator2 cdagg_add_last,
        CIterator2     c_add_first,      CIterator2     c_add_last
      );

      /**
       * Remove some operators and add new operators: actual update
       */
      void perform_update();

      /**
       * Remove some operators and add new operators: reject this update
       */
      void reject_update();

      /**
       * Rebuild the matrix from scratch
       */
      Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>
      compute_G_matrix() const;

      /**
       * Rebuild the matrix from scratch
       */
      void rebuild_inverse_matrix();

      /**
       * Compute the inverse matrix for the time-ordered set of operators
       * This could cost O(N^3) because rows and cols are time-ordered if needed
       */
      //eigen_matrix_t compute_inverse_matrix_time_ordered();

      void print_operators() const;

    private:
      DeterminantMatrixState state_;
      UpdateMode  update_mode_;

      //inverse matrix
      ResizableMatrix<Scalar> inv_matrix_;

      //permutation of time-ordering of rows and cols
      int permutation_row_col_;//1 or -1

      //a vector of creation and annihilation operators in the order in which they appear in the rows and cols of the matrix
      cdagg_container_t cdagg_ops_;
      c_container_t c_ops_;

      //Time-ordered set
      cdagg_set_t cdagg_ops_set_;
      c_set_t c_ops_set_;

      boost::shared_ptr<GreensFunction> p_gf_;

      //key: the imaginary time of an operator, the index of row or col in the matrix
      operator_map_t cop_pos_, cdagg_op_pos_;

      //work space and helper
      int perm_rat_, nop_added_;
      bool update_impossible_;
      Scalar det_rat_;
      CdaggerOp new_cdagg_, old_cdagg_;
      COp new_c_, old_c_;
      std::vector<int> rem_cols_, rem_rows_;
      std::vector<std::pair<CdaggerOp,COp> > removed_op_pairs_;

      eigen_matrix_t G_n_n_, G_n_j_, G_j_n_;
      ReplaceHelper<Scalar,eigen_matrix_t,eigen_matrix_t,eigen_matrix_t> replace_helper_;

      /*
       * Private auxially functions
       */
    private:
      inline void check_state(DeterminantMatrixState state) const {
        if (state_ != state) {
          throw std::logic_error("Error: the system is not in a correct state!");
        }
      }

      /*
       * Try to remove some operators and add new operators
       * This function actually remove and insert operators in cdagg_ops_, c_ops_ but does not update the matrix.
       * After calling try_add(), either of perform_add() or reject_add() must be called.
       */
      template<typename CdaggCIterator>
      Scalar try_add(
        CdaggCIterator cdagg_c_add_first,
        CdaggCIterator cdagg_c_add_last
      );

      void perform_add();

      void reject_add();

      /**
       * Try to remove some operators and add new operators
       * This function actually remove and insert operators in cdagg_ops_, c_ops_ but does not update the matrix
       */
      template<typename CdaggCIterator, typename CdaggCIterator2>
      Scalar try_remove_add(
        CdaggCIterator  cdagg_c_rem_first,
        CdaggCIterator  cdagg_c_rem_last,
        CdaggCIterator2 cdagg_c_add_first,
        CdaggCIterator2 cdagg_c_add_last
      );

      /**
       *  Remove some operators and add new operators
       *  This function actually update the matrix
       */
      void perform_remove_add();

      void reject_remove_add();

      /**
       * Try to remove some operators
       * This function actually remove and insert operators in cdagg_ops_, c_ops_ but does not update the matrix
       */
      template<typename CdaggCIterator>
      Scalar try_remove(
        CdaggCIterator cdagg_c_rem_first,
        CdaggCIterator cdagg_c_rem_last
      );

      void perform_remove();

      void reject_remove();

      /**
       * Try to replace a creation operator
       */
      Scalar try_replace_cdagg(
        const CdaggerOp & old_cdagg,
        const CdaggerOp & new_cdagg
      );

      void perform_replace_cdagg();

      void reject_replace_cdagg();

      /**
       * Try to replace an annihilation operator
       */
      Scalar try_replace_c(
        const COp & old_c,
        const COp & new_c
      );

      void perform_replace_c();

      void reject_replace_c();

      /** swap cols of the matrix (and the rows of the inverse matrix)*/
      void swap_cdagg_op(int col1, int col2);

      /** swap rows of the matrix (and the cols of the inverse matrix)*/
      void swap_c_op(int row1, int row2);

      /** return if once can insert given operators. Note: duplicate members are not allowed in any configuration. */
      template<typename CdaggCIterator>
      bool insertion_possible(
        CdaggCIterator first,
        CdaggCIterator last
      );

      /** return if once can remove given operators, i.e, if they actually exist in the configuraiton */
      template<typename CdaggCIterator>
      bool removal_insertion_possible(
        CdaggCIterator first_removal,
        CdaggCIterator last_removal,
        CdaggCIterator first_insertion,
        CdaggCIterator last_insertion
      );

      /** return if once can remove given operators */
      template<typename CdaggCIterator>
      bool removal_possible(
        CdaggCIterator first_removal,
        CdaggCIterator last_removal
      ) const;

      /** add new operators and keeping the inverse matrix unchanged */
      template<typename Iterator>
      int add_new_operators(Iterator first,Iterator last);

      /** add operators and keeping the inverse matrix unchanged */
      int remove_last_operators(int num_operators_remove);

      /** remove excess operators, which were inserted by add_new_operators() */
      void remove_excess_operators();

      inline Scalar compute_g(int row, int col) const {return p_gf_->operator()(c_ops_[row], cdagg_ops_[col]); }

      /** return if there is an operator at a given time */
      inline bool exist_cdagg(const CdaggerOp& cdagg) const {
        return exist(operator_time(cdagg));
      }

      /** return if there is an operator at a given time */
      inline bool exist_c(const COp& c) const {
        return exist(operator_time(c));
      }

      /** return if there is an operator at a given time */
      inline bool exist(itime_t time) const {
        return cop_pos_.find(time)!=cop_pos_.end() || cdagg_op_pos_.find(time)!=cdagg_op_pos_.end();
      }

      inline int find_cdagg(const CdaggerOp& cdagg) const {
        assert(cdagg_op_pos_.find(operator_time(cdagg))!=cdagg_op_pos_.end());
        return cdagg_op_pos_.at(operator_time(cdagg));
      }

      inline int find_c(const COp& c) const {
        assert(cop_pos_.find(operator_time(c))!=cop_pos_.end());
        return cop_pos_.at(operator_time(c));
      }

      void sanity_check() const;

    };
  }
}

#include "./detail/determinant_matrix.ipp"
#include "./detail/determinant_matrix_add.ipp"
#include "./detail/determinant_matrix_remove.ipp"
#include "./detail/determinant_matrix_remove_add.ipp"
#include "./detail/determinant_matrix_replace.ipp"
