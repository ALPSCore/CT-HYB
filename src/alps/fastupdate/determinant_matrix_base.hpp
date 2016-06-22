/**
 * Copyright (C) 2016 by Hiroshi Shinaoka <h.shinaoka@gmail.com>
 */
#pragma once

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/sequenced_index.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/identity.hpp>
#include <boost/shared_ptr.hpp>

#include <Eigen/Dense>

namespace alps {
  namespace fastupdate {

    /**
     * CdaggerOp and COp must have the following functionalities
     *   CdaggerOp::itime_type, COp::itime_type the type of time
     *
     *  itime_type operator_time(const CdaggerOp&)
     *  itime_type operator_time(const COp&)
     *  int        operator_flavor(const CdaggerOp&)
     *  int        operator_flavor(const COp&)
     *
     *  For an instance of GreenFunction gf,
     *  Scalar gf.operator()(const COp&, const CdaggerOp&)
     *  bool   gf.is_connected(int flavor1, int flavor2);
     *  int    gf.num_flavors();
     */
    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp,
      typename Derived
    >
    class DeterminantMatrixBase {
    public:
      typedef Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> eigen_matrix_t;
      typedef std::vector<CdaggerOp> cdagg_container_t;
      typedef std::vector<COp> c_container_t;
      typedef boost::multi_index::multi_index_container<CdaggerOp> cdagg_set_t;
      typedef boost::multi_index::multi_index_container<COp> c_set_t;

      DeterminantMatrixBase(
        boost::shared_ptr<GreensFunction>& p_gf
      ) {};

      template<typename CdaggCIterator>
      DeterminantMatrixBase(
        boost::shared_ptr<GreensFunction>& p_gf,
        CdaggCIterator first,
        CdaggCIterator last
      ) {};

      /** size of matrix */
      int size() const;

      /** return if the matrix is singular */
      bool is_singular() const;

      /** size of block matrix */
      int block_matrix_size(int block) const;

      /** number of blocks */
      int num_blocks() const;

      /** flavors belonging to a given block */
      const std::vector<int>& flavors(int block) const;

      /** return the index of the block to which a given flavor belongs to */
      const std::vector<int>& block_belonging_to(int flavor) const;

      /**
       * Return creation operators in the same order as in the result of compute_inverse_matrix
       * The behavior is not unspecified when it's called during in an update, i.e., after calling update().
       */
      const cdagg_container_t& get_cdagg_ops() const;

      /**
       * Return annihilation operators in the same order as in the result of compute_inverse_matrix
       * The behavior is not unspecified when it's called during in an update, i.e., after calling update().
       */
      const c_container_t& get_c_ops() const;

      /**
       * Similar to get_cdagg_ops (without flattening over blocks)
       */
      const cdagg_container_t& get_cdagg_ops(int block) const;

      /**
       * Similar to get_c_ops (without flattening over blocks)
       */
      const c_container_t& get_c_ops(int block) const;

      /**
       * Return a reference to a set of time-ordered creation operators
       */
      const cdagg_set_t& get_cdagg_ops_set() const;

      /**
       * Return a reference to a set of time-ordered creation operators for a given block
       */
      const cdagg_set_t& get_cdagg_ops_set(int block) const;

      /**
       * Return a reference to a set of time-ordered annihilation operators
       */
      const c_set_t& get_c_ops_set() const;

      /**
       * Return a reference to a set of time-ordered annihilation operators for a given block
       */
      const c_set_t& get_c_ops_set(int block) const;

      /**
       * Compute determinant. This may suffer from overflow
       */
      Scalar compute_determinant() const;

      /**
       * Compute determinant as a product of Scalar
       */
      std::vector<Scalar> compute_determinant_as_product() const;

      /**
       * Compute inverse matrix. The rows and cols may not be time-ordered.
       */
      eigen_matrix_t compute_inverse_matrix() const;

      /**
       * Compute determinant ratio
       */
      template<typename CdaggIterator, typename CIterator, typename CdaggIterator2, typename CIterator2>
      Scalar try_update(
        CdaggIterator  cdagg_rem_first,  CdaggIterator  cdagg_rem_last,
        CIterator      c_rem_first,      CIterator      c_rem_last,
        CdaggIterator2 cdagg_add_first,  CdaggIterator2 cdagg_add_last,
        CIterator2     c_add_first,      CIterator2     c_add_last
      );

      /**
       * Actually update inverse matrix
       */
      void perform_update();

      /**
       * Reject update
       */
      void reject_update();

      /**
       * Rebuild the matrix from scratch
       */
      ResizableMatrix<Scalar> compute_G_matrix(int block) const;

      /**
       * Rebuild the matrix from scratch
       */
      void rebuild_inverse_matrix();
    };

    using detail::comb_sort;
  }
}
