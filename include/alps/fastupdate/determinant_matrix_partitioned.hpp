/**
 * Copyright (C) 2016 by Hiroshi Shinaoka <h.shinaoka@gmail.com>
 */
#pragma once

#include <boost/scoped_ptr.hpp>

#include <Eigen/Dense>

#include "determinant_matrix.hpp"
#include "fastupdate_formula.hpp"
#include "./detail/util.hpp"
#include "./detail/clustering.hpp"

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
    class DeterminantMatrixPartitioned
            : public DeterminantMatrixBase<
                    Scalar,
                    GreensFunction,
                    CdaggerOp,
                    COp,
                    DeterminantMatrixPartitioned<Scalar, GreensFunction, CdaggerOp, COp>
            >
    {
    private:
      typedef std::vector<CdaggerOp> cdagg_container_t;
      typedef std::vector<COp> c_container_t;
      typedef Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> eigen_matrix_t;
      typedef DeterminantMatrixBase<
              Scalar,
              GreensFunction,
              CdaggerOp,
              COp,
              DeterminantMatrixPartitioned<Scalar, GreensFunction, CdaggerOp, COp>
            > Base;
      typedef typename Base::cdagg_set_t cdagg_set_t;
      typedef typename Base::c_set_t c_set_t;
      typedef typename CdaggerOp::itime_type iTimeType;

      enum State {
        waiting = 0,
        try_add_called = 1,
        try_rem_called = 2,
        try_rem_add_called = 3,
        try_replace_cdagg_called = 4,
        try_replace_c_called = 5
      };

      //for time-ordering in each sector
      template<typename T>
      struct CompareWithinSectors {
        bool operator()(const std::pair<int,T>& t1, const std::pair<int,T>& t2) const{
          if (t1.first == t2.first) {
            return operator_time(t1.second) < operator_time(t2.second);
          } else {
            return t1.first < t2.first;
          }
        }
      };

      //for time-ordering over sectors
      template<typename T>
      struct CompareOverSectors {
        bool operator()(const std::pair<int,T>& t1, const std::pair<int,T>& t2) const {
          return operator_time(t1.second) < operator_time(t2.second);
        }
      };

    public:
      DeterminantMatrixPartitioned (
        boost::shared_ptr<GreensFunction> p_gf
      );

      template<typename CdaggCIterator>
      DeterminantMatrixPartitioned (
        boost::shared_ptr<GreensFunction> p_gf,
        CdaggCIterator first,
        CdaggCIterator last
      );

      boost::shared_ptr<GreensFunction>
      get_greens_function() const {
        return p_gf_;
      }

      /** see determinant_matrix_base.hpp */
      inline int size() const {return cdagg_times_set_.size();};

      /** see determinant_matrix_base.hpp */
      inline bool is_singular() const {return singular_;}

      /** see determinant_matrix_base.hpp */
      inline int block_matrix_size(int block) const {
        assert(block>=0 && block<num_blocks());
        if (singular_) {
          throw std::runtime_error("Matrix is singular!");
        }
        return det_mat_[block].size();
      };

      /** see determinant_matrix_base.hpp */
      inline int num_blocks() const {
        return num_sectors_;
      }

      /** see determinant_matrix_base.hpp */
      int num_flavors(int block) const {
        assert(block>=0 && block<num_blocks());
        return sector_members_[block].size();
      }

      /** see determinant_matrix_base.hpp */
      const std::vector<int>& flavors(int block) const {
        return sector_members_[block];
      }

      /** see determinant_matrix_base.hpp */
      int block_belonging_to(int flavor) const {
        return sector_belonging_to_[flavor];
      }

      /** see determinant_matrix_base.hpp */
      inline const cdagg_container_t& get_cdagg_ops() const {
        assert(state_==waiting);
        return cdagg_ops_actual_order_;
      }

      /** see determinant_matrix_base.hpp */
      inline const c_container_t& get_c_ops() const {
        assert(state_==waiting);
        return c_ops_actual_order_;
      }

      /** see determinant_matrix_base.hpp */
      inline const cdagg_container_t& get_cdagg_ops(int block) const {
        assert(state_==waiting);
        return det_mat_[block].get_cdagg_ops();
      }

      /** see determinant_matrix_base.hpp */
      inline const c_container_t& get_c_ops(int block) const {
        assert(state_==waiting);
        return det_mat_[block].get_c_ops();
      }

      /** see determinant_matrix_base.hpp */
      const cdagg_set_t& get_cdagg_ops_set() const {
        throw std::runtime_error("Not implemented!");
      }

      /** see determinant_matrix_base.hpp */
      const c_set_t& get_c_ops_set() const {
        throw std::runtime_error("Not implemented!");
      }

      /** see determinant_matrix_base.hpp */
      const cdagg_set_t& get_cdagg_ops_set(int block) const {
        assert(block>=0 && block<num_blocks());
        return det_mat_[block].get_cdagg_ops_set(0);
      }

      /** see determinant_matrix_base.hpp */
      const c_set_t& get_c_ops_set(int block) const {
        assert(block>=0 && block<num_blocks());
        return det_mat_[block].get_c_ops_set(0);
      }

      /**
       * Compute determinant. This may suffer from overflow
       */
      inline Scalar compute_determinant() const {
        if (singular_) {
          return 0.0;
        }
        Scalar r = 1.0;
        for (int sector=0; sector<num_sectors_; ++sector) {
          r *= det_mat_[sector].compute_determinant();
        }
        return (1.*permutation_)*r;
      }

      /**
       * Compute determinant as a product of Scalar
       */
      std::vector<Scalar> compute_determinant_as_product() const {
        std::vector<Scalar> r;
        if (singular_) {
          return std::vector<Scalar>(1, (Scalar)0.0);
        }
        for (int sector=0; sector<num_sectors_; ++sector) {
          const std::vector<Scalar>& vec = det_mat_[sector].compute_determinant_as_product();
          std::copy(vec.begin(), vec.end(), std::back_inserter(r));
        }
        if (r.size() > 0) {
          r[0] *= permutation_;
        }
        std::sort(r.begin(), r.end(), detail::lesser_by_abs<Scalar>);
        return r;
      }

      /**
       * Compute inverse matrix. The rows and cols may not be time-ordered.
       */
      eigen_matrix_t compute_inverse_matrix() const {
        if (singular_) {
          throw std::runtime_error("Inverse matrix is not available because the matrix is singular!");
        }
        eigen_matrix_t inv(size(), size());
        inv.setZero();
        int offset = 0;
        for (int sector=0; sector<num_sectors_; ++sector) {
          int block_size = det_mat_[sector].size();
          inv.block(offset, offset, block_size, block_size) = det_mat_[sector].compute_inverse_matrix();
          offset += block_size;
        }
        return inv;
      }

      /**
       * Compute inverse matrix for a given block. The rows and cols may not be time-ordered.
       */
      eigen_matrix_t compute_inverse_matrix(int block) const {
        assert(block>=0 && block<num_blocks());
        if (singular_) {
          throw std::runtime_error("Inverse matrix is not available because the matrix is singular!");
        }
        return det_mat_[block].compute_inverse_matrix();
      }

      /**
       * Compute the determinant ratio without actual update
       */
      template<typename CdaggIterator, typename CIterator, typename CdaggIterator2, typename CIterator2>
      Scalar try_update(
        CdaggIterator  cdagg_rem_first,  CdaggIterator  cdagg_rem_last,
        CIterator      c_rem_first,      CIterator      c_rem_last,
        CdaggIterator2 cdagg_add_first,  CdaggIterator2 cdagg_add_last,
        CIterator2     c_add_first,      CIterator2     c_add_last
      );

      /**
       * Perform the update
       */
      void perform_update();
      
      /**
       * Cancel the update
       */
      void reject_update();

      /**
       * Rebuild the matrix from scratch
       */
      Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>
      compute_G_matrix(int block) const {
        return det_mat_[block].compute_G_matrix();
      };

      /**
       * Rebuild the matrix from scratch
       */
      void rebuild_inverse_matrix() {
        for (int sector=0; sector<num_sectors_; ++sector) {
          det_mat_[sector].rebuild_inverse_matrix();
        }
      }

    private:
      typedef DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp> BlockMatrixType;

      /**
       * Try to remove some operators and add new operators
       * This function actually remove and insert operators in cdagg_ops_, c_ops_ but does not update the matrix.
       * After calling try_add(), either of perform_add() or reject_add() must be called.
       */
      template<typename CdaggCIterator>
      Scalar try_add(
        CdaggCIterator cdagg_c_add_first,
        CdaggCIterator cdagg_c_add_last
      );

      void perform_add() {perform_update();};

      void reject_add() {reject_update();};

      /**
       * Try to remove some operators and add new operators
       * This function actually remove and insert operators in cdagg_ops_, c_ops_ but does not update the matrix
       */
      template<typename CdaggCIterator>
      Scalar try_remove_add(
        CdaggCIterator cdagg_c_rem_first,
        CdaggCIterator cdagg_c_rem_last,
        CdaggCIterator cdagg_c_add_first,
        CdaggCIterator cdagg_c_add_last
      );

      /**
       *  Remove some operators and add new operators
       *  This function actually update the matrix
       */
      void perform_remove_add() {perform_update();};

      template<typename CdaggCIterator>
      void reject_remove_add(
        CdaggCIterator cdagg_c_rem_first,
        CdaggCIterator cdagg_c_rem_last,
        CdaggCIterator cdagg_c_add_first,
        CdaggCIterator cdagg_c_add_last
      ) {reject_update();};

      /**
       * Try to remove some operators
       * This function actually remove and insert operators in cdagg_ops_, c_ops_ but does not update the matrix
       */
      template<typename CdaggCIterator>
      Scalar try_remove(
        CdaggCIterator cdagg_c_rem_first,
        CdaggCIterator cdagg_c_rem_last
      );

      void perform_remove() {perform_update();};

      void reject_remove() {reject_update();};

      /**
       * Try to replace a creation operator
       */
      Scalar try_replace_cdagg(
              const CdaggerOp & old_cdagg,
              const CdaggerOp & new_cdagg
      );

      void perform_replace_cdagg() {perform_update();};

      void reject_replace_cdagg() {reject_update();};

      /**
       * Try to replace an annihilation operator
       */
      Scalar try_replace_c(
        const COp & old_c,
        const COp & new_c
      );

      void perform_replace_c() {perform_update();};

      void reject_replace_c() {reject_update();};


      State state_;
      bool singular_;

      int num_flavors_;                     //number of flavors
      int num_sectors_;                     //number of sectors
      std::vector<std::vector<int> > sector_members_;     //members of each sector
      std::vector<int> sector_belonging_to_; //remember to which sector each flavor belongs

      boost::shared_ptr<GreensFunction> p_gf_;
      std::vector<BlockMatrixType>  det_mat_;

      //permutation from a set that is time-ordered in each sector to a time-ordered set
      int permutation_;//1 or -1

      //a vector of creation and annihilation operators time-ordered in each sector
      //first element: sector
      //second element: operator
      //std::vector<std::pair<int,CdaggerOp> > cdagg_ops_ordered_in_sectors_;
      //std::vector<std::pair<int,COp> > c_ops_ordered_in_sectors_;

      //Creation and annihilation operators in the same as appearing in the block matrices (not always time-ordered at all)
      //They are expected to be well-defined only when state is "waiting".
      mutable std::vector<CdaggerOp> cdagg_ops_actual_order_;
      mutable std::vector<COp> c_ops_actual_order_;

      //time-ordered set
      std::set<CdaggerOp> cdagg_times_set_;
      std::set<COp> c_times_set_;
      std::vector<std::set<CdaggerOp> > cdagg_times_sectored_set_;
      std::vector<std::set<COp> > c_times_sectored_set_;

      //for update
      int new_perm_;
      //std::vector<std::pair<int,CdaggerOp> > cdagg_ops_work_;
      //std::vector<std::pair<int,COp> > c_ops_work_;
      std::vector<std::vector<CdaggerOp> > cdagg_ops_add_, cdagg_ops_rem_;
      std::vector<std::vector<COp> > c_ops_add_, c_ops_rem_;

      void clear_work() {
        for (int sector=0; sector < num_sectors_; ++sector) {
          cdagg_ops_add_[sector].resize(0);
          cdagg_ops_rem_[sector].resize(0);
          c_ops_add_[sector].resize(0);
          c_ops_rem_[sector].resize(0);
        }
      }

      void init(boost::shared_ptr<GreensFunction> p_gf);

      inline void check_state(State state) const {
        if (state_ != state) {
          throw std::logic_error("Error: the system is not in a correct state!");
        }
      }

      void reconstruct_operator_list_in_actual_order() {
        cdagg_ops_actual_order_.resize(0);
        c_ops_actual_order_.resize(0);
        for (int sector=0; sector<num_sectors_; ++sector) {
          const std::vector<CdaggerOp>& cdagg_ops_tmp =
            det_mat_[sector].get_cdagg_ops();
          std::copy(cdagg_ops_tmp.begin(), cdagg_ops_tmp.end(), std::back_inserter(cdagg_ops_actual_order_));

          const std::vector<COp>& c_ops_tmp =
            det_mat_[sector].get_c_ops();
          std::copy(c_ops_tmp.begin(), c_ops_tmp.end(), std::back_inserter(c_ops_actual_order_));
        }

        assert(cdagg_ops_actual_order_.size()==size());
        assert(c_ops_actual_order_.size()==size());
      };

      void sanity_check();

    };
  }
}

#include "./detail/determinant_matrix_partitioned.ipp"
