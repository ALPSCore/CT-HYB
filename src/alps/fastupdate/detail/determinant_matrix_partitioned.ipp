#include "../determinant_matrix_partitioned.hpp"

namespace alps {
  namespace fastupdate {

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    DeterminantMatrixPartitioned<Scalar, GreensFunction, CdaggerOp, COp>::DeterminantMatrixPartitioned(
        boost::shared_ptr<GreensFunction> p_gf
    )
      : Base(p_gf),
        p_gf_(p_gf),
        state_(waiting),
        singular_(false),
        num_flavors_(p_gf->num_flavors()),
        num_sectors_(-1),
        permutation_(1) {

      init(p_gf);
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    template<typename CdaggCIterator>
    DeterminantMatrixPartitioned<Scalar, GreensFunction, CdaggerOp, COp>::DeterminantMatrixPartitioned(
      boost::shared_ptr<GreensFunction> p_gf,
      CdaggCIterator first,
      CdaggCIterator last
    )
      :
        Base(p_gf),
        p_gf_(p_gf),
        state_(waiting),
        singular_(false),
        num_flavors_(p_gf->num_flavors()),
        num_sectors_(-1),
        permutation_(1) {

      init(p_gf);

      std::vector<CdaggerOp> cdagg_ops;
      std::vector<COp> c_ops;
      for (CdaggCIterator it = first; it != last; ++it) {
        cdagg_ops.push_back(it->first);
        c_ops.    push_back(it->second);
      }
      const Scalar det_rat = try_update(
        (CdaggerOp*)NULL,  (CdaggerOp*)NULL,
        (COp*)NULL,        (COp*)NULL,
        cdagg_ops.begin(), cdagg_ops.end(),
        c_ops.begin(),     c_ops.end()
      );
      perform_update();
      sanity_check();
    }

    //Partitioning of flavors
    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    void DeterminantMatrixPartitioned<Scalar, GreensFunction, CdaggerOp, COp>::init(
        boost::shared_ptr<GreensFunction> p_gf
    ) {
      Clustering cl(num_flavors_);
      for (int flavor=0; flavor<num_flavors_; ++flavor) {
        for (int flavor2=0; flavor2<flavor; ++flavor2) {
          assert(p_gf->is_connected(flavor, flavor2)==p_gf->is_connected(flavor2, flavor));
          if (p_gf->is_connected(flavor, flavor2)) {
            cl.connect_vertices(flavor, flavor2);
          }
        }
      }
      cl.finalize_labeling();
      num_sectors_ = cl.get_num_clusters();
      sector_members_ = cl.get_cluster_members();
      sector_belonging_to_.resize(num_flavors_);
      for (int flavor=0; flavor<num_flavors_; ++flavor) {
        sector_belonging_to_[flavor] = cl.get_cluster_label(flavor);
      }

      //Sanity check
      for (int flavor=0; flavor<num_flavors_; ++flavor) {
        for (int flavor2 = 0; flavor2 < flavor; ++flavor2) {
          if (sector_belonging_to_[flavor] != sector_belonging_to_[flavor2]) {
            assert(!p_gf->is_connected(flavor, flavor2));
          }
        }
      }

      //Prepare DeterminantMatrix for each sector
      for (int sector=0; sector<num_sectors_; ++sector) {
        det_mat_.push_back(
          BlockMatrixType(p_gf)
        );
      }

      cdagg_ops_add_.resize(num_flavors_);
      cdagg_ops_rem_.resize(num_flavors_);
      c_ops_add_.resize(num_flavors_);
      c_ops_rem_.resize(num_flavors_);
      cdagg_times_sectored_set_.resize(num_sectors_);
      c_times_sectored_set_.resize(num_sectors_);
    }

    namespace detail {
      //note: set.lower_bound() points the element we're going to erase.
      template<typename T>
      int erase_and_compute_perm_sign_change(std::set<T>& set, std::vector<std::set<T> >& sectored_set, const T& t, int target_sector) {
        int num_ops = std::distance(set.lower_bound(t), set.end());
        for (int sector = target_sector + 1; sector < sectored_set.size(); ++sector) {
          num_ops += sectored_set[sector].size();
        }
        num_ops += std::distance(sectored_set[target_sector].lower_bound(t), sectored_set[target_sector].end());
        set.erase(t);
        sectored_set[target_sector].erase(t);
        return num_ops%2 == 0 ? 1 : -1;
      }

      template<typename T>
      int insert_and_compute_perm_sign_change(std::set<T>& set, std::vector<std::set<T> >& sectored_set, const T& t, int target_sector) {
        int num_ops = std::distance(set.lower_bound(t), set.end());
        for (int sector = target_sector + 1; sector < sectored_set.size(); ++sector) {
          num_ops += sectored_set[sector].size();
        }
        num_ops += std::distance(sectored_set[target_sector].lower_bound(t), sectored_set[target_sector].end() );
        set.insert(t);
        sectored_set[target_sector].insert(t);
        return num_ops%2 == 0 ? 1 : -1;
      }
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    template<typename CdaggIterator, typename CIterator, typename CdaggIterator2, typename CIterator2>
    Scalar
    DeterminantMatrixPartitioned<Scalar,GreensFunction,CdaggerOp,COp>::try_update(
      CdaggIterator  cdagg_rem_first,  CdaggIterator  cdagg_rem_last,
      CIterator      c_rem_first,      CIterator      c_rem_last,
      CdaggIterator2 cdagg_add_first,  CdaggIterator2 cdagg_add_last,
      CIterator2     c_add_first,      CIterator2     c_add_last
    ) {
      int perm_sign_change = 1;

      //Creation operators to be removed
      for (CdaggIterator it=cdagg_rem_first; it!=cdagg_rem_last; ++it) {
        const int sector = sector_belonging_to_[operator_flavor(*it)];
        cdagg_ops_rem_[sector].push_back(*it);
        perm_sign_change *= detail::erase_and_compute_perm_sign_change(cdagg_times_set_, cdagg_times_sectored_set_, *it, sector);
      }

      //Annihilation operators to be removed
      for (CIterator it=c_rem_first; it!=c_rem_last; ++it) {
        const int sector = sector_belonging_to_[operator_flavor(*it)];
        c_ops_rem_[sector].push_back(*it);
        perm_sign_change *= detail::erase_and_compute_perm_sign_change(c_times_set_, c_times_sectored_set_, *it, sector);
      }

      //Creation operators to be added
      for (CdaggIterator2 it=cdagg_add_first; it!=cdagg_add_last; ++it) {
        const int sector = sector_belonging_to_[operator_flavor(*it)];
        cdagg_ops_add_[sector].push_back(*it);
        perm_sign_change *= detail::insert_and_compute_perm_sign_change(cdagg_times_set_, cdagg_times_sectored_set_, *it, sector);
      }

      //Annihilation operators to be added
      for (CIterator2 it=c_add_first; it!=c_add_last; ++it) {
        const int sector = sector_belonging_to_[operator_flavor(*it)];
        c_ops_add_[sector].push_back(*it);
        perm_sign_change *= detail::insert_and_compute_perm_sign_change(c_times_set_, c_times_sectored_set_, *it, sector);
      }

      new_perm_ = perm_sign_change * permutation_;

      //Second, compute determinant ratio from each sector
      Scalar det_rat = 1.0;
      for (int sector=0; sector<num_sectors_; ++sector) {
        det_rat *= det_mat_[sector].try_update(
          cdagg_ops_rem_[sector].begin(), cdagg_ops_rem_[sector].end(),
          c_ops_rem_[sector].    begin(),     c_ops_rem_[sector].end(),
          cdagg_ops_add_[sector].begin(), cdagg_ops_add_[sector].end(),
          c_ops_add_[sector].    begin(),     c_ops_add_[sector].end()
        );
      }

      return det_rat*(1.*perm_sign_change);
    }

    template<
            typename Scalar,
            typename GreensFunction,
            typename CdaggerOp,
            typename COp
    >
    void
    DeterminantMatrixPartitioned<Scalar,GreensFunction,CdaggerOp,COp>::perform_update() {
      for (int sector=0; sector<num_sectors_; ++sector) {
        det_mat_[sector].perform_update();
      }

      reconstruct_operator_list_in_actual_order();

      permutation_ = new_perm_;

      clear_work();

      sanity_check();
    };

    template<
              typename Scalar,
              typename GreensFunction,
              typename CdaggerOp,
              typename COp
      >
    void
    DeterminantMatrixPartitioned<Scalar,GreensFunction,CdaggerOp,COp>::reject_update() {
      for (int sector=0; sector<num_sectors_; ++sector) {
        det_mat_[sector].reject_update();

        //revert the changes in the time ordered sets
        for (int iop = 0; iop < cdagg_ops_add_[sector].size(); ++iop) {
          cdagg_times_set_.erase(cdagg_ops_add_[sector][iop]);
          cdagg_times_sectored_set_[sector].erase(cdagg_ops_add_[sector][iop]);
        }
        for (int iop = 0; iop < c_ops_add_[sector].size(); ++iop) {
          c_times_set_.erase(c_ops_add_[sector][iop]);
          c_times_sectored_set_[sector].erase(c_ops_add_[sector][iop]);
        }
        for (int iop = 0; iop < cdagg_ops_rem_[sector].size(); ++iop) {
          cdagg_times_set_.insert(cdagg_ops_rem_[sector][iop]);
          cdagg_times_sectored_set_[sector].insert(cdagg_ops_rem_[sector][iop]);
        }
        for (int iop = 0; iop < c_ops_rem_[sector].size(); ++iop) {
          c_times_set_.insert(c_ops_rem_[sector][iop]);
          c_times_sectored_set_[sector].insert(c_ops_rem_[sector][iop]);
        }

      }
      reconstruct_operator_list_in_actual_order();//Operators may be swapped even if an update is rejected.

      clear_work();
    };

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    void
    DeterminantMatrixPartitioned<Scalar,GreensFunction,CdaggerOp,COp>::sanity_check() {
#ifndef NDEBUG
      if (singular_) {
        return;
      }

      if (state_ == waiting) {
        std::set<CdaggerOp> cdagg_work;
        std::set<COp> c_work;
        for (int sector=0; sector<num_sectors_; ++sector) {
          cdagg_work.insert(cdagg_times_sectored_set_[sector].begin(), cdagg_times_sectored_set_[sector].end());
          c_work.insert(c_times_sectored_set_[sector].begin(), c_times_sectored_set_[sector].end());
        }
        assert(cdagg_work == cdagg_times_set_);
        assert(c_work == c_times_set_);
        assert(cdagg_work == std::set<CdaggerOp>(cdagg_ops_actual_order_.begin(), cdagg_ops_actual_order_.end()));
        assert(c_work == std::set<COp>(c_ops_actual_order_.begin(), c_ops_actual_order_.end()));
      }

      for (int sector=0; sector<num_sectors_; ++sector) {
        for (typename std::set<CdaggerOp>::iterator it = cdagg_times_sectored_set_[sector].begin(); it != cdagg_times_sectored_set_[sector].end(); ++it) {
          assert(block_belonging_to(it->flavor()) == sector);
        }
        for (typename std::set<COp>::iterator it = c_times_sectored_set_[sector].begin(); it != c_times_sectored_set_[sector].end(); ++it) {
          assert(block_belonging_to(it->flavor()) == sector);
        }
      }

      int pert_order = 0;
      for (int sector=0; sector<num_sectors_; ++sector) {
        pert_order += det_mat_[sector].size();
      }
      assert(size()==pert_order);

      std::vector<std::pair<int,CdaggerOp> > cdagg_ops_work;
      std::vector<std::pair<int,COp> > c_ops_work;
      for (typename std::set<CdaggerOp>::iterator it = cdagg_times_set_.begin(); it != cdagg_times_set_.end(); ++it) {
        cdagg_ops_work.push_back(std::make_pair(block_belonging_to(it->flavor()), *it));
      }
      for (typename std::set<COp>::iterator it = c_times_set_.begin(); it != c_times_set_.end(); ++it) {
        c_ops_work.push_back(std::make_pair(block_belonging_to(it->flavor()), *it));
      }

      detail::comb_sort(cdagg_ops_work.begin(), cdagg_ops_work.end(), CompareOverSectors<CdaggerOp>());
      detail::comb_sort(c_ops_work.begin(),     c_ops_work.end(),     CompareOverSectors<COp>());
      const int perm_recomputed =
        detail::comb_sort(cdagg_ops_work.begin(), cdagg_ops_work.end(), CompareWithinSectors<CdaggerOp>())*
        detail::comb_sort(c_ops_work.begin(),     c_ops_work.end(),     CompareWithinSectors<COp>());
      assert(permutation_ == perm_recomputed);

      //check list of operators in actual order
      if (state_ == waiting) {
        int iop = 0;
        for (int sector=0; sector<num_sectors_; ++sector) {
          int sector_size = det_mat_[sector].size();
          for (int i=0; i<sector_size; ++i) {
            assert(get_cdagg_ops()[iop]==det_mat_[sector].get_cdagg_ops()[i]);
            assert(get_c_ops()[iop]==det_mat_[sector].get_c_ops()[i]);
            ++iop;
          }
        }
      }
#endif
    }
  }
}
