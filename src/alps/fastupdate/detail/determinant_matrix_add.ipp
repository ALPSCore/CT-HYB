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
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::try_add(
      CdaggCIterator cdagg_c_add_first,
      CdaggCIterator cdagg_c_add_last
    ) {
      check_state(waiting);
      state_ = try_add_called;

      const int nop_add = std::distance(cdagg_c_add_first, cdagg_c_add_last);
      const int nop = inv_matrix_.size1();

      //This should come after add_new_operators
      if (!insertion_possible(cdagg_c_add_first, cdagg_c_add_last)) {
        return 0.0;
      }

      //Add new operators and compute new elements
      perm_rat_ = add_new_operators(cdagg_c_add_first, cdagg_c_add_last);

      //compute the values of new elements
      G_n_n_.resize(nop_add, nop_add);
      G_n_j_.resize(nop_add, nop);
      G_j_n_.resize(nop, nop_add);
      for(int i=0;i<nop;++i) {
        for (int iv=0; iv<nop_add; ++iv) {
          G_n_j_(iv,i) = compute_g(nop+iv, i);
        }
      }
      for(int i=0;i<nop; ++i){
        for (int iv=0; iv<nop_add; ++iv) {
          G_j_n_(i,iv) = compute_g(i, nop+iv);
        }
      }
      for (int iv2=0; iv2<nop_add; ++iv2) {
        for (int iv = 0; iv < nop_add; ++iv) {
          G_n_n_(iv, iv2) = compute_g(nop + iv, nop + iv2);
        }
      }

      return static_cast<double>(perm_rat_)*compute_det_ratio_up(G_j_n_, G_n_j_, G_n_n_, inv_matrix_);
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    void
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::perform_add() {
      check_state(try_add_called);
      state_ = waiting;
      compute_inverse_matrix_up(G_j_n_, G_n_j_, G_n_n_, inv_matrix_);
      permutation_row_col_ *= perm_rat_;
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    void
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::reject_add() {
      check_state(try_add_called);
      state_ = waiting;
      remove_excess_operators();
    }

  }
}
