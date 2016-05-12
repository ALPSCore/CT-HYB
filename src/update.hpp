#ifndef ___UPDATE___
#define ___UPDATE___

template<class SCALAR>
inline SCALAR dsign(SCALAR s) {
    double abs_s = std::abs(s);
    if (abs_s==0.0) {
        throw std::runtime_error("dsign: s must not be zero");
    } else {
        return s/abs_s;
    }
}

template<class MAT>
void move_row(MAT& M, int old_row, int new_row) {
    assert(0<=old_row&&old_row<M.size1());
    assert(0<=new_row&&new_row<M.size1());
    if (old_row==new_row) {
        return;
    } else if (old_row<new_row) {
        for (int i=old_row; i<new_row; ++i) {
            for (int m=0; m<M.size1(); ++m) {
                std::swap(M(i,m),M(i+1,m));
            }
        }
    } else if (old_row>new_row) {
        for (int i=old_row-1; i>=new_row; --i) {
            for (int m=0; m<M.size1(); ++m) {
                std::swap(M(i,m),M(i+1,m));
            }
        }
    }
}

template<class MAT>
void move_column(MAT& M, int old_column, int new_column) {
    assert(0<=old_column&&old_column<M.size1());
    assert(0<=new_column&&new_column<M.size1());
    if (old_column==new_column) {
        return;
    } else if (old_column<new_column) {
        for (int i=old_column; i<new_column; ++i) {
            for (int m=0; m<M.size1(); ++m) {
                std::swap(M(m,i),M(m,i+1));
            }
        }
    } else if (old_column>new_column) {
        for (int i=old_column-1; i>=new_column; --i) {
            for (int m=0; m<M.size1(); ++m) {
                std::swap(M(m,i),M(m,i+1));
            }
        }
    }
}


//interpolate F[orbital] linearly between two points
template<class HYB>
typename HYB::value_type
interpolate_F(double t, double BETA, const HYB& F) {

  double sign=1;
  if (t<0) {
    t += BETA;
    sign=-1;
  }

  int N = F.size()-1;
  double n = t/BETA*N;
  int n_lower =(int) n; 
  return sign*(F[n_lower] + (n-n_lower)*(F[n_lower+1]-F[n_lower])); // interpolate linearly between n_lower and n_lower+1
}

//rebuild matrix from scratch
template<class MAT, class HYB>
void construct_blas_matrix(MAT & M, const operator_container_t &creation_operators,
  const operator_container_t &annihilation_operators, double BETA, const HYB & F)
{

  int N = creation_operators.size();
  M.resize(N, N);
  int row = -1;
  int column = -1;
  for (operator_container_t::iterator ita = annihilation_operators.begin(); ita != annihilation_operators.end(); ita++) {
    row++;
    for (operator_container_t::iterator itc = creation_operators.begin(); itc != creation_operators.end(); itc++) {
      column++;

      double argument = ita->time() - itc->time();
      double sign = 1;
      if (argument < 0) {
        argument += BETA;
        sign = -1;
      }
      M(row, column) = interpolate_F(argument, BETA, F[ita->flavor()][itc->flavor()]) * sign;
    }
    column = -1;
  }
}

inline
double make_set_impl(const operator_container_t &annihilation_operators, double BETA, double dtau, std::set<boost::tuple<double,int,double> >& annset) {
  int row = -1;
  annset.clear();
  int num_op_shifted = 0;
  for (operator_container_t::iterator ita = annihilation_operators.begin(); ita != annihilation_operators.end(); ita++) {
    row++;
    double p = 1.0;
    double t = ita->time() + dtau;
    if (t > BETA) {
      t -= BETA;
      p = -1.0;
      ++num_op_shifted;
    }
    annset.insert(boost::make_tuple(t,row,p));
  }
  return ((annihilation_operators.size()-num_op_shifted)*num_op_shifted)%2==0 ? 1 : -1;
}

//shift all operaters in imaginary time
template<class MAT>
double update_inverse_matrix_global_shift(const MAT & M, MAT & M_new, const operator_container_t &creation_operators,
                           const operator_container_t &annihilation_operators, double BETA, double dtau)
{
  typedef boost::tuple<double,int,double> key_t;
  typedef std::set<key_t> map_t;

  map_t annset, crset;
  double det_rat = 1.0;
  det_rat *= make_set_impl(annihilation_operators, BETA, dtau, annset);
  det_rat *= make_set_impl(creation_operators, BETA, dtau, crset);

  M_new.destructive_resize(M.size1(), M.size2());
  int row = -1;
  int column = -1;
  for (map_t::iterator ita = annset.begin(); ita != annset.end(); ita++) {
    row++;
    for (map_t::iterator itc = crset.begin(); itc != crset.end(); itc++) {
      column++;

      M_new(row, column) = M(boost::get<1>(*ita), boost::get<1>(*itc))*
        boost::get<2>(*ita)*boost::get<2>(*itc);
    }
    column = -1;
  }

  return det_rat;
}



//////////////////////////////////////////////////////////////////////////////////////////
// functions required to compute determinant ratios and perform fast matrix updates 
//////////////////////////////////////////////////////////////////////////////////////////

template<class SCALAR, class MAT, class HYB>
SCALAR det_rat_shift_start(double new_t_start, int k, int flavor_ins,
 MAT & M, operator_container_t & annihilation_operators, const HYB & F, double BETA) {

  SCALAR det_rat = 0;
  operator_container_t::iterator ita = annihilation_operators.begin();
  for (int i = 0; i < M.size1(); i++) {
    det_rat += interpolate_F(ita->time() - new_t_start, BETA, F[ita->flavor()][flavor_ins]) * M(k, i);//right
    ita++;
  }

  return det_rat;
}

template<class SCALAR, class MAT, class HYB>
int compute_M_shift_start(double new_t_start, int k, int new_position, int flavor_ins,
        MAT & M, const operator_container_t & annihilation_operators, const HYB & F, double BETA, SCALAR det_rat) {

  std::vector<SCALAR> R(M.size1(), 0), M_k(M.size1(), 0), Fs(M.size1(), 0);

  operator_container_t::const_iterator ita = annihilation_operators.begin();
  for (int i = 0; i < (int) M_k.size(); i++) {
    M_k[i] = M(k, i);
    Fs[i] = interpolate_F(ita->time() - new_t_start, BETA, F[ita->flavor()][flavor_ins]);
    ita++;
  }

  for (int i = 0; i < (int) R.size(); i++) {
    if (i != k) {
      for (int j = 0; j < (int) R.size(); j++)
        R[i] += M(i, j) * Fs[j];
    }
  }

  for (int n = 0; n < (int) M.size1(); n++) {
    if (n != k) {
      for (int m = 0; m < (int) M.size1(); m++) {
        M(n, m) -= M_k[m] * R[n] / det_rat;
      }
    } else {
      for (int m = 0; m < (int) M.size1(); m++) {
        M(n, m) = M_k[m] / det_rat;
      }
    }
  }

  //swap rows
  move_row(M, k, new_position);

  return std::abs(k-new_position);
}  

// shift end point of segment
template<class SCALAR, class MAT, class HYB>
SCALAR det_rat_shift_end(double new_t_end, int k, int flavor_rem, MAT & M,
 const operator_container_t & creation_operators, const HYB & F, double BETA) {

  SCALAR det_rat = 0;
  operator_container_t::const_iterator itc = creation_operators.begin();
  for (int i = 0; i < M.size1(); i++) {
    det_rat += interpolate_F(new_t_end - itc->time(), BETA, F[flavor_rem][itc->flavor()]) * M(i, k);
    itc++;
  }
  return det_rat;
}

template<class SCALAR, class MAT, class HYB>
int compute_M_shift_end(double new_t_end, int k, int new_position, int flavor_rem, MAT & M,
 const operator_container_t & creation_operators, const HYB & F,
 double BETA, SCALAR det_rat) {

  std::vector<SCALAR> R(M.size1(), 0), M_k(M.size1(), 0), Fe(M.size1(), 0);

  operator_container_t::const_iterator itc = creation_operators.begin();
  for (int i = 0; i < (int) M_k.size(); i++) {
    M_k[i] = M(i, k);
    Fe[i] = interpolate_F(new_t_end - itc->time(), BETA, F[flavor_rem][itc->flavor()]);
    itc++;
  }
  for (int i = 0; i < (int) R.size(); i++) {
    if (i != k) {
      for (int j = 0; j < (int) R.size(); j++)
        R[i] += Fe[j] * M(j, i);
    }
  }
  for (int m = 0; m < (int) M.size1(); m++) {
    if (m != k) {
      for (int n = 0; n < (int) M.size1(); n++) {
        M(n, m) -= M_k[n] * R[m] / det_rat;
      }
    } else {
      for (int n = 0; n < (int) M.size1(); n++) {
        M(n, m) = M_k[n] / det_rat;
      }
    }
  }

  //swap column
  move_column(M, k, new_position);

  return std::abs(k-new_position);
}  
  
template<class MAT, class MAT2, class HYB>
typename MAT::type
det_rat_row_column_up(int flavor_ins, int flavor_rem, double t_ins, double t_rem, int row, int column, const MAT & M,
  const operator_container_t& creation_operators, const operator_container_t& annihilation_operators, const HYB& F, MAT2& sign_Fs, MAT2& Fe_M, double BETA) {

  typedef typename MAT::type SCALAR;
  
  const double sign=((row+column)%2==0 ? 1 : -1);
  
  if (annihilation_operators.size()>0) {
    operator_container_t::iterator it_c=creation_operators.begin();
    operator_container_t::iterator it_a=annihilation_operators.begin();
    Eigen::Matrix<SCALAR,Eigen::Dynamic,Eigen::Dynamic> Fe(1,annihilation_operators.size());
    if (annihilation_operators.size()!=creation_operators.size()) {
      throw std::runtime_error("annihilation_operators.size()!=creation_operators.size() in det_rat_row_column_up");
    }

    for (int i=0; i<(int)creation_operators.size(); i++) {
      Fe(0,i) = interpolate_F(t_rem-it_c->time(), BETA, F[flavor_rem][it_c->flavor()]);
      sign_Fs(i,0) = sign*interpolate_F(it_a->time()-t_ins, BETA, F[(int)(it_a->flavor())][flavor_ins]);
      it_c++;
      it_a++;
    }
    
    Fe_M = Fe*M.block();
  
    return sign*interpolate_F(t_rem-t_ins, BETA, F[flavor_rem][flavor_ins])-(Fe_M*sign_Fs)(0,0);
  } else {
    return sign*interpolate_F(t_rem-t_ins, BETA, F[flavor_rem][flavor_ins]);
  }
}

template<class MAT, class MAT2>
void compute_M_row_column_up(int row, int column, MAT & M, MAT2& sign_Fs, MAT2& Fe_M, typename MAT::type det_rat) {

  typedef typename MAT::type SCALAR;

  SCALAR det_rat_inv=1./det_rat;
  double sign=((row+column)%2==0 ? 1 : -1);
  Eigen::Matrix<SCALAR,Eigen::Dynamic,Eigen::Dynamic> M_sign_Fs(sign_Fs.size(),1);
  M_sign_Fs = M.block()*sign_Fs;

  assert(M.size1()==M.size2());
  MAT M_new(M.size1()+1,M.size2()+1);
  int i_new;
  int j_new;
    
  // element (j,i)
  M_new(column,row) = sign*det_rat_inv;
  
  // row j and column i
  for (int k=0; k<M.size1(); k++) {
    i_new = (k<column ? k : k+1);
    j_new = (k<row ? k : k+1);	
	  M_new(i_new,row) = -M_sign_Fs(k)*det_rat_inv;
	  M_new(column,j_new) = -sign*Fe_M(k)*det_rat_inv;
  }
  
  // remaining elements
  for (int k=0; k<M.size1(); k++) {
    i_new = (k<column ? k : k+1);
    for (int l=0; l<M.size1(); l++) {
      j_new = (l<row ? l : l+1);
	    M_new(i_new, j_new) = M(k,l) + M_sign_Fs(k)*Fe_M(l)*det_rat_inv;
    }
  }
  
  M_new.swap(M);
  return;
}  

template<class MAT>
typename MAT::type
det_rat_row_column_down(int position_c, int position_a, MAT & M) {

  typename MAT::type sign=((position_c+position_a)%2==0 ? 1 : -1);
  
  return sign*M(position_c, position_a);
  
}

template<class MAT>
void compute_M_row_column_down(int position_c, int position_a, MAT & M) {
  MAT M_new(M.size1()-1,M.size2()-1);
  int k_old;
  int l_old;
  
  for (int k=0; k<M_new.size1(); k++) {
	k_old = (k<position_c ? k : k+1);
    for (int l=0; l<M_new.size2(); l++) {
	  l_old = (l<position_a ? l : l+1);	
	  M_new(k,l) = M(k_old, l_old)-M(k_old,position_a)*M(position_c,l_old)/M(position_c,position_a);
    }
  }
  M_new.swap(M);
}

template <class O, class G, class MAT> 
typename MAT::type
cal_det(const O& creation_operators, const O& annihilation_operators, MAT& M, double BETA, const G& F) {
    typedef typename MAT::type SCALAR;

    const int size1 = creation_operators.size();

    if (size1==0) return 1.0;

    assert(creation_operators.size()==annihilation_operators.size());

    Eigen::Matrix<SCALAR,Eigen::Dynamic,Eigen::Dynamic> matrix(size1,size1);
    construct_blas_matrix(matrix, creation_operators, annihilation_operators, BETA, F);
    SCALAR det = matrix.determinant();

    M.destructive_resize(size1,size1);
    if (size1>0) {
      M.block() = matrix;
      M.invert();
    }
    return det;

    /*
     * SCALAR dummy;
    dense_matrix_t M_dense(size1,size1);
    for (int i = 0; i < size1; i++) {
        for (int j = 0; j < size1; j++) {
            M_dense(i, j) = matrix(i, j);
        }
    }
    try {
      invert(M_dense, dummy);
    } catch (std::exception& exc) {
      return 0.0;
    }
    */
    /*
    Eigen::Matrix<SCALAR,Eigen::Dynamic,Eigen::Dynamic> M_dens = ;
    for (int i = 0; i < size1; i++) {
        for (int j = 0; j < size1; j++) {
            M(i, j) = M_dense(i, j);
        }
    }
    */
}

template<class O>
int copy_swap_flavors_ops(const O& operators, O& operators_new, int flavor1, int flavor2) {
    assert(flavor1!=flavor2);
    int count = 0;
    operators_new.clear();
    for (typename O::const_iterator it = operators.begin(); it != operators.end(); ++it) {
        typename O::value_type op = *it;//copy
       	if (op.flavor() == flavor1) {
           	op.set_flavor(flavor2);
           	++count;
       	} else if (op.flavor() == flavor2) {
           	op.set_flavor(flavor1);
           	++count;
       	}
       	operators_new.insert(op);
    }
    assert(operators.size()==operators_new.size());
    return count;
}

template<class O>
int global_shift_ops(const O& operators, O& operators_new, double BETA, double shift) {
    assert(shift>=0);
    assert(shift<BETA);
    int count = 0;
    operators_new.clear();
    for (typename O::const_iterator it = operators.begin(); it != operators.end(); ++it) {
        typename O::value_type op = *it;//copy
        double new_time = op.time()+shift;
        if (new_time>BETA) {
            new_time -= BETA;
            ++count;
        }
        op.set_time(new_time);
       	operators_new.insert(op);
    }
    assert(operators.size()==operators_new.size());
    return count;
}

#endif
