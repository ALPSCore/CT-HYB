#pragma once

#include "../operator.hpp"
#include "src/orthogonal_basis/basis.hpp"

#include <boost/timer/timer.hpp>


//Measure G2 by removing hyridization lines
template<typename SCALAR>
Eigen::Tensor<SCALAR,7>
measure_g2(double beta,
           int num_flavors,
           boost::shared_ptr<OrthogonalBasis> p_basis_f,
           boost::shared_ptr<OrthogonalBasis> p_basis_b,
           const std::vector<psi> &creation_ops,
           const std::vector<psi> &annihilation_ops,
           const Eigen::Matrix<SCALAR,Eigen::Dynamic,Eigen::Dynamic> &M
) {
  const double temperature = 1. / beta;
  const int dim_f = p_basis_f->dim();
  const int dim_b = p_basis_b->dim();
  const int num_phys_rows = creation_ops.size();

  assert(M.rows() == creation_ops.size());

  if (creation_ops.size() != annihilation_ops.size() || creation_ops.size() != M.rows()) {
    throw std::runtime_error("Fatal error in measure_g2");
  }
  boost::timer::cpu_timer timer;

  boost::multi_array<double, 3>
      Pl_f(boost::extents[num_phys_rows][num_phys_rows][dim_f]);//annihilator, creator, ir
  boost::multi_array<double, 3>
      Pl_b(boost::extents[num_phys_rows][num_phys_rows][dim_b]);//annihilator, creator, ir
  {
    //Normalization factor of basis functions
    std::vector<double> norm_coeff_f(dim_f), norm_coeff_b(dim_b);
    for (int il = 0; il < dim_f; ++il) {
      norm_coeff_f[il] = sqrt(2.0/p_basis_f->norm2(il));
    }
    for (int il = 0; il < dim_b; ++il) {
      norm_coeff_b[il] = sqrt(2.0/p_basis_b->norm2(il));
    }

    std::vector<double> Pl_tmp(std::max(dim_f, dim_b));
    for (int k = 0; k < num_phys_rows; k++) {
      for (int l = 0; l < num_phys_rows; l++) {
        double argument = annihilation_ops[k].time() - creation_ops[l].time();
        double arg_sign = 1.0;
        if (argument < 0) {
          argument += beta;
          arg_sign = -1.0;
        }
        const double x = 2 * argument * temperature - 1.0;
        p_basis_f->value(x, Pl_tmp);
        for (int il = 0; il < dim_f; ++il) {
          Pl_f[k][l][il] = arg_sign * norm_coeff_f[il] * Pl_tmp[il];
        }

        p_basis_b->value(x, Pl_tmp);
        for (int il = 0; il < dim_b; ++il) {
          Pl_b[k][l][il] = norm_coeff_b[il] * Pl_tmp[il];
        }
      }
    }
  }
  const double time1 = timer.elapsed().wall * 1E-9;

  //The indices of M are reverted from (C. 24) of L. Boehnke (2011) because we're using the F convention here.
  //First, compute relative weights. This costs O(num_phys_rows^4) operations.
  double norm = 0.0;
  for (int a = 0; a < num_phys_rows; ++a) {
    for (int b = 0; b < num_phys_rows; ++b) {
      for (int c = a+1; c < num_phys_rows; ++c) {
        for (int d = b+1; d < num_phys_rows; ++d) {
          /*
           * Delta convention
           * M_ab  M_ad
           * M_cb  M_cd
           */
          norm += std::abs((M(b,a) * M(d,c) - M(b,c) * M(d,a)));
        }
      }
    }
  }
  //The factor 4 is from the degree of freedom of exchange a and c or b and d.
  norm *= 4;
  const double time2 = timer.elapsed().wall * 1E-9;

  Eigen::Tensor<SCALAR,3> tensor1(dim_f, num_flavors, num_phys_rows);//(l1, flavor_b, a)
  tensor1.setZero();
  for (int il1 = 0; il1 < dim_f; ++il1) {
    for (int a = 0; a < num_phys_rows; ++a) {
      for (int b = 0; b < num_phys_rows; ++b) {
        int flavor_b = creation_ops[b].flavor();
        tensor1(il1, flavor_b, a) += M(b,a) * Pl_f[a][b][il1];
      }
    }
  }

  Eigen::Tensor<SCALAR,3> tensor2(dim_f, num_flavors, num_phys_rows);//(f3, d, il2)
  tensor2.setZero();
  for (int il2 = 0; il2 < dim_f; ++il2) {
    for (int d = 0; d < num_phys_rows; ++d) {
      for (int c = 0; c < num_phys_rows; ++c) {
        const int flavor_c = annihilation_ops[c].flavor();
        tensor2(il2, flavor_c, d) += M(d,c) * Pl_f[c][d][il2] * (il2%2 == 0 ? -1.0 : 1.0);
      }
    }
  }

  const double time3 = timer.elapsed().wall * 1E-9;
  //Contraction requires O(num_phys_rows^2 Nl^3 num_flavors^2) operators
  //Cancellation requires O(num_phys_rows^3 Nl^3) operators
  Eigen::Tensor<SCALAR,7> result_H(dim_f, num_flavors, dim_f, num_flavors, dim_b, num_flavors, num_flavors);
  result_H.setZero();
  const SCALAR coeff = 1.0 / (norm * beta);//where is this beta factor from?
  Eigen::Tensor<SCALAR,2> map3(1,dim_b);


  std::vector<std::vector<int>> flavor_creation(num_flavors), flavor_annihilation(num_flavors);
  for (int i = 0; i < num_phys_rows; ++i) {
    flavor_creation[creation_ops[i].flavor()].push_back(i);
    flavor_annihilation[annihilation_ops[i].flavor()].push_back(i);
  }

  const int cache_size = 100;
  Eigen::Tensor<SCALAR,4> right_tensor(dim_f, num_flavors, dim_b, cache_size);
  Eigen::Tensor<SCALAR,3> left_tensor(dim_f, num_flavors, cache_size);

  for (int flavor_d = 0; flavor_d < num_flavors; ++flavor_d) {
    for (int flavor_a = 0; flavor_a < num_flavors; ++flavor_a) {
      int D = flavor_annihilation[flavor_a].size() * flavor_creation[flavor_d].size();

      //Eigen::array<long,4> right_dims {D,dim_f,num_flavors,dim_b};

      //map_H (l1, flavor_b, l2, flavor_c, l3)
      Eigen::TensorMap<Eigen::Tensor<SCALAR,5>> map_H(
          &result_H(0, 0, 0, 0, 0, flavor_a, flavor_d), dim_f, num_flavors, dim_f, num_flavors, dim_b
      );

      int idx = 0;
      int tot_idx = 0;
      for (auto a : flavor_annihilation[flavor_a]) {
        for (auto d : flavor_creation[flavor_d]) {
          //map3(1, il3)
          for (int il3 = 0; il3 < dim_b; ++il3) {
            map3(0, il3) = coeff * Pl_b[a][d][il3];
          }

          //(1,l2,flavor_c) * (1, l3) => (l2, flavor_c, l3)
          Eigen::TensorMap<Eigen::Tensor<SCALAR, 3>> map2(&tensor2(0, 0, d), 1, dim_f, num_flavors);
          Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(0, 0) };
          right_tensor.chip(idx,3) = map2.contract(map3, product_dims);

          left_tensor.chip(idx,2) = tensor1.chip(a,2);

          if (idx == cache_size-1 || tot_idx == D-1) {
            Eigen::array<int, 4> r_offsets = {0, 0, 0, 0};
            Eigen::array<int, 4> r_extents = {dim_f, num_flavors, dim_b, idx+1};
            Eigen::array<int, 3> l_offsets = {0, 0, 0};
            Eigen::array<int, 3> l_extents = {dim_f, num_flavors, idx+1};
            map_H += left_tensor.slice(l_offsets, l_extents).contract(
                right_tensor.slice(r_offsets, r_extents),
                Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(2, 3)}
            );
            idx = 0;
          } else {
            ++idx;
          }
          ++ tot_idx;
        } //d
      } //a
    }//flavor_a
  }//flavor_d

  const double time4 = timer.elapsed().wall * 1E-9;

  //substract contributions from terms for a==c or b==d.
  bool exact_cancellation = true;
  Eigen::Tensor<SCALAR,7> result_H_cancel2(dim_f, dim_f, dim_b, num_flavors, num_flavors, num_flavors, num_flavors);
  result_H_cancel2.setZero();

  if (exact_cancellation) {
    //a==c
    {
      //(l1, flavor_b, l2, l3, flavor_d, flavor_a=flavor_c)
      Eigen::Tensor<SCALAR,6> cancel_ad(dim_f, num_flavors, dim_f, dim_b, num_flavors, num_flavors);
      cancel_ad.setZero();
      for (int a = 0; a < num_phys_rows; ++a) {
        int flavor_a = annihilation_ops[a].flavor();
        //auto c = a;
        //int flavor_c = annihilation_ops[c].flavor();

        Eigen::Tensor<SCALAR, 3> left_tensor(1, dim_f, num_flavors);
        left_tensor.setZero();
        for (int b = 0; b < num_phys_rows; ++b) {
          int flavor_b = creation_ops[b].flavor();
          for (int il1 = 0; il1 < dim_f; ++il1) {
            left_tensor(0, il1, flavor_b) += M(b, a) * Pl_f[a][b][il1];
          }
        }

        Eigen::Tensor<SCALAR, 4> right_tensor(1, dim_f, dim_b, num_flavors);
        right_tensor.setZero();
        for (int d = 0; d < num_phys_rows; ++d) {
          int flavor_d = creation_ops[d].flavor();
          for (int il2 = 0; il2 < dim_f; ++il2) {
            for (int il3 = 0; il3 < dim_b; ++il3) {
              right_tensor(0, il2, il3, flavor_d) += M(d, a) * Pl_f[a][d][il2] * Pl_b[a][d][il3] * (il2 % 2 == 0 ? -1.0 : 1.0);
            }
          }
        }

        //(l1, flavor_b, l2, l3, flavor_d)
        Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(0, 0)};
        Eigen::TensorMap<Eigen::Tensor<SCALAR,5>> map(&cancel_ad(0,0,0,0,0,flavor_a), dim_f, num_flavors, dim_f, dim_b, num_flavors);
        map -= coeff * left_tensor.contract(right_tensor, product_dims);
      }

      for (int flavor_d = 0; flavor_d < num_flavors; ++flavor_d) {
        for (int flavor_b = 0; flavor_b < num_flavors; ++flavor_b) {
          for (int flavor_a = 0; flavor_a < num_flavors; ++flavor_a) {
            for (int il3 = 0; il3 < dim_b; ++il3) {
              for (int il2 = 0; il2 < dim_f; ++il2) {
                for (int il1 = 0; il1 < dim_f; ++il1) {
                  result_H_cancel2(il1, il2, il3, flavor_a, flavor_b, flavor_a, flavor_d)
                      += cancel_ad(il1, flavor_b, il2, il3, flavor_d, flavor_a);
                }
              }
            }
          }
        }
      }
    }

    //b==d
    {
      //(l1, l3, flavor_a, l2, flavor_c, flavor_b=flavor_d)
      Eigen::Tensor<SCALAR,6> cancel_bd(dim_f, dim_b, num_flavors, dim_f, num_flavors, num_flavors);
      cancel_bd.setZero();
      for (int b = 0; b < num_phys_rows; ++b) {
        auto d = b;
        int flavor_b = creation_ops[b].flavor();
        int flavor_d = flavor_b;

        //(1, l1, l3, flavor_a)
        Eigen::Tensor<SCALAR, 4> left_tensor(1, dim_f, dim_b, num_flavors);
        left_tensor.setZero();
        for (int a = 0; a < num_phys_rows; ++a) {
          int flavor_a = annihilation_ops[a].flavor();
          for (int il1 = 0; il1 < dim_f; ++il1) {
            for (int il3 = 0; il3 < dim_b; ++il3) {
              left_tensor(0, il1, il3, flavor_a) += M(b, a) * Pl_f[a][b][il1] * Pl_b[a][d][il3];
            }
          }
        }

        //(1, l2, flavor_c)
        Eigen::Tensor<SCALAR, 3> right_tensor(1, dim_f, num_flavors);
        right_tensor.setZero();
        for (int c = 0; c < num_phys_rows; ++c) {
          int flavor_c = annihilation_ops[c].flavor();
          for (int il2 = 0; il2 < dim_f; ++il2) {
            right_tensor(0, il2, flavor_c) += M(d, c) * Pl_f[c][d][il2] * (il2 % 2 == 0 ? -1.0 : 1.0);
          }
        }

        //(l1, l3, flavor_a, l2, flavor_c)
        Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(0, 0)};
        Eigen::TensorMap<Eigen::Tensor<SCALAR, 5>>
            map(&cancel_bd(0, 0, 0, 0, 0, flavor_b), dim_f, dim_b, num_flavors, dim_f, num_flavors);
        map -= coeff * left_tensor.contract(right_tensor, product_dims);
      }

      for (int flavor_c = 0; flavor_c < num_flavors; ++flavor_c) {
        for (int flavor_b = 0; flavor_b < num_flavors; ++flavor_b) {
          for (int flavor_a = 0; flavor_a < num_flavors; ++flavor_a) {
            for (int il3 = 0; il3 < dim_b; ++il3) {
              for (int il2 = 0; il2 < dim_f; ++il2) {
                for (int il1 = 0; il1 < dim_f; ++il1) {
                  result_H_cancel2(il1, il2, il3, flavor_a, flavor_b, flavor_c, flavor_b)
                      += cancel_bd(il1, il3, flavor_a, il2, flavor_c, flavor_b);
                }
              }
            }
          }
        }
      }
    }

    //a==c && b==d (double counting)
    {
      //(l1, l2, l3, flavor_a=flavor_c, flavor_b=flavor_d)
      Eigen::Tensor<SCALAR,5> cancel_abcd(dim_f, dim_f, dim_b, num_flavors, num_flavors);
      cancel_abcd.setZero();
      for (int a = 0; a < num_phys_rows; ++a) {
        int flavor_a = annihilation_ops[a].flavor();
        for (int b = 0; b < num_phys_rows; ++b) {
          int flavor_b = creation_ops[b].flavor();

          for (int il3 = 0; il3 < dim_b; ++il3) {
            for (int il2 = 0; il2 < dim_f; ++il2) {
              for (int il1 = 0; il1 < dim_f; ++il1) {
                cancel_abcd(il1, il2, il3, flavor_a, flavor_b) +=
                    M(b, a) * M(b, a) * Pl_f[a][b][il1] * Pl_f[a][b][il2] * Pl_b[a][b][il3]
                        * (il2 % 2 == 0 ? -1.0 : 1.0);
              }
            }
          }
        }
      }

      for (int flavor_b = 0; flavor_b < num_flavors; ++flavor_b) {
        for (int flavor_a = 0; flavor_a < num_flavors; ++flavor_a) {
          for (int il3 = 0; il3 < dim_b; ++il3) {
            for (int il2 = 0; il2 < dim_f; ++il2) {
              for (int il1 = 0; il1 < dim_f; ++il1) {
                result_H_cancel2(il1, il2, il3, flavor_a, flavor_b, flavor_a, flavor_b)
                    += coeff * cancel_abcd(il1, il2, il3, flavor_a, flavor_b);
              }
            }
          }
        }
      }
    } //a==c && b==d

    //Eigen::Tensor<SCALAR,7> tmp = result_H_cancel-result_H_cancel2;
    //std::cout << tmp(0,0,0,0,0,0,0) << " " << result_H_cancel(0,0,0,0,0,0,0) << " " << result_H_cancel2(0,0,0,0,0,0,0) << std::endl;
    //std::cout << tmp.abs().maximum() << std::endl;

  }//if (exact_cancellation)
  const double time5 = timer.elapsed().wall * 1E-9;

  //Then, accumulate data
  Eigen::Tensor<SCALAR,7> result(dim_f, dim_f, dim_b, num_flavors, num_flavors, num_flavors, num_flavors);
  for (int flavor_a = 0; flavor_a < num_flavors; ++flavor_a) {
  for (int flavor_b = 0; flavor_b < num_flavors; ++flavor_b) {
  for (int flavor_c = 0; flavor_c < num_flavors; ++flavor_c) {
  for (int flavor_d = 0; flavor_d < num_flavors; ++flavor_d) {
    for (int il1 = 0; il1 < dim_f; ++il1) {
    for (int il2 = 0; il2 < dim_f; ++il2) {
    for (int il3 = 0; il3 < dim_b; ++il3) {
      result(il1, il2, il3, flavor_a, flavor_b, flavor_c, flavor_d) =
          result_H(il1, flavor_b, il2, flavor_c, il3, flavor_a, flavor_d)
              + result_H_cancel2(il1, il2, il3, flavor_a, flavor_b, flavor_c, flavor_d);
    }
    }
    }
  }
  }
  }
  }
  const double time6 = timer.elapsed().wall * 1E-9;
  //std::cout << "timing21 " << time2-time1 << std::endl;
  //std::cout << "timing32 " << time3-time2 << std::endl;
  //std::cout << "timing43 " << time4-time3 << std::endl;
  //std::cout << "timing54 " << time5-time4 << std::endl;
  //std::cout << "timing65 " << time6-time5 << std::endl;

  return result;
};

//Measure G2 by removing hyridization lines (reference, slow but simple algorithm)
template<typename SCALAR>
Eigen::Tensor<SCALAR,7>
measure_g2_ref(double beta,
           int num_flavors,
           boost::shared_ptr<OrthogonalBasis> p_basis_f,
           boost::shared_ptr<OrthogonalBasis> p_basis_b,
           const std::vector<psi> &creation_ops,
           const std::vector<psi> &annihilation_ops,
           const Eigen::Matrix<SCALAR,Eigen::Dynamic,Eigen::Dynamic> &M
) {
  const double temperature = 1. / beta;
  const int dim_f = p_basis_f->dim();
  const int dim_b = p_basis_b->dim();
  const int num_phys_rows = creation_ops.size();

  assert(M.rows() == creation_ops.size());

  if (creation_ops.size() != annihilation_ops.size() || creation_ops.size() != M.rows()) {
    throw std::runtime_error("Fatal error in measure_g2");
  }
  boost::timer::cpu_timer timer;

  boost::multi_array<double, 3>
      Pl_f(boost::extents[num_phys_rows][num_phys_rows][dim_f]);//annihilator, creator, ir
  boost::multi_array<double, 3>
      Pl_b(boost::extents[num_phys_rows][num_phys_rows][dim_b]);//annihilator, creator, ir
  {
    //Normalization factor of basis functions
    std::vector<double> norm_coeff_f(dim_f), norm_coeff_b(dim_b);
    for (int il = 0; il < dim_f; ++il) {
      norm_coeff_f[il] = sqrt(2.0/p_basis_f->norm2(il));
    }
    for (int il = 0; il < dim_b; ++il) {
      norm_coeff_b[il] = sqrt(2.0/p_basis_b->norm2(il));
    }

    std::vector<double> Pl_tmp(std::max(dim_f, dim_b));
    for (int k = 0; k < num_phys_rows; k++) {
      for (int l = 0; l < num_phys_rows; l++) {
        double argument = annihilation_ops[k].time() - creation_ops[l].time();
        double arg_sign = 1.0;
        if (argument < 0) {
          argument += beta;
          arg_sign = -1.0;
        }
        const double x = 2 * argument * temperature - 1.0;
        p_basis_f->value(x, Pl_tmp);
        for (int il = 0; il < dim_f; ++il) {
          Pl_f[k][l][il] = arg_sign * norm_coeff_f[il] * Pl_tmp[il];
        }

        p_basis_b->value(x, Pl_tmp);
        for (int il = 0; il < dim_b; ++il) {
          Pl_b[k][l][il] = norm_coeff_b[il] * Pl_tmp[il];
        }
      }
    }
  }
  const double time1 = timer.elapsed().wall * 1E-9;

  //The indices of M are reverted from (C. 24) of L. Boehnke (2011) because we're using the F convention here.
  //First, compute relative weights. This costs O(num_phys_rows^4) operations.
  double norm = 0.0;
  for (int a = 0; a < num_phys_rows; ++a) {
    for (int b = 0; b < num_phys_rows; ++b) {
      for (int c = a+1; c < num_phys_rows; ++c) {
        for (int d = b+1; d < num_phys_rows; ++d) {
          /*
           * Delta convention
           * M_ab  M_ad
           * M_cb  M_cd
           */
          norm += std::abs((M(b,a) * M(d,c) - M(b,c) * M(d,a)));
        }
      }
    }
  }
  //The factor 4 is from the degree of freedom of exchange a and c or b and d.
  norm *= 4;

  const SCALAR coeff = 1.0 / (norm * beta);//where is this beta factor from?

  Eigen::Tensor<SCALAR,7> result(dim_f, dim_f, dim_b, num_flavors, num_flavors, num_flavors, num_flavors);
  result.setZero();
  for (int a = 0; a < num_phys_rows; ++a) {
    int flavor_a = annihilation_ops[a].flavor();
    for (int c = 0; c < num_phys_rows; ++c) {
      int flavor_c = annihilation_ops[c].flavor();
      for (int b = 0; b < num_phys_rows; ++b) {
        int flavor_b = creation_ops[b].flavor();
        for (int d = 0; d < num_phys_rows; ++d) {
          int flavor_d = creation_ops[d].flavor();

          if (a == c || b == d) {
            continue;
          }

          for (int il1 = 0; il1 < dim_f; ++il1) {
            for (int il2 = 0; il2 < dim_f; ++il2) {
              for (int il3 = 0; il3 < dim_b; ++il3) {
                result(il1, il2, il3, flavor_a, flavor_b, flavor_c, flavor_d) +=
                    coeff * M(b, a) * M(d, c) * Pl_f[a][b][il1] * Pl_f[c][d][il2] * Pl_b[a][d][il3]
                        * (il2%2 == 0 ? -1.0 : 1.0);
              }
            }
          }
        }
      }
    }
  }

  return result;
};
