#include <string>
#include <vector>
#include <fstream>
#include <tuple>

#include <boost/multi_array.hpp>

#include "../util.hpp"

/*
 * Read nonzero elements of Coulomb tensor from a text file
 */
template <typename SCALAR>
void read_U_tensor(const std::string &input_file, int flavors_, std::vector<std::tuple<int, int, int, int, SCALAR>> &nonzero_U_vals, bool verbose_=true)
{
  std::ifstream infile_f(input_file.c_str());
  if (!infile_f.is_open())
  {
    std::cerr << "We cannot open " << input_file << "!" << std::endl;
    exit(1);
  }
  if (verbose_)
  {
    std::cout << "Reading " << input_file << "..." << std::endl;
  }

  int num_elem;
  infile_f >> num_elem;
  if (num_elem < 0)
  {
    std::runtime_error("The number of elements in U_TENSOR_INPUT_FILE cannot be negative!");
  }
  if (verbose_)
  {
    std::cout << "Number of non-zero elements in U tensor is " << num_elem << std::endl;
  }

  nonzero_U_vals.clear();
  nonzero_U_vals.reserve(num_elem);
  for (int i_elem = 0; i_elem < num_elem; ++i_elem)
  {
    double re, im;
    int line, f0, f1, f2, f3;
    infile_f >> line >> f0 >> f1 >> f2 >> f3 >> re >> im;
    if (line != i_elem)
    {
      throw std::runtime_error(boost::str(boost::format("First column of line %1% is incorrect.") % i_elem));
    }
    if (f0 < 0 || f0 >= flavors_)
    {
      throw std::runtime_error(boost::str(boost::format("Second column of line %1% is incorrect.") % i_elem));
    }
    if (f1 < 0 || f1 >= flavors_)
    {
      throw std::runtime_error(boost::str(boost::format("Third column of line %1% is incorrect.") % i_elem));
    }
    if (f2 < 0 || f2 >= flavors_)
    {
      throw std::runtime_error(boost::str(boost::format("Fourth column of line %1% is incorrect.") % i_elem));
    }
    if (f3 < 0 || f3 >= flavors_)
    {
      throw std::runtime_error(boost::str(boost::format("Fifth column of line %1% is incorrect.") % i_elem));
    }
    const SCALAR uval = 0.5 * mycast<SCALAR>(std::complex<double>(re, im));
    nonzero_U_vals.push_back(std::make_tuple(f0, f1, f2, f3, uval));
  }
}

/*
 * Read hopping matrix from a text file
 */
template <typename SCALAR>
void read_hopping(const std::string &input_file, int flavors_, std::vector<std::tuple<int, int, SCALAR>> &nonzero_t_vals)
{
  std::ifstream infile_f(input_file.c_str());
  if (!infile_f.is_open())
  {
    std::cerr << "We cannot open " << input_file << "!" << std::endl;
    exit(1);
  }

  nonzero_t_vals.resize(0);
  int line = 0;
  for (int f0 = 0; f0 < flavors_; ++f0)
  {
    for (int f1 = 0; f1 < flavors_; ++f1)
    {
      double re, im;
      int f0_in, f1_in;
      infile_f >> f0_in >> f1_in >> re >> im;
      if (f0 != f0_in)
      {
        throw std::runtime_error(boost::str(boost::format("First column of line %1% is incorrect.") % line));
      }
      if (f1 != f1_in)
      {
        throw std::runtime_error(boost::str(boost::format("Second column of line %1% is incorrect.") % line));
      }
      const SCALAR hopping = mycast<SCALAR>(std::complex<double>(re, im));
      if (std::abs(hopping) != 0.0)
      {
        nonzero_t_vals.push_back(std::make_tuple(f0, f1, hopping));
      }
      ++line;
    }
  }
}

/*
 * Read hopping matrix from a text file
 */
template <typename SCALAR>
void read_hybridization_function(const std::string &input_file, int flavors_, int Np1_, boost::multi_array<SCALAR, 3> &F)
{
  F.resize(boost::extents[flavors_][flavors_][Np1_]);
  // read hybridization function from input file with FLAVORS+1 colums \tau, G_1_up, G_1_down, G_2_up ..., G_SITES_down)
  std::ifstream infile_f(input_file.c_str());
  if (!infile_f.is_open())
  {
    std::cerr << "Input file for F cannot be opened!" << std::endl;
    exit(1);
  }

  double real, imag;
  int dummy_it, dummy_i, dummy_j;

  for (int time = 0; time < Np1_; time++)
  {
    for (int i = 0; i < flavors_; i++)
    {
      for (int j = 0; j < flavors_; j++)
      {
        infile_f >> dummy_it >> dummy_i >> dummy_j >> real >> imag;
        if (dummy_it != time)
        {
          throw std::runtime_error("Format of " + input_file + 
                                   " is wrong. The value at the first colum should be " +
                                   boost::lexical_cast<std::string>(time) + "Error at line " +
                                   boost::lexical_cast<std::string>(time + 1) + ".");
        }
        if (dummy_i != i)
        {
          throw std::runtime_error("Format of " + input_file +
                                   " is wrong. The value at the second colum should be " +
                                   boost::lexical_cast<std::string>(i) + "Error at line " +
                                   boost::lexical_cast<std::string>(time + 1) + ".");
        }
        if (dummy_j != j)
        {
          throw std::runtime_error("Format of " + input_file +
                                   " is wrong. The value at the third colum should be " +
                                   boost::lexical_cast<std::string>(j) + "Error at line " +
                                   boost::lexical_cast<std::string>(time + 1) + ".");
        }
        //F_ij(tau) = - Delta_ji (beta - tau)
        F[j][i][Np1_ - time - 1] = -mycast<SCALAR>(std::complex<double>(real, imag));
      }
    }
  }
}