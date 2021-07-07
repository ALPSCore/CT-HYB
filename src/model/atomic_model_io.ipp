#include <string>
#include <vector>
#include <fstream>
#include <tuple>

#include <boost/multi_array.hpp>

#include "../common/util.hpp"

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