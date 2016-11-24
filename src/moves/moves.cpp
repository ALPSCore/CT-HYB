#include <alps/mc/random01.hpp>

#include "moves.hpp"
#include "moves.ipp"

typedef SlidingWindowManager<REAL_EIGEN_BASIS_MODEL> SW_REAL_MATRIX;
typedef SlidingWindowManager<COMPLEX_EIGEN_BASIS_MODEL> SW_COMPLEX_MATRIX;

#undef PP_REAL
#undef PP_EXTENDED_SCALAR
#undef PP_SW
#define PP_REAL double
#define PP_EXTENDED_SCALAR EXTENDED_REAL
#define PP_SW SW_REAL_MATRIX
#include "moves_explicit.def"

#undef PP_REAL
#undef PP_EXTENDED_SCALAR
#undef PP_SW
#define PP_REAL std::complex<double>
#define PP_EXTENDED_SCALAR EXTENDED_COMPLEX
#define PP_SW SW_COMPLEX_MATRIX
#include "moves_explicit.def"
