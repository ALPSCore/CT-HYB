#pragma once

#include <utility>
#include <tuple>

template<typename T>
class HashPair
{
public:
    std::size_t operator()(const std::pair<T,T>& p) const
    {
      std::size_t seed = 0;
      boost::hash_combine(seed, p.first);
      boost::hash_combine(seed, p.second);
      return seed;
    }
};

template<typename T>
class HashTuple3
{
public:
    std::size_t operator()(const std::tuple<T,T,T>& p) const
    {
      std::size_t seed = 0;
      boost::hash_combine(seed, std::get<0>(p));
      boost::hash_combine(seed, std::get<1>(p));
      boost::hash_combine(seed, std::get<2>(p));
      return seed;
    }
};

using HashIntPair = HashPair<int>;
using HashIntTuple3 = HashTuple3<int>;
