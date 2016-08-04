#include <alps/params.hpp>

#include <boost/random.hpp>

#include<Eigen/Dense>
#include<Eigen/LU>

#include "gtest.h"

#include <alps/fastupdate/detail/util.hpp>
#include "../src/model/model.hpp"

/*
template<class S>
void
init_config(int seed, int FLAVORS, int Np1, double BETA, int N_init_pairs, std::vector<std::vector<std::vector<S> > >& F, operator_container_t& operators, operator_container_t& creation_operators, operator_container_t& annihilation_operators);

template<>
void
init_config<double>(int seed, int FLAVORS, int Np1, double BETA, int N_init_pairs,
         std::vector<std::vector<std::vector<double> > >& F, operator_container_t& operators, operator_container_t& creation_operators, operator_container_t& annihilation_operators) {

    boost::random::mt19937 gen(seed);
    boost::uniform_smallint<> dist(0,FLAVORS-1);

    boost::uniform_real<> uni_dist(0,1);
    typedef double SCALAR;

    //initialize F
    F.resize(FLAVORS);
    for (int i=0; i<FLAVORS; i++) {
        F[i].resize(FLAVORS);
        for (int j=0; j<FLAVORS; j++) {
            F[i][j].resize(Np1);
        }
    }
    for (int i=0; i<FLAVORS; i++) {
        for (int j=0; j<FLAVORS; j++) {
            for (int time=0; time<Np1; ++time) {
                //F[i][j][time] = static_cast<SCALAR>(std::complex<double>(uni_dist(gen),uni_dist(gen)));
                F[i][j][time] = uni_dist(gen);
            }
        }
    }

    for (int i=0; i<N_init_pairs; ++i) {
        int flavor_ins = dist(gen);
        int flavor_rem = dist(gen);
        double t_ins = BETA*uni_dist(gen);
        double t_rem = BETA*uni_dist(gen);
        operators_insert_nocopy(operators, t_ins, t_rem, flavor_ins, flavor_rem);
        creation_operators.insert(psi(t_ins,0,flavor_ins));
        annihilation_operators.insert(psi(t_rem,1,flavor_rem));
    }
    assert(creation_operators.size()==annihilation_operators.size());
    assert(operators.size()==2*annihilation_operators.size());
}

template<>
void
init_config<std::complex<double> >(int seed, int FLAVORS, int Np1, double BETA, int N_init_pairs,
         std::vector<std::vector<std::vector<std::complex<double> > > >& F, operator_container_t& operators, operator_container_t& creation_operators, operator_container_t& annihilation_operators) {

    boost::random::mt19937 gen(seed);
    boost::uniform_smallint<> dist(0,FLAVORS-1);
    boost::uniform_real<> uni_dist(0,1);
    typedef std::complex<double> SCALAR;

    //initialize F
    F.resize(FLAVORS);
    for (int i=0; i<FLAVORS; i++) {
        F[i].resize(FLAVORS);
        for (int j=0; j<FLAVORS; j++) {
            F[i][j].resize(Np1);
        }
    }
    for (int i=0; i<FLAVORS; i++) {
        for (int j=0; j<FLAVORS; j++) {
            for (int time=0; time<Np1; ++time) {
                F[i][j][time] = std::complex<double>(uni_dist(gen),uni_dist(gen));
            }
        }
    }

    for (int i=0; i<N_init_pairs; ++i) {
        int flavor_ins = dist(gen);
        int flavor_rem = dist(gen);
        double t_ins = BETA*uni_dist(gen);
        double t_rem = BETA*uni_dist(gen);
        operators_insert_nocopy(operators, t_ins, t_rem, flavor_ins, flavor_rem);
        creation_operators.insert(psi(t_ins,0,flavor_ins));
        annihilation_operators.insert(psi(t_rem,1,flavor_rem));
    }
    assert(creation_operators.size()==annihilation_operators.size());
    assert(operators.size()==2*annihilation_operators.size());
}
*/

template<typename T>
boost::tuple<int,int,int,int,T>
get_tuple(int o0, int o1, int o2, int o3, int spin, int spin2, T val, int sites) {
  return boost::make_tuple(o0+spin*sites, o1+spin2*sites, o2+spin2*sites, o3+spin*sites, val);
};
