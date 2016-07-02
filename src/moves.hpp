#pragma once

#include <cmath>
#include <limits.h>
#include <math.h>

#include <boost/assert.hpp>
#include <boost/optional.hpp>
#include <boost/math/special_functions/factorials.hpp>
#include <boost/math/special_functions/binomial.hpp>
#include <boost/range/algorithm.hpp>

#include "operator.hpp"
#include "operator_util.hpp"
#include "wide_scalar.hpp"
#include "update_histogram.hpp"
#include "accumulator.hpp"
#include "sliding_window.hpp"

/**
 * @brief Change flavors of operators
 */
struct ExchangeFlavor {
  ExchangeFlavor(int *first) : first_(first) { }
  psi operator()(const psi &op) const {
    psi op_new = op;
    op_new.set_flavor(
        first_[op.flavor()]
    );
    return op_new;
  }
 private:
  int *first_;
};

/**
 * @brief Try to shift the positions of all operators (in imaginary time) by random step size.
 *
 * This update is always accepted if the impurity model is translationally invariant in imaginary time.
 * If you introduce a cutoff in outer states of the trace, it may not be always the case.
 * This update will prevent Monte Carlo dynamics from getting stuck in a local minimum in such cases.
 */
struct OperatorShift {
  OperatorShift(double beta, double shift) : beta_(beta), shift_(shift) { }
  psi operator()(const psi &op) const {
    assert(shift_ >= 0.0);
    psi op_new = op;

    double new_t = op.time().time() + shift_;
    if (new_t > beta_) {
      new_t -= beta_;
    }
    assert(new_t >= 0 && new_t <= beta_);

    OperatorTime new_time(op.time());
    new_time.set_time(new_t);
    op_new.set_time(new_time);
    return op_new;
  }
 private:
  double beta_, shift_;
};

template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
class LocalUpdater {
 public:
  LocalUpdater() { }
  virtual ~LocalUpdater() { }

  /** Update the configuration */
  void update(
      alps::random01 &rng, double BETA,
      MonteCarloConfiguration<SCALAR> &mc_config,
      SLIDING_WINDOW &sliding_window
  );

  /** To be implemented in a derived class */
  virtual bool propose(
      alps::random01 &rng,
      MonteCarloConfiguration<SCALAR> &mc_config,
      const SLIDING_WINDOW &sliding_window
  ) = 0;

  /** Will be called on the exit of update() */
  virtual void call_back() { };

  /** updates parameters for Monte Carlo updates */
  virtual void update_parameters() { };

  /** fix parameters for Monte Carlo updates before measurement steps */
  virtual void finalize_learning() { }

  /** create measurement */
  virtual void create_measurement_acc_rate(alps::accumulators::accumulator_set &measurements) { }

  /** measure acceptance rate */
  virtual void measure_acc_rate(alps::accumulators::accumulator_set &measurements) { }

 protected:
  std::string name_;//name of this updator

  //the following variables will be set in virtual function propose();
  boost::optional<SCALAR> acceptance_rate_correction_;
  std::vector<psi> cdagg_ops_rem_; //hybrized with bath
  std::vector<psi> c_ops_rem_;     //hybrized with bath
  std::vector<psi> cdagg_ops_add_; //hybrized with bath
  std::vector<psi> c_ops_add_;     //hybrized with bath
  boost::shared_ptr<Worm> p_new_worm_; //New worm

  //some variables set on the exit of update()
  bool valid_move_generated_;
  bool accepted_;

 private:
  std::vector<psi> duplicate_check_work_;

  bool update_operators(MonteCarloConfiguration<SCALAR> &mc_config);

  void revert_operators(MonteCarloConfiguration<SCALAR> &mc_config);

  void finalize_update();

  std::vector<EXTENDED_REAL> trace_bound;//must be resized
};

/**
 * Update creation and annihilation operators hybridized with the bath
 * Do not update the worm
 */
template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
class InsertionRemovalUpdater: public LocalUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW> {
 public:
  typedef LocalUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW> BaseType;
  InsertionRemovalUpdater(int update_rank, int num_flavors)
      : update_rank_(update_rank),
        num_flavors_(num_flavors),
        tau_low_(-1.0),
        tau_high_(-1.0) { }

  virtual bool propose(
      alps::random01 &rng,
      MonteCarloConfiguration<SCALAR> &mc_config,
      const SLIDING_WINDOW &sliding_window
  );

 protected:
  typedef operator_container_t::iterator it_t;

  const int num_flavors_;
  double tau_low_, tau_high_; //, max_distance_;

  std::vector<int> num_cdagg_ops_in_range_, num_c_ops_in_range_;
  std::vector<std::pair<it_t, it_t> > cdagg_ops_range_, c_ops_range_;

  /** 1 for two-operator update, 2 for four-operator update, ..., N for 2N-operator update*/
  const int update_rank_;

/**
 * Propose insertion update
 */
  bool propose_insertion(alps::random01 &rng,
                         MonteCarloConfiguration<SCALAR> &mc_config);

/**
 * Propose removal update
 */
  bool propose_removal(alps::random01 &rng,
                       MonteCarloConfiguration<SCALAR> &mc_config);
};

/**
 * Update creation and annihilation operators hybridized with the bath (the same flavor)
 * Do not update the worm
 */
template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
class InsertionRemovalDiagonalUpdater: public LocalUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW> {
 public:
  typedef LocalUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW> BaseType;
  InsertionRemovalDiagonalUpdater(int update_rank, int num_flavors, double beta, int num_bins)
      : update_rank_(update_rank),
        num_flavors_(num_flavors),
        beta_(beta),
        tau_low_(-1.0),
        tau_high_(-1.0),
        acc_rate_(num_bins, 0.5 * beta, num_flavors, 0.5 * beta) { }

  virtual bool propose(
      alps::random01 &rng,
      MonteCarloConfiguration<SCALAR> &mc_config,
      const SLIDING_WINDOW &sliding_window
  );

  virtual void call_back();

  virtual void finalize_learning() { acc_rate_.reset(); }

  virtual void create_measurement_acc_rate(alps::accumulators::accumulator_set &measurements);

  virtual void measure_acc_rate(alps::accumulators::accumulator_set &measurements);

 private:
  const int num_flavors_;
  const double beta_;
  double tau_low_, tau_high_;
  int flavor_;

  std::vector<psi> cdagg_ops_in_range_, c_ops_in_range_;

  /** 1 for two-operator update, 2 for four-operator update, ..., N for 2N-operator update*/
  const int update_rank_;

  scalar_histogram_flavors acc_rate_;
  double distance_;
};

/**
 * Update creation and annihilation operators hybridized with the bath (the same flavor)
 * Do not update the worm
 */
template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
class SingleOperatorShiftUpdater: public LocalUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW> {
 public:
  typedef LocalUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW> BaseType;
  SingleOperatorShiftUpdater(double beta, int num_flavors, int num_bins) :
      num_flavors_(num_flavors),
      max_distance_(num_flavors, 0.5 * beta),
      acc_rate_(num_bins, 0.5 * beta, num_flavors, 0.5 * beta) { }

  virtual bool propose(
      alps::random01 &rng,
      MonteCarloConfiguration<SCALAR> &mc_config,
      const SLIDING_WINDOW &sliding_window
  );


  virtual void call_back();

  virtual void update_parameters();

  virtual void finalize_learning() { acc_rate_.reset(); }

  virtual void create_measurement_acc_rate(alps::accumulators::accumulator_set &measurements);

  virtual void measure_acc_rate(alps::accumulators::accumulator_set &measurements);

 private:
  int num_flavors_;
  scalar_histogram_flavors acc_rate_;

  std::vector<double> max_distance_;
  double distance_;
  int flavor_;

  static int gen_new_flavor(const MonteCarloConfiguration<SCALAR> &mc_config, int old_flavor, alps::random01 &rng);
};

template<typename SCALAR, typename EXTENDED_SCALAR, typename R, typename SLIDING_WINDOW, typename OperatorTransformer>
bool
    global_update(R &rng,
                  double BETA,
                  MonteCarloConfiguration<SCALAR> &mc_config,
                  std::vector<SCALAR> &det_vec,
                  SLIDING_WINDOW &sliding_window,
                  int num_flavors,
                  OperatorTransformer transformer,
                  int Nwin
);

/**
 * Template class for move/insert/remove a worm
 * This class is just a template.
 * Actual updates are managed by a derived class which implements the member function propose().
 */
template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
class WormUpdater: public LocalUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW> {
 public:
  WormUpdater(const std::string &str, double beta, int num_flavors, double tau_lower_limit, double tau_upper_limit);
  virtual ~WormUpdater() { }

  virtual bool propose(
      alps::random01 &rng,
      MonteCarloConfiguration<SCALAR> &mc_config,
      const SLIDING_WINDOW &sliding_window
  ) = 0;

  //set the additional weight of the worm configuration space
  virtual void set_worm_space_weight(double weight) {worm_space_weight_ = weight;};

  virtual double worm_space_weight() const {return worm_space_weight_;};

  /** Will be called on the exit of update() */
  virtual void call_back();

  /** updates parameters for Monte Carlo updates */
  virtual void update_parameters();

  /** fix parameters for Monte Carlo updates before measurement steps */
  virtual void finalize_learning() { acc_rate_.reset(); }

  /** create measurement */
  virtual void create_measurement_acc_rate(alps::accumulators::accumulator_set &measurements);

  /** measure acceptance rate */
  virtual void measure_acc_rate(alps::accumulators::accumulator_set &measurements);

 protected:
  std::string str_;
  double beta_;
  int num_flavors_;
  double tau_lower_limit_, tau_upper_limit_;
  scalar_histogram_flavors acc_rate_;
  double max_distance_, distance_;
  double worm_space_weight_;
};

/**
 * Class managing the move of the existing worm
 */
template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
class WormMover: public WormUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW> {
  typename WormMover<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW> BaseType;

  //WormMover(const std::string &str, double beta, int num_flavors, double tau_lower_limit, double tau_upper_limit)
      //: BaseType(str, beta, num_flavors, tau_lower_limit, tau_upper_limit) { }
  WormMover(const std::string &str, double beta, int num_flavors, double tau_lower_limit, double tau_upper_limit) {}

  //virtual ~WormMover() : BaseType() { }

  virtual bool propose(
      alps::random01 &rng,
      MonteCarloConfiguration<SCALAR> &mc_config,
      const SLIDING_WINDOW &sliding_window
  );

  virtual void set_weight();
};

/**
 * Class managing the insertion and removal of a worm
 */
template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
class WormInsertionRemover: public WormUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW> {
  typename WormInsertionRemover<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW> BaseType;

  WormInsertionRemover(const std::string &str,
                       double beta,
                       int num_flavors,
                       double tau_lower_limit,
                       double tau_upper_limit,
                       boost::shared_ptr<Worm> p_worm_template
  ) : BaseType(str, beta, num_flavors, tau_lower_limit, tau_upper_limit), p_worm_template_(p_worm_template) {
  }

  virtual bool propose(
      alps::random01 &rng,
      MonteCarloConfiguration<SCALAR> &mc_config,
      const SLIDING_WINDOW &sliding_window
  );

  virtual void set_weight();

 private:
  boost::shared_ptr<Worm> p_worm_template_;

};

#include "moves.ipp"
