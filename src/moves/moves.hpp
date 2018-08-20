#pragma once

#include <cmath>
#include <limits.h>
#include <math.h>

#include <boost/assert.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/math/special_functions/factorials.hpp>
#include <boost/math/special_functions/binomial.hpp>
#include <boost/range/algorithm.hpp>

#include "../operator.hpp"
#include "../operator_util.hpp"
#include "../wide_scalar.hpp"
#include "../update_histogram.hpp"
#include "../accumulator.hpp"
#include "../sliding_window/sliding_window.hpp"
#include "../mc_config.hpp"

/**
 * @brief Change flavors of operators
 */
struct ExchangeFlavor {
  ExchangeFlavor(int *first) : first_(first) {}
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
  OperatorShift(double beta, double shift) : beta_(beta), shift_(shift) {}
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
  typedef std::map<ConfigSpace, double> weight_map_t;
 public:
  LocalUpdater(const std::string &name) : name_(name), num_attempted_(0), num_valid_move_(0), num_accepted_(0) {}
  virtual ~LocalUpdater() {}

  /** Update the configuration. Return true if the update is accepted. */
  bool update(
      alps::random01 &rng, double BETA,
      MonteCarloConfiguration<SCALAR> &mc_config,
      SLIDING_WINDOW &sliding_window,
      const weight_map_t &config_space_weight = weight_map_t()
  );

  /** To be implemented in a derived class */
  virtual bool propose(
      alps::random01 &rng,
      const MonteCarloConfiguration<SCALAR> &mc_config,
      const SLIDING_WINDOW &sliding_window,
      const std::map<ConfigSpace, double> &config_space_weight
  ) {
    return false;
  };

  /** Will be called on the exit of update() */
  virtual void call_back() {};

  /** updates parameters for Monte Carlo updates */
  virtual void update_parameters() {};

  /** fix parameters for Monte Carlo updates before measurement steps */
  virtual void finalize_learning() {}

  /** create measurement */
  virtual void create_measurement_acc_rate(alps::accumulators::accumulator_set &measurements) {
    measurements <<
                 alps::accumulators::NoBinningAccumulator<double>(get_name() + "_attempted_scalar");
    measurements <<
                 alps::accumulators::NoBinningAccumulator<double>(get_name() + "_valid_move_scalar");
    measurements <<
                 alps::accumulators::NoBinningAccumulator<double>(get_name() + "_accepted_scalar");
  }

  /** measure acceptance rate */
  virtual void measure_acc_rate(alps::accumulators::accumulator_set &measurements) {
    measurements[get_name() + "_attempted_scalar"] << num_attempted_;
    measurements[get_name() + "_valid_move_scalar"] << num_valid_move_;
    measurements[get_name() + "_accepted_scalar"] << num_accepted_;
    num_attempted_ = 0;
    num_valid_move_ = 0;
    num_accepted_ = 0;
  }

  virtual std::string get_name() const {
    return name_;
  }

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
  //std::vector<psi> duplicate_check_work_;

  static bool update_operators(operator_container_t &operators,
                               const std::vector<psi> &ops_rem, const std::vector<psi> &ops_add,
                               std::vector<std::pair<psi, ActionType> > &update_record);

  //void revert_operators(MonteCarloConfiguration<SCALAR> &mc_config,
  //const std::vector<psi> &worm_ops_rem, const std::vector<psi> &worm_ops_add);

  void finalize_update();

  std::vector<EXTENDED_REAL> trace_bound;//must be resized

  int num_attempted_, num_valid_move_, num_accepted_;
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
      : BaseType(boost::lexical_cast<std::string>(update_rank) + std::string("-pair_insertion_remover")),
        update_rank_(update_rank),
        num_flavors_(num_flavors),
        tau_low_(-1.0),
        tau_high_(-1.0) {}

  virtual bool propose(
      alps::random01 &rng,
      const MonteCarloConfiguration<SCALAR> &mc_config,
      const SLIDING_WINDOW &sliding_window,
      const std::map<ConfigSpace, double> &config_space_weight
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
                         const MonteCarloConfiguration<SCALAR> &mc_config);

/**
 * Propose removal update
 */
  bool propose_removal(alps::random01 &rng,
                       const MonteCarloConfiguration<SCALAR> &mc_config);
};

/**
 * Change the flavors of a pair of the creation and annihilation operators hybridized with the bath
 */
template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
class OperatorPairFlavorUpdater: public LocalUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW> {
 public:
  typedef LocalUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW> BaseType;
  OperatorPairFlavorUpdater(int num_flavors)
      : BaseType("Operator_pair_flavor_updater"),
        num_flavors_(num_flavors),
        num_attempted_(0.0),
        num_accepted_(0.0) {}

  virtual bool propose(
      alps::random01 &rng,
      const MonteCarloConfiguration<SCALAR> &mc_config,
      const SLIDING_WINDOW &sliding_window,
      const std::map<ConfigSpace, double> &config_space_weight
  );

 private:
  const int num_flavors_;
  double num_attempted_, num_accepted_;
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
      BaseType("Single_operator_shift_updater"),
      num_flavors_(num_flavors),
      max_distance_(num_flavors, 0.5 * beta),
      acc_rate_(num_bins, 0.5 * beta, num_flavors, 0.5 * beta) {}

  virtual bool propose(
      alps::random01 &rng,
      const MonteCarloConfiguration<SCALAR> &mc_config,
      const SLIDING_WINDOW &sliding_window,
      const std::map<ConfigSpace, double> &config_space_weight
  );


  virtual void call_back();

  virtual void update_parameters();

  virtual void finalize_learning() { acc_rate_.reset(); }

  virtual void create_measurement_acc_rate(alps::accumulators::accumulator_set &measurements);

  virtual void measure_acc_rate(alps::accumulators::accumulator_set &measurements);

 private:
  int num_flavors_;
  StepSizeOptimizer acc_rate_;

  std::vector<double> max_distance_;
  double distance_;
  int flavor_;

  static int gen_new_flavor(const MonteCarloConfiguration<SCALAR> &mc_config, int old_flavor, alps::random01 &rng);
};

template<typename SCALAR, typename EXTENDED_SCALAR, typename R, typename SLIDING_WINDOW,
    typename HybridizedOperatorTransformer, typename WormTransformer>
bool
global_update(R &rng,
              double BETA,
              MonteCarloConfiguration<SCALAR> &mc_config,
              std::vector<SCALAR> &det_vec,
              SLIDING_WINDOW &sliding_window,
              int num_flavors,
              const HybridizedOperatorTransformer &hyb_op_transformer,
              const WormTransformer &worm_transformer,
              int Nwin
);

/**
 * Template class for move/insert/remove a worm
 * This class is just a template.
 * Actual updates are managed by a derived class which implements the member function propose().
 */
template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
class WormUpdater: public LocalUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW> {
  typedef LocalUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW> BaseType;

 public:
  WormUpdater(const std::string &str, double beta, int num_flavors, double tau_lower_limit, double tau_upper_limit);
  virtual ~WormUpdater() {}

  virtual bool propose(
      alps::random01 &rng,
      const MonteCarloConfiguration<SCALAR> &mc_config,
      const SLIDING_WINDOW &sliding_window,
      const std::map<ConfigSpace, double> &config_space_weight
  ) = 0;

 protected:
  std::string str_;
  double beta_;
  int num_flavors_;
  double tau_lower_limit_, tau_upper_limit_;
};

/**
 * Class managing the move of the existing worm
 */
template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
class WormMover: public WormUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW> {
 public:
  typedef WormUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW> BaseType;

  WormMover(const std::string &str, double beta, int num_flavors, double tau_lower_limit, double tau_upper_limit)
      : BaseType(str, beta, num_flavors, tau_lower_limit, tau_upper_limit),
        //acc_rate_(100, 0.5 * beta, 1, 0.5 * beta),
        beta_(beta),
        max_distance_(0.5 * beta),
        distance_(-1.0) {}

  //virtual void call_back();

  //virtual void update_parameters();

  //virtual void finalize_learning() { acc_rate_.reset(); }

 private:
  virtual bool propose(
      alps::random01 &rng,
      const MonteCarloConfiguration<SCALAR> &mc_config,
      const SLIDING_WINDOW &sliding_window,
      const std::map<ConfigSpace, double> &config_space_weight
  );

  //StepSizeOptimizer acc_rate_;
  double beta_, max_distance_, distance_;
};

/**
 * Change flavor of worm
 */
template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
class WormFlavorChanger: public WormUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW> {
 public:
  typedef WormUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW> BaseType;

  WormFlavorChanger(const std::string &str,
                    double beta,
                    int num_flavors,
                    double tau_lower_limit,
                    double tau_upper_limit)
      : BaseType(str, beta, num_flavors, tau_lower_limit, tau_upper_limit) {}

 private:
  virtual bool propose(
      alps::random01 &rng,
      const MonteCarloConfiguration<SCALAR> &mc_config,
      const SLIDING_WINDOW &sliding_window,
      const std::map<ConfigSpace, double> &config_space_weight
  );
};

/**
 * Class for the insertion and removal of a worm
 */
template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
class WormInsertionRemover: public WormUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW> {
  typedef WormUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW> BaseType;

 public:
  WormInsertionRemover(const std::string &str,
                       double beta,
                       int num_flavors,
                       double tau_lower_limit,
                       double tau_upper_limit,
                       boost::shared_ptr<Worm> p_worm_template
  ) : BaseType(str, beta, num_flavors, tau_lower_limit, tau_upper_limit), p_worm_template_(p_worm_template),
      insertion_proposal_rate_(0.0) {
  }

  void set_relative_insertion_proposal_rate(double insertion_proposal_rate) {
    insertion_proposal_rate_ = insertion_proposal_rate;
  };

 private:
  virtual bool propose(
      alps::random01 &rng,
      const MonteCarloConfiguration<SCALAR> &mc_config,
      const SLIDING_WINDOW &sliding_window,
      const std::map<ConfigSpace, double> &config_space_weight
  );

  bool propose_by_trace_impl(
      alps::random01 &rng,
      const MonteCarloConfiguration<SCALAR> &mc_config,
      const SLIDING_WINDOW &sliding_window,
      const std::map<ConfigSpace, double> &config_space_weight
  );

  bool propose_by_trace_hyb_impl(
      alps::random01 &rng,
      const MonteCarloConfiguration<SCALAR> &mc_config,
      const SLIDING_WINDOW &sliding_window,
      const std::map<ConfigSpace, double> &config_space_weight
  );

  boost::shared_ptr<Worm> p_worm_template_;
  //double weight_;
  double insertion_proposal_rate_;
};

/**
 * @brief Class for the insertion and removal of a Green's function worm by connecting or cutting hybridization lines
 * This may be more efficient than the general version, WormInsertionRemover,
 * because one does not need to re-evaluate the trace.
 */
template<typename SCALAR, int RANK, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
class GWormInsertionRemover: public LocalUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW> {
  typedef LocalUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW> BaseType;

 public:
  GWormInsertionRemover(const std::string &str,
                        double beta,
                        int num_flavors,
                        boost::shared_ptr<Worm> p_worm_template
  ) : BaseType(str), p_worm_template_(p_worm_template) {
  }

  //virtual void set_worm_space_weight(double weight) {weight_ = weight;};

 private:
  virtual bool propose(
      alps::random01 &rng,
      const MonteCarloConfiguration<SCALAR> &mc_config,
      const SLIDING_WINDOW &sliding_window,
      const std::map<ConfigSpace, double> &config_space_weight
  );

  boost::shared_ptr<Worm> p_worm_template_;
};

/**
 * @brief Exchange flavors of a worm
 */
struct WormExchangeFlavor {
  WormExchangeFlavor(int *first) : first_(first) {}
  boost::shared_ptr<Worm> operator()(const Worm &worm) const {
    boost::shared_ptr<Worm> new_worm = worm.clone();
    for (int findx = 0; findx < worm.num_independent_flavors(); ++findx) {
      new_worm->set_flavor(findx, first_[worm.get_flavor(findx)]);
    }
    return new_worm;
  }
 private:
  int *first_;
};

/**
 * @brief Shift a worm by a constant time
 */
struct WormShift {
  WormShift(double beta, double shift) : beta_(beta), shift_(shift) {}
  boost::shared_ptr<Worm> operator()(const Worm &worm) const {
    assert(shift_ >= 0.0);

    boost::shared_ptr<Worm> new_worm = worm.clone();
    for (int tindx = 0; tindx < worm.num_independent_times(); ++tindx) {
      double new_t = worm.get_time(tindx) + shift_;
      if (new_t > beta_) {
        new_t -= beta_;
      }
      new_worm->set_time(tindx, new_t);
    }
    return new_worm;
  }
 private:
  double beta_, shift_;
};

