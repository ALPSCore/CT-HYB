#pragma once

#include <ctime>
#include "../common/logger.hpp"

class MyCheckSchedule
{
 public:
  typedef std::time_t time_point;
  typedef double duration;

  /// Constructor
  MyCheckSchedule(double tcheck = 10)
      :   tcheck_(tcheck)
  {
  }

  bool pending() const
  {
    time_point now = std::time(NULL);
    return now > (last_check_time_ + next_check_);
  }

  void update(double fraction)
  {
    time_point now = std::time(NULL);
    next_check_ = tcheck_;
    last_check_time_ = now;
  }

 private:
  duration tcheck_;

  time_point start_time_;
  time_point last_check_time_;
  duration next_check_;
};

/// mycustum MPI adapter for a MC simulation class
/// For use, a MC simulation class implement several additional member functions:
///   update_thermalized_status()
///   prepare_for_measurement()
///   finish_measurement()
template<typename Base> class mymcmpiadapter : public Base {
 private:
  alps::params parameters_;
  alps::mpi::communicator comm_;
  MyCheckSchedule checker_;

 public:
  typedef typename alps::params parameters_type;
  typedef typename alps::accumulators::result_set results_type;

  /// Construct mcmpiadapter with a custom scheduler
  // Just forwards to the base class constructor
  mymcmpiadapter(
      alps::params const & parameters,
      alps::mpi::communicator const & comm
  ) : Base(parameters, comm), parameters_(parameters), comm_(comm), checker_()
  {}

  std::pair<bool,bool> run(boost::function<bool ()> const & stop_callback) {
    bool done = false, stopped = false;
    const std::time_t start_time = std::time(NULL);
    bool all_processes_thermalized = false;
    bool this_thermalized = false;
    std::time_t last_output_time = std::time(NULL);

    do {
      if (!this->is_thermalized()) {
        this->update_thermalization_status();
        this_thermalized = this->is_thermalized();
      }

      if (!all_processes_thermalized) {
        int in_buff = this_thermalized ? 1 : 0;
        int out_buff;
        MPI_Allreduce((void *) &in_buff,
                    (void *) &out_buff,
                    1,
                    MPI_INTEGER,
                    MPI_PROD,
                    comm_);
        if (out_buff != 0) {
          all_processes_thermalized = true;
          this->prepare_for_measurement();
        }
      }

      this->update();
      if (all_processes_thermalized) {
        this->measure();
      }
      if (stopped || checker_.pending() || !all_processes_thermalized) {
        stopped = stop_callback();
        done = stopped;
        checker_.update(0.0);
        if (comm_.rank() == 0 && std::time(NULL) - last_output_time > 1.0) {
          logger_out << "Checking if the simulation is finished: "
                  <<  std::time(NULL) - start_time << " sec passed." << std::endl;
          last_output_time = std::time(NULL);
        }
      }
    } while(!done);

    this->finish_measurement();
    return std::make_pair(!stopped, this->is_thermalized());
  }

  static parameters_type& define_parameters(parameters_type & parameters) {
    //base_type_::define_parameters(parameters);
    return parameters;
  }
};
