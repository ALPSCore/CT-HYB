#pragma once

#include <boost/format.hpp>
//#include <boost/date_time/posix_time/posix_time_types.hpp>

#include <alps/mc/mpiadapter.hpp>

#include <time.h>

class my_check_schedule
{
 public:
  typedef boost::chrono::high_resolution_clock clock;
  typedef clock::time_point time_point;
  typedef boost::chrono::duration<double> duration;

  /// Constructor
  my_check_schedule(double tcheck = 10)
      :   tcheck_(tcheck)
  {
  }

  bool pending() const
  {
    time_point now = clock::now();
    return now > (last_check_time_ + next_check_);
  }

  void update(double fraction)
  {
    time_point now = clock::now();
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
template<typename Base> class mymcmpiadapter : public alps::mcmpiadapter<Base,my_check_schedule> {
 private:
  typedef alps::mcmpiadapter<Base,my_check_schedule> base_type_;
  alps::mpi::communicator comm_;

 public:
  typedef typename base_type_::parameters_type parameters_type;

  /// Construct mcmpiadapter with a custom scheduler
  // Just forwards to the base class constructor
  mymcmpiadapter(
      parameters_type const & parameters,
      alps::mpi::communicator const & comm
  ) : base_type_(parameters, comm, my_check_schedule()), comm_(comm)
  {}

  std::pair<bool,bool> run(boost::function<bool ()> const & stop_callback) {
    bool done = false, stopped = false;
    const time_t start_time = time(NULL);
    bool all_processes_thermalized = false;
    bool this_thermalized = false;
    time_t last_output_time = time(NULL);

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
      if (stopped || base_type_::schedule_checker.pending() || !all_processes_thermalized) {
        stopped = stop_callback();
        done = stopped;
        base_type_::schedule_checker.update(0.0);
        if (base_type_::communicator.rank() == 0 && time(NULL) - last_output_time > 1.0) {
          std::cout << "Checking if the simulation is finished: "
                  <<  time(NULL) - start_time << " sec passed." << std::endl;
          last_output_time = time(NULL);
        }
      }
    } while(!done);

    this->finish_measurement();
    return std::make_pair(!stopped, this->is_thermalized());
  }

  static parameters_type& define_parameters(parameters_type & parameters) {
    base_type_::define_parameters(parameters);
    return parameters;
  }
};
