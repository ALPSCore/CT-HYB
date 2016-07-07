#pragma once

#include <boost/format.hpp>
//#include <boost/date_time/posix_time/posix_time_types.hpp>

#include <alps/mc/mpiadapter.hpp>

#include <time.h>

template<typename Base, typename ScheduleChecker = alps::check_schedule>
class mymcmpiadapter: public alps::mcmpiadapter<Base, ScheduleChecker> {
 public:
  typedef typename Base::parameters_type parameters_type;
  typedef typename alps::mcmpiadapter<Base> ABase;
  //typedef boost::posix_time::ptime ptime;

  /// Construct mcmpiadapter with alps::check_schedule with the relevant parameters Tmin and Tmax taken from the provided parameters
  mymcmpiadapter(parameters_type const &parameters, alps::mpi::communicator const &comm) : alps::mcmpiadapter<Base>(
      parameters,
      comm) { }

  bool run(boost::function<bool()> const &stop_callback) {
    bool done = false, stopped = false;
    bool was_thermalized_before = false;
    const time_t start_time = time(NULL);
    time_t time_last_output = start_time;
    const double min_output_interval = 10; //1 sec
    do {
      const bool is_thermalized = this->is_thermalized();
      if (is_thermalized && !was_thermalized_before) {
        this->prepare_for_measurement();
      }

      this->update();

      if (is_thermalized) {
        this->measure();
      }

      was_thermalized_before = is_thermalized;
      const time_t current_time = time(NULL);
      //if (stopped || ABase::schedule_checker.pending()) {
        //std::cout << "collecting " << " " << std::endl;
        stopped = stop_callback();
        double local_fraction = stopped ? 1. : Base::fraction_completed();
        ABase::schedule_checker.update(
            ABase::fraction = alps::alps_mpi::all_reduce(ABase::communicator, local_fraction, std::plus<double>()));
        if (ABase::communicator.rank() == 0 && (current_time - time_last_output) > min_output_interval) {
          std::cout << "Checking if the simulation is finished: "
              << std::min(static_cast<int>(ABase::fraction * 100), 100) << "% of Monte Carlo steps done." << std::endl;
          time_last_output = current_time;
        }
        done = ABase::fraction >= 1.;
        //} else if (!stopped && !is_thermalized) {
        //if ((current_time - time_last_output) > min_output_interval) {
        //if (ABase::communicator.rank() == 0) {
        //std::cout
        //<< boost::format("Not thermalized yet: %1% sec passed.") % static_cast<int>((current_time - start_time))
        //<< std::endl;
        //}
        //time_last_output = current_time;
        //}
      //}
    } while (!done);
    return !stopped;
  }
};
