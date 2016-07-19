#pragma once

#include <boost/format.hpp>
//#include <boost/date_time/posix_time/posix_time_types.hpp>

#include <alps/mc/mpiadapter.hpp>

#include <time.h>

extern int global_mpi_rank;//for debug

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

  std::pair<bool,bool> run(boost::function<bool()> const &stop_callback) {
    bool done = false, stopped = false;
    bool was_thermalized_before = false;
    const time_t start_time = time(NULL);
    time_t last_output_time = start_time;
    const time_t output_interval = 10;//10 sec
    do {
      const bool is_thermalized = this->is_thermalized();
      if (is_thermalized && !was_thermalized_before) {
        //MPI communication is NOT allowed in prepare_for_measurements()
        this->prepare_for_measurement();
      }

      //MPI communication is NOT allowed in update()
      this->update();

      if (is_thermalized) {
        //MPI communication is NOT allowed in measure()
        this->measure();
      }

      was_thermalized_before = is_thermalized;
      const time_t current_time = time(NULL);
      stopped = stop_callback();
      if (stopped || ABase::schedule_checker.pending()) {
        //const double local_fraction = stopped ? 1.1 : Base::fraction_completed();
        //ABase::schedule_checker.update(
            //ABase::fraction = alps::alps_mpi::all_reduce(ABase::communicator, local_fraction, std::plus<double>()));
        //done = ABase::fraction >= 1.;
        const int local_data = is_thermalized ? 1 : 0;
        const int num_thermalized = alps::alps_mpi::all_reduce(ABase::communicator, local_data, std::plus<int>());
        done = stopped;
        if (ABase::communicator.rank() == 0) {
          if (current_time - last_output_time > output_interval) {
            last_output_time = current_time;
            if (num_thermalized == ABase::communicator.size()) {
              std::cout << "Checking if the simulation is finished: "
                  <<  current_time - start_time << " sec passed." << std::endl;
            } else {
              std::cout
                  << boost::format("%1% processes are not thermalized yet. : %2% sec passed.") % (ABase::communicator.size() - num_thermalized) % static_cast<int>((current_time - start_time))
                  << std::endl;
            }
          }
        }
      }
    } while (!done);
    return std::make_pair(!stopped, this->is_thermalized());
  }
};
