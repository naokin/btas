#ifndef __BTAS_TIME_STAMP_H
#define __BTAS_TIME_STAMP_H

#include <chrono> // C++11 clock classes

/// time stamp class
/// functions to record system clock
class time_stamp {

public:

  typedef std::chrono::time_point<std::chrono::system_clock, std::chrono::microseconds> time_point_type;

private:

  /// starting time point
  time_point_type start_;

  /// lap time record
  time_point_type lap_;

public:

  /// default constructor
  time_stamp () { start(); }

  /// start & restart clock
  void start ()
  {
    start_ = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());
    lap_   = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());
  }

  /// get elapased time in microseconds
  double elapsed (unsigned long periods_ = 1000000 /* to seconds */) const
  {
    time_point_type record_ = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());
    auto t = record_-start_;
    return static_cast<double>(t.count())/periods_;
  }

  /// get lap time in microseconds and reset lap_ to record a next lap
  double lap (unsigned long periods_ = 1000000 /* to seconds */)
  {
    time_point_type record_ = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());
    auto t = record_-lap_;
    lap_ = record_;
    return static_cast<double>(t.count())/periods_;
  }
};

#endif // __BTAS_TIME_STAMP_H
