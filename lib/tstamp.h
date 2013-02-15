#ifndef _TIME_STAMP_H
#define _TIME_STAMP_H 1

#include <ostream>
extern "C"
{
#include <sys/time.h>
}

class TimeStamp
{
private:
  timeval
    m_tval_obj;

  int
    m_tot_sec;
  int
    m_tot_usec;
  int
    m_lap_sec;
  int
    m_lap_usec;

  bool
    m_active;

public:
  TimeStamp()
  {
    m_tot_sec  = 0;
    m_tot_usec = 0;
    m_lap_sec  = 0;
    m_lap_usec = 0;
    m_active   = false;
  }

  void start()
  {
    if(m_active) {
      timeval tval_end;
      gettimeofday(&tval_end, NULL);
      m_tot_sec  += (tval_end.tv_sec  - m_tval_obj.tv_sec);
      m_tot_usec += (tval_end.tv_usec - m_tval_obj.tv_usec);
    }
    m_lap_sec  = 0;
    m_lap_usec = 0;
    m_active   = true;
    gettimeofday(&m_tval_obj, NULL);
  }

  void stop()
  {
    if(m_active) {
      timeval tval_end;
      gettimeofday(&tval_end, NULL);
      m_lap_sec   = tval_end.tv_sec  - m_tval_obj.tv_sec;
      m_lap_usec  = tval_end.tv_usec - m_tval_obj.tv_usec;
      m_tot_sec  += m_lap_sec;
      m_tot_usec += m_lap_usec;
      m_active    = false;
    }
  }

  void reset()
  {
    m_tot_sec  = 0;
    m_tot_usec = 0;
    m_lap_sec  = 0;
    m_lap_usec = 0;
    m_active   = false;
  }

  double lap()
  {
    double lap_time;
    if(m_active) {
      timeval tval_lap;
      gettimeofday(&tval_lap, NULL);
      m_lap_sec   = tval_lap.tv_sec  - m_tval_obj.tv_sec;
      m_lap_usec  = tval_lap.tv_usec - m_tval_obj.tv_usec;
        lap_time  = (double) (m_lap_sec) + ((double) (m_lap_usec)) / 1000000;
      m_tot_sec  += m_lap_sec;
      m_tot_usec += m_lap_usec;
      m_lap_sec   = 0;
      m_lap_usec  = 0;
      gettimeofday(&m_tval_obj, NULL);
    }
    else {
        lap_time  = (double) (m_lap_sec) + ((double) (m_lap_usec)) / 1000000;
    }
    return lap_time;
  }

  double total()
  {
    stop();
    return (double) (m_tot_sec) + ((double) (m_tot_usec)) / 1000000;
  }
};

#endif // _TIME_STAMP_H
