#include "timer.h"
#include <iostream>

timeval myTimer1, myTimer2;

int timeval_subtract (timeval *result, timeval *x, timeval *y)
{
  /* Perform the carry for the later subtraction by updating y. */
  if (x->tv_usec < y->tv_usec) {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
    y->tv_usec -= 1000000 * nsec;
    y->tv_sec += nsec;
  }
  if (x->tv_usec - y->tv_usec > 1000000) {
    int nsec = (x->tv_usec - y->tv_usec) / 1000000;
    y->tv_usec += 1000000 * nsec;
    y->tv_sec -= nsec;
       }

  /* Compute the time remaining to wait.
     tv_usec is certainly positive. */
  result->tv_sec = x->tv_sec - y->tv_sec;
  result->tv_usec = x->tv_usec - y->tv_usec;

  /* Return 1 if result is negative. */
  return x->tv_sec < y->tv_sec;
}

void startTimer()
{
  gettimeofday(&myTimer1, 0);

}


double stopTimer()
{
  gettimeofday(&myTimer2, 0);

  timeval result;
  timeval_subtract(&result, &myTimer2, &myTimer1);

  return (result.tv_sec*1000000 + result.tv_usec)/1000.0; // return elapsed time in ms

}


double startTimeMeasure(){
  timeval tim;
  gettimeofday(&tim, 0);
  return tim.tv_sec+(tim.tv_usec/1000000.0);
}


double stopTimeMeasure(double t1){
  timeval tim;
  gettimeofday(&tim, 0);
  double t2=tim.tv_sec+(tim.tv_usec/1000000.0);
  return t2-t1;
}

// ===============================================
// ===============================================
// ===============================================
void Timer::tic(const std::string &name)
{
  if (!_active) return;

  if (measurements.count(name) == 0) create(name);
  measurements[name].startTime = startTimeMeasure();
}

// ===============================================
double Timer::toc(const std::string &name)
{
  if (!_active) return 0;

  if (measurements.count(name) == 0) create(name);
  else measurements[name].overallTime += stopTimeMeasure(measurements[name].startTime);

  return measurements[name].overallTime;
}

// ===============================================
double Timer::get(const std::string &name)
{
  if (!_active) return 0;

  if (measurements.count(name) == 0) return -1;
  else return measurements[name].overallTime;
}


// ===============================================
void Timer::reset(const std::string &name)
{
  if (!_active) return;
  create(name);
}

// ===============================================
void Timer::create(const std::string &name)
{
  if (!_active) return;

  Timer::Measurement m;
  m.name = name;
  m.overallTime = 0;
  m.startTime = 0;
  measurements[name]=m;
}

// ===============================================
void Timer::print(std::ostream &os)
{
  os << "Timings: " << std::endl;
  for (std::map<std::string, Measurement>::iterator iter=measurements.begin(); iter!=measurements.end(); iter++) {
    os << "  " << iter->first.c_str() << " : " << iter->second.overallTime << " s" << std::endl;
  }
}
