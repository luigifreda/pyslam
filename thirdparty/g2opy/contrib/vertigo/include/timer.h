#pragma once

#include "sys/time.h"
#include <map>
#include <string>


int timeval_subtract (timeval *result, timeval *x, timeval *y);
void startTimer();
double stopTimer();

/**
  start a time measurement
  - to get passed time since started, use stopTimeMeasure with the return parameter of startTimeMeasure as parameter
  - it's possible to have several time measurements run at the same time

  example:
    double t = startTimeMeasure();
    functionToEvaluate();
    cout << "passed time: "<< stopTimeMeasure(t)<<endl;

*/
double startTimeMeasure();

/**
  evaluate a time measurement
  - has to be called with a return value from a prior startTimeMeasure call
  - the same return value can be used later again for a further time measurement
*/
double stopTimeMeasure(double t1);


// ===============================================
class Timer
{
  public:
    Timer() : _active(true) {};

    void tic(const std::string &name);
    double toc(const std::string &name);
    double get(const std::string &name);
    void reset(const std::string &name);
    void create(const std::string &name);

    void print(std::ostream &os);

    void active(bool a) { _active = a;};
    bool active() { return _active; };

  protected:
    bool _active;

    struct Measurement {
      double startTime;
      double overallTime;
      std::string name;
    };

    std::map<std::string, Measurement> measurements;
};
