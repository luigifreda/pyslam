#pragma once

#include <pangolin/utils/timer.h>

#include <memory>

namespace pangolin
{

class ConditionVariableInterface
{
public:
  virtual ~ConditionVariableInterface()
  {
  }

  virtual void wait() = 0;
  virtual bool wait(timespec t) = 0;
  virtual void signal() = 0;
  virtual void broadcast() = 0;
};

std::shared_ptr<ConditionVariableInterface> create_named_condition_variable(const
  std::string& name);
std::shared_ptr<ConditionVariableInterface> open_named_condition_variable(const
  std::string& name);
}
