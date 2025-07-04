#pragma once

#include <memory>

#include <string>

namespace pangolin
{

  class SemaphoreInterface
  {
  public:

    virtual ~SemaphoreInterface() {
    }

    virtual bool tryAcquire() = 0;
    virtual void acquire() = 0;
    virtual void release() = 0;
  };

  std::shared_ptr<SemaphoreInterface> create_named_semaphore(const std::string& name,
    unsigned int value);
  std::shared_ptr<SemaphoreInterface> open_named_semaphore(const std::string& name);

}
