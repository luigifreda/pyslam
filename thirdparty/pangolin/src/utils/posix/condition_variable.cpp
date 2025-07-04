#include <pangolin/utils/posix/condition_variable.h>

#include <pangolin/utils/posix/shared_memory_buffer.h>

#include <pthread.h>

using namespace std;

namespace pangolin {

struct PThreadSharedData {
  pthread_mutex_t lock;
  pthread_cond_t cond;
};

class PThreadConditionVariable : public ConditionVariableInterface {
public:
  PThreadConditionVariable(std::shared_ptr<SharedMemoryBufferInterface> &shmem)
      : _shmem(shmem),
        _pthread_data(reinterpret_cast<PThreadSharedData *>(_shmem->ptr())) {}

  ~PThreadConditionVariable() {}

  void wait() {
    _lock();
    pthread_cond_wait(&_pthread_data->cond, &_pthread_data->lock);
    _unlock();
  }

  bool wait(timespec abstime) {
    _lock();
    int err = pthread_cond_timedwait(&_pthread_data->cond, &_pthread_data->lock,
                           &abstime);
    _unlock();

    return 0 == err;
  }

  void signal() { pthread_cond_signal(&_pthread_data->cond); }

  void broadcast() { pthread_cond_broadcast(&_pthread_data->cond); }

private:
  void _lock() { pthread_mutex_lock(&_pthread_data->lock); }

  void _unlock() { pthread_mutex_unlock(&_pthread_data->lock); }

  std::shared_ptr<SharedMemoryBufferInterface> _shmem;
  PThreadSharedData *_pthread_data;
};

std::shared_ptr<ConditionVariableInterface>
create_named_condition_variable(const string &name) {
  std::shared_ptr<SharedMemoryBufferInterface> shmem =
      create_named_shared_memory_buffer(name, sizeof(PThreadSharedData));
  std::shared_ptr<ConditionVariableInterface> ptr;

  PThreadSharedData *pthread_data =
      reinterpret_cast<PThreadSharedData *>(shmem->ptr());

  pthread_mutexattr_t mattr;
  pthread_mutexattr_init(&mattr);
  pthread_mutexattr_setpshared(&mattr, PTHREAD_PROCESS_SHARED);

  pthread_condattr_t cattr;
  pthread_condattr_init(&cattr);
  pthread_condattr_setpshared(&cattr, PTHREAD_PROCESS_SHARED);

  pthread_mutex_init(&pthread_data->lock, &mattr);
  pthread_cond_init(&pthread_data->cond, &cattr);

  ptr.reset(static_cast<ConditionVariableInterface *>(
      new PThreadConditionVariable(shmem)));
  return ptr;
}

std::shared_ptr<ConditionVariableInterface>
open_named_condition_variable(const string &name) {
  std::shared_ptr<SharedMemoryBufferInterface> shmem =
    open_named_shared_memory_buffer(name, true);
  std::shared_ptr<ConditionVariableInterface> ptr;

  if(shmem) {
    ptr.reset(new PThreadConditionVariable(shmem));
  }
  return ptr;
}

}
