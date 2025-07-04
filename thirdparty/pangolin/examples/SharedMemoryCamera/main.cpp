#include <pangolin/pangolin.h>

#include <pangolin/utils/posix/condition_variable.h>
#include <pangolin/utils/posix/shared_memory_buffer.h>
#include <pangolin/utils/timer.h>

#include <cmath>
#include <memory>

// This sample acts as a soft camera. It writes a pattern of GRAY8 pixels to the
// shared memory space. It can be seen in Pangolin's SimpleVideo sample using
// the shmem:[size=640x480]//example video URI.

using namespace pangolin;

unsigned char generate_value(double t)
{
  // 10s sinusoid
  const double d = std::sin(t * 10.0 / M_PI) * 128.0 + 128.0;
  return static_cast<unsigned char>(d);
}

int main(/*int argc, char *argv[]*/)
{
  std::string shmem_name = "/example";

  std::shared_ptr<SharedMemoryBufferInterface> shmem_buffer =
    create_named_shared_memory_buffer(shmem_name, 640 * 480);
  if (!shmem_buffer) {
    perror("Unable to create shared memory buffer");
    exit(1);
  }

  std::string cond_name = shmem_name + "_cond";
  std::shared_ptr<ConditionVariableInterface> buffer_full =
    create_named_condition_variable(cond_name);

  // Sit in a loop and write gray values based on some timing pattern.
  while (true) {
    shmem_buffer->lock();
    unsigned char *ptr = shmem_buffer->ptr();
    unsigned char value = generate_value(std::chrono::system_clock::now().time_since_epoch().count());

    for (int i = 0; i < 640*480; ++i) {
      ptr[i] = value;
    }

    shmem_buffer->unlock();
    buffer_full->signal();

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}
