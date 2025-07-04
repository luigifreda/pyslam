#pragma once

#include <pangolin/pangolin.h>
#include <pangolin/video/video.h>
#include <pangolin/utils/posix/condition_variable.h>
#include <pangolin/utils/posix/shared_memory_buffer.h>

#include <memory>
#include <vector>

namespace pangolin
{

class SharedMemoryVideo : public VideoInterface
{
public:
  SharedMemoryVideo(size_t w, size_t h, std::string pix_fmt,
    const std::shared_ptr<SharedMemoryBufferInterface>& shared_memory,
    const std::shared_ptr<ConditionVariableInterface>& buffer_full);
  ~SharedMemoryVideo();

  size_t SizeBytes() const;
  const std::vector<StreamInfo>& Streams() const;
  void Start();
  void Stop();
  bool GrabNext(unsigned char *image, bool wait);
  bool GrabNewest(unsigned char *image, bool wait);

private:
  PixelFormat _fmt;
  size_t _frame_size;
  std::vector<StreamInfo> _streams;
  std::shared_ptr<SharedMemoryBufferInterface> _shared_memory;
  std::shared_ptr<ConditionVariableInterface> _buffer_full;
};

}
