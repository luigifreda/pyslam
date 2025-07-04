#include <pangolin/factory/factory_registry.h>
#include <pangolin/video/drivers/shared_memory.h>
#include <pangolin/video/iostream_operators.h>

using namespace std;

namespace pangolin
{

SharedMemoryVideo::SharedMemoryVideo(size_t w, size_t h, std::string pix_fmt,
    const std::shared_ptr<SharedMemoryBufferInterface>& shared_memory,
    const std::shared_ptr<ConditionVariableInterface>& buffer_full) :
    _fmt(PixelFormatFromString(pix_fmt)),
    _frame_size(w*h*_fmt.bpp/8),
    _shared_memory(shared_memory),
    _buffer_full(buffer_full)
{
    const size_t pitch = w * _fmt.bpp/8;
    const StreamInfo stream(_fmt, w, h, pitch, 0);
    _streams.push_back(stream);
}

SharedMemoryVideo::~SharedMemoryVideo()
{
}

void SharedMemoryVideo::Start()
{
}

void SharedMemoryVideo::Stop()
{
}

size_t SharedMemoryVideo::SizeBytes() const
{
    return _frame_size;
}

const std::vector<StreamInfo>& SharedMemoryVideo::Streams() const
{
    return _streams;
}

bool SharedMemoryVideo::GrabNext(unsigned char* image, bool wait)
{
    // If a condition variable exists, try waiting on it.
    if(_buffer_full) {
        timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);

        if (wait) {
            _buffer_full->wait();
        } else if (!_buffer_full->wait(ts)) {
            return false;
        }
    }

    // Read the buffer.
    _shared_memory->lock();
    memcpy(image, _shared_memory->ptr(), _frame_size);
    _shared_memory->unlock();

    return true;
}

bool SharedMemoryVideo::GrabNewest(unsigned char* image, bool wait)
{
    return GrabNext(image,wait);
}

PANGOLIN_REGISTER_FACTORY(SharedMemoryVideo)
{
    struct SharedMemoryVideoFactory final : public FactoryInterface<VideoInterface> {
        std::unique_ptr<VideoInterface> Open(const Uri& uri) override {
            const ImageDim dim = uri.Get<ImageDim>("size", ImageDim(0, 0));
            const std::string sfmt = uri.Get<std::string>("fmt", "GRAY8");
            const PixelFormat fmt = PixelFormatFromString(sfmt);
            const std::string shmem_name = std::string("/") + uri.url;
            std::shared_ptr<SharedMemoryBufferInterface> shmem_buffer =
                open_named_shared_memory_buffer(shmem_name, true);
            if (dim.x == 0 || dim.y == 0 || !shmem_buffer) {
                throw VideoException("invalid shared memory parameters");
            }

            const std::string cond_name = shmem_name + "_cond";
            std::shared_ptr<ConditionVariableInterface> buffer_full =
                open_named_condition_variable(cond_name);

            return std::unique_ptr<VideoInterface>(
                new SharedMemoryVideo(dim.x, dim.y, fmt, shmem_buffer,buffer_full)
            );
        }
    };

    FactoryRegistry<VideoInterface>::I().RegisterFactory(std::make_shared<SharedMemoryVideoFactory>(), 10, "shmem");
}

}
