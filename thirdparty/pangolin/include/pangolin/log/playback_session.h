#pragma once

#include <memory>
#include <vector>

#include <pangolin/log/packetstream_reader.h>
#include <pangolin/utils/file_utils.h>
#include <pangolin/utils/registration.h>

namespace pangolin {

class Params;

class PlaybackSession
{
public:
    // Singleton Instance
    static std::shared_ptr<PlaybackSession> Default();

    // Return thread-safe, shared instance of PacketStreamReader, providing
    // serialised read for PacketStreamReader
    std::shared_ptr<PacketStreamReader> Open(const std::string& filename)
    {
        const std::string path = SanitizePath(PathExpand(filename));

        auto i = readers.find(path);
        if(i == readers.end()) {
            auto psr = std::make_shared<PacketStreamReader>(path);
            readers[path] = psr;
            return psr;
        }else{
            return i->second;
        }
    }

    SyncTime& Time()
    {
        return time;
    }

    static std::shared_ptr<PlaybackSession> ChooseFromParams(const Params& params);

private:
    std::map<std::string,std::shared_ptr<PacketStreamReader>> readers;
    SyncTime time;
};

}
