#include <pangolin/log/playback_session.h>
#include <pangolin/utils/params.h>

namespace pangolin {

std::shared_ptr<PlaybackSession> PlaybackSession::Default()
{
    static std::shared_ptr<PlaybackSession> instance = std::make_shared<PlaybackSession>();
    return instance;
}

std::shared_ptr<PlaybackSession> PlaybackSession::ChooseFromParams(const Params& params)
{
    bool use_ordered_playback = params.Get<bool>("OrderedPlayback", false);
    std::shared_ptr<pangolin::PlaybackSession> playback_session;
    if(use_ordered_playback)
    {
        return Default();
    }
    else
    {
        return std::make_shared<PlaybackSession>();
    }
}

}
