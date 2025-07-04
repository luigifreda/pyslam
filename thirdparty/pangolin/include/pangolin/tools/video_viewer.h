#pragma once

#include <pangolin/display/window.h>
#include <pangolin/platform.h>
#include <pangolin/video/video_input.h>

#include <functional>
#include <mutex>
#include <string>
#include <thread>

namespace pangolin
{

PANGOLIN_EXPORT
class VideoViewer
{
public:
    typedef std::function<void(const unsigned char* data,
                               const std::vector<Image<unsigned char> >& images,
                               const picojson::value& properties)> FrameChangedCallbackFn;

    static constexpr int FRAME_SKIP = 30;

    VideoViewer(const std::string& window_name, const std::string& input_uri, const std::string& output_uri = "video.pango" );
    VideoViewer(const VideoViewer&) = delete;

    virtual ~VideoViewer();

    void Run();
    void RunAsync();

    void Quit();
    void QuitAndWait();

    inline int TotalFrames() const
    {
        return video_playback ? video_playback->GetTotalFrames() : std::numeric_limits<int>::max();
    }

    // Control playback
    void OpenInput(const std::string& input_uri);
    void CloseInput();

    // Control recording
    void Record();
    void RecordOneFrame();
    void StopRecording();

    // Useful for user-control
    void TogglePlay();
    void ToggleRecord();
    void ToggleDiscardBufferedFrames();
    void ToggleWaitForFrames();
    void SetDiscardBufferedFrames(bool new_state);
    void SetWaitForFrames(bool new_state);
    void Skip(int frames);
    bool ChangeExposure(int delta_us);
    bool ChangeGain(float delta);
    void SetActiveCamera(int delta);
    void DrawEveryNFrames(int n);


    // Register to be notified of new image data
    void SetFrameChangedCallback(FrameChangedCallbackFn cb);

    void WaitUntilExit();


    VideoInput& Video() {return video;}
    const VideoInput& Video() const {return video;}

    void SetRecordNthFrame(int record_nth_frame_) {
      record_nth_frame = record_nth_frame_;
    }


protected:
    void RegisterDefaultKeyShortcutsAndPangoVariables();

    std::mutex control_mutex;
    std::string window_name;
    std::thread vv_thread;

    VideoInput video;
    VideoPlaybackInterface* video_playback;
    VideoInterface* video_interface;

    std::string output_uri;

    int current_frame;
    int grab_until;
    int record_nth_frame;
    int draw_nth_frame;
    bool video_grab_wait;
    bool video_grab_newest;
    bool should_run;
    uint16_t active_cam;

    FrameChangedCallbackFn frame_changed_callback;
};


void PANGOLIN_EXPORT RunVideoViewerUI(const std::string& input_uri, const std::string& output_uri);

}
