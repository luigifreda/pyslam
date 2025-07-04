#include <pangolin/tools/video_viewer.h>

#include <pangolin/display/image_view.h>
#include <pangolin/gl/glpixformat.h>
#include <pangolin/pangolin.h>
#include <pangolin/utils/file_utils.h>
#include <pangolin/utils/sigstate.h>
#include <pangolin/video/video_input.h>


namespace pangolin
{

void videoviewer_signal_quit(int) {
    pango_print_info("Caught signal. Program will exit after any IO is complete.\n");
    pangolin::QuitAll();
}

VideoViewer::VideoViewer(const std::string& window_name, const std::string& input_uri, const std::string& output_uri)
    : window_name(window_name),
      video_playback(nullptr),
      video_interface(nullptr),
      output_uri(output_uri),
      current_frame(-1),
      grab_until(std::numeric_limits<int>::max()),
      record_nth_frame(1),
      draw_nth_frame(1),
      video_grab_wait(true),
      video_grab_newest(false),
      should_run(true),
      active_cam(0)
{
    pangolin::Var<int>::Attach("ui.frame", current_frame);
    pangolin::Var<int>::Attach("ui.record_nth_frame", record_nth_frame);
    pangolin::Var<int>::Attach("ui.draw_nth_frame", draw_nth_frame);


    if(!input_uri.empty()) {
        OpenInput(input_uri);
    }
}

VideoViewer::~VideoViewer()
{
    QuitAndWait();

}

void VideoViewer::Quit()
{
    // Signal any running thread to stop
    should_run = false;
}

void VideoViewer::QuitAndWait()
{
    Quit();

    if(vv_thread.joinable()) {
        vv_thread.join();
    }
}

void VideoViewer::RunAsync()
{
    if(!should_run) {
        // Make sure any other thread has finished
        if(vv_thread.joinable()) {
            vv_thread.join();
        }

        // Launch in another thread
        vv_thread = std::thread(&VideoViewer::Run, this);
    }
}

void VideoViewer::Run()
{
    should_run = true;

    /////////////////////////////////////////////////////////////////////////
    /// Register pangolin variables
    /////////////////////////////////////////////////////////////////////////

    std::unique_ptr<unsigned char[]> buffer(new unsigned char[video.SizeBytes()+1]);

    const int slider_size = (TotalFrames() < std::numeric_limits<int>::max() ? 20 : 0);

    // Create OpenGL window - guess sensible dimensions
    pangolin::CreateWindowAndBind( window_name,
        (int)(video.Width() * video.Streams().size()),
        (int)(video.Height() + slider_size)
    );
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::Var<int> frame("ui.frame");
    pangolin::Slider frame_slider("frame", frame.Ref() );

    if(video_playback && TotalFrames() < std::numeric_limits<int>::max())
    {
        // frame_slider should be added first so that it can be rendered correctly.
        pangolin::DisplayBase().AddDisplay(frame_slider);
        frame_slider.SetBounds(0.0, pangolin::Attach::Pix(slider_size), 0.0, 1.0);
    }

    pangolin::View& container = pangolin::Display("streams");
    container.SetLayout(pangolin::LayoutEqual)
             .SetBounds(pangolin::Attach::Pix(slider_size), 1.0, 0.0, 1.0);

    std::vector<ImageView> stream_views(video.Streams().size());
    for(auto& sv : stream_views) {
        container.AddDisplay(sv);
    }

    pangolin::View& record_graphic = pangolin::Display("record_glyth").
            SetBounds(pangolin::Attach::Pix(-28),1.0f, pangolin::Attach::Pix(-28), 1.0f);
    record_graphic.extern_draw_function = [&](pangolin::View& v){
        if(video.IsRecording()) {
            v.ActivatePixelOrthographic();
            pangolin::glRecordGraphic(14.0f, 14.0f, 7.0f);
        }
    };

    std::vector<pangolin::Image<unsigned char> > images;

    /////////////////////////////////////////////////////////////////////////
    /// Register key shortcuts
    /////////////////////////////////////////////////////////////////////////

    RegisterDefaultKeyShortcutsAndPangoVariables();

    const char show_hide_keys[]  = {'1','2','3','4','5','6','7','8','9'};
    const char screenshot_keys[] = {'!','@','#','$','%','^','&','*','('};

    // Show/hide streams
    for(size_t v=0; v < container.NumChildren() && v < 9; v++) {
        pangolin::RegisterKeyPressCallback(show_hide_keys[v], [v,&container](){
            container[v].ToggleShow();
        } );
        pangolin::RegisterKeyPressCallback(screenshot_keys[v], [this,v,&images](){
            if(v < images.size() && images[v].ptr) {
                try{
                    pangolin::SaveImage(
                        images[v], video.Streams()[v].PixFormat(),
                        pangolin::MakeUniqueFilename("capture.png")
                    );
                }catch(const std::exception& e){
                    pango_print_error("Unable to save frame: %s\n", e.what());
                }
            }
        } );
    }

    video.Start();

    // Stream and display video
    while(should_run && !pangolin::ShouldQuit())
    {
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
        glColor3f(1.0f, 1.0f, 1.0f);

        {
            std::lock_guard<std::mutex> lock(control_mutex);

            if(frame.GuiChanged()) {
                if(video_playback) {
                    frame = video_playback->Seek(frame) -1;
                }
                grab_until = frame + 1;
            }

            if ( frame < grab_until && video.Grab(&buffer[0], images, video_grab_wait, video_grab_newest)) {
                frame = frame +1;

                if(frame_changed_callback) {
                    frame_changed_callback(buffer.get(), images, GetVideoFrameProperties(video_interface));
                }

                // Update images
                if((frame-1) % draw_nth_frame == 0) {
                    for(unsigned int i=0; i<images.size(); ++i) 
                        if(stream_views[i].IsShown()) {
                            stream_views[i].SetImage(images[i], pangolin::GlPixFormat(video.Streams()[i].PixFormat() ));
                        }
                }
            }
        }

        // leave in pixel orthographic for slider to render.
        pangolin::DisplayBase().ActivatePixelOrthographic();
        pangolin::FinishFrame();
    }

    pangolin::DestroyWindow(window_name);
}

void VideoViewer::RegisterDefaultKeyShortcutsAndPangoVariables()
{
    pangolin::RegisterKeyPressCallback(' ', [this](){TogglePlay();} );
    pangolin::RegisterKeyPressCallback('r', [this](){ToggleRecord();});
    pangolin::RegisterKeyPressCallback('w', [this](){ToggleWaitForFrames();} );
    pangolin::RegisterKeyPressCallback('d', [this](){ToggleDiscardBufferedFrames();} );
    pangolin::RegisterKeyPressCallback(',', [this](){Skip(-1);} );
    pangolin::RegisterKeyPressCallback('.', [this](){Skip(+1);} );
    pangolin::RegisterKeyPressCallback('<', [this](){Skip(-FRAME_SKIP);} );
    pangolin::RegisterKeyPressCallback('>', [this](){Skip(+FRAME_SKIP);} );
    pangolin::RegisterKeyPressCallback('0', [this](){RecordOneFrame();} );
    pangolin::RegisterKeyPressCallback('E', [this](){ChangeExposure(1000);} );
    pangolin::RegisterKeyPressCallback('e', [this](){ChangeExposure(-1000);} );
    pangolin::RegisterKeyPressCallback('G', [this](){ChangeGain(1);} );
    pangolin::RegisterKeyPressCallback('g', [this](){ChangeGain(-1);} );
    pangolin::RegisterKeyPressCallback('c', [this](){SetActiveCamera(+1);} );
}

void VideoViewer::OpenInput(const std::string& input_uri)
{
    std::lock_guard<std::mutex> lock(control_mutex);
    video.Open(input_uri, output_uri);

    // Output details of video stream
    for(size_t s = 0; s < video.Streams().size(); ++s) {
        const pangolin::StreamInfo& si = video.Streams()[s];
        std::cout << FormatString(
            "Stream %: % x % % (pitch: % bytes)",
            s, si.Width(), si.Height(), si.PixFormat().format, si.Pitch()
        ) << std::endl;
    }

    if(video.Streams().size() == 0) {
        pango_print_error("No video streams from device.\n");
        return;
    }

    video_playback = pangolin::FindFirstMatchingVideoInterface<pangolin::VideoPlaybackInterface>(video);
    video_interface = pangolin::FindFirstMatchingVideoInterface<pangolin::VideoInterface>(video);

    if(TotalFrames() < std::numeric_limits<int>::max() ) {
        std::cout << "Video length: " << TotalFrames() << " frames" << std::endl;
        grab_until = 0;
    }

    pangolin::Var<int> frame("ui.frame");
    frame.Meta().range[0] = 0;
    frame.Meta().range[1] = TotalFrames()-1;
}

void VideoViewer::CloseInput()
{
    std::lock_guard<std::mutex> lock(control_mutex);
    video.Close();
}

void VideoViewer::Record()
{
    std::lock_guard<std::mutex> lock(control_mutex);
    video.Record();
}

void VideoViewer::RecordOneFrame()
{
    std::lock_guard<std::mutex> lock(control_mutex);
    video.RecordOneFrame();
}

void VideoViewer::StopRecording()
{
    std::lock_guard<std::mutex> lock(control_mutex);
    if(video.IsRecording()) {
        video.Stop();
    }
}

void VideoViewer::TogglePlay()
{
    std::lock_guard<std::mutex> lock(control_mutex);
    grab_until = (current_frame < grab_until) ? current_frame: std::numeric_limits<int>::max();
}

void VideoViewer::ToggleRecord()
{
    std::lock_guard<std::mutex> lock(control_mutex);
    if(!video.IsRecording()) {
        video.SetTimelapse( static_cast<size_t>(record_nth_frame) );
        video.Record();
        pango_print_info("Started Recording.\n");
    }else{
        video.Stop();
        pango_print_info("Finished recording.\n");
    }
    fflush(stdout);
}

void VideoViewer::ToggleDiscardBufferedFrames()
{
    std::lock_guard<std::mutex> lock(control_mutex);
    video_grab_newest = !video_grab_newest;
    if(video_grab_newest) {
        pango_print_info("Discarding old frames.\n");
    }else{
        pango_print_info("Not discarding old frames.\n");
    }
}

void VideoViewer::ToggleWaitForFrames()
{
    std::lock_guard<std::mutex> lock(control_mutex);
    video_grab_wait = !video_grab_wait;
    if(video_grab_wait) {
        pango_print_info("Gui wait's for video frame.\n");
    }else{
        pango_print_info("Gui doesn't wait for video frame.\n");
    }
}

void VideoViewer::SetDiscardBufferedFrames(bool new_state)
{
    std::lock_guard<std::mutex> lock(control_mutex);
    video_grab_newest = new_state;
    if(video_grab_newest) {
        pango_print_info("Discarding old frames.\n");
    }else{
        pango_print_info("Not discarding old frames.\n");
    }
}


void VideoViewer::DrawEveryNFrames(int n)
{
    if(n <= 0) {
        pango_print_warn("Cannot draw every %d frames. Ignoring request.\n",n);
        return;
    }

    if(n != draw_nth_frame && n == 1)
        pango_print_info("Drawing every frame.\n");
    if(n != draw_nth_frame && n > 1)
        pango_print_info("Drawing one in every %d frames.\n",n);

    draw_nth_frame=n;
}



void VideoViewer::SetWaitForFrames(bool new_state)
{
    std::lock_guard<std::mutex> lock(control_mutex);
    video_grab_wait = new_state;
    if(video_grab_wait) {
        pango_print_info("Gui wait's for video frame.\n");
    }else{
        pango_print_info("Gui doesn't wait for video frame.\n");
    }
}

void VideoViewer::Skip(int frames)
{
    std::lock_guard<std::mutex> lock(control_mutex);

    if(video_playback) {
        const int next_frame = current_frame + frames;
        if (next_frame >= 0) {
            current_frame = video_playback->Seek(next_frame) -1;
            grab_until = current_frame + 1;
        } 
    }else{
        if(frames >= 0) {
            grab_until = current_frame + frames;
        }else{
            pango_print_warn("Unable to skip backward.");
        }
    }

}

bool VideoViewer::ChangeExposure(int delta_us)
{
    std::lock_guard<std::mutex> lock(control_mutex);

    std::vector<pangolin::GenicamVideoInterface*> ifs = FindMatchingVideoInterfaces<pangolin::GenicamVideoInterface>(video);
    std::string exposure_time;
    if (ifs[active_cam]->GetParameter("ExposureTime",exposure_time))
    {
            return false;
    }

    int exp = atoi(exposure_time.c_str());

    return ifs[active_cam]->SetParameter("ExposureTime", std::to_string(exp+delta_us));
}

bool VideoViewer::ChangeGain(float delta)
{
    std::lock_guard<std::mutex> lock(control_mutex);

    std::vector<pangolin::GenicamVideoInterface*> ifs = FindMatchingVideoInterfaces<pangolin::GenicamVideoInterface>(video);

    std::string gain_string;
    if (!ifs[active_cam]->GetParameter("Gain", gain_string))
    {
        return false;
    }
    double gain = atoi(gain_string.c_str());

    return ifs[active_cam]->SetParameter("Gain", std::to_string(gain+delta));
}


void VideoViewer::SetActiveCamera(int delta)
{
    std::lock_guard<std::mutex> lock(control_mutex);

    std::vector<pangolin::GenicamVideoInterface*> ifs = FindMatchingVideoInterfaces<pangolin::GenicamVideoInterface>(video);

    active_cam += delta;

    if(active_cam >= ifs.size()) {active_cam = 0;}
}


void VideoViewer::SetFrameChangedCallback(FrameChangedCallbackFn cb)
{
    std::lock_guard<std::mutex> lock(control_mutex);
    frame_changed_callback=cb;
}



void VideoViewer::WaitUntilExit()
{
    if(vv_thread.joinable()) {
        vv_thread.join();
    }
}


void RunVideoViewerUI(const std::string& input_uri, const std::string& output_uri)
{
    RegisterNewSigCallback(videoviewer_signal_quit, nullptr, SIGINT);
    RegisterNewSigCallback(videoviewer_signal_quit, nullptr, SIGTERM);

    VideoViewer vv("VideoViewer", input_uri, output_uri);
    vv.Run();
}

}
