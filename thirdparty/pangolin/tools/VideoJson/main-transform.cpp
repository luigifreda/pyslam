#include <pangolin/pangolin.h>

#include <pangolin/log/packetstream_reader.h>
#include <pangolin/log/packetstream_writer.h>
#include <pangolin/video/video.h>

int main( int argc, char* argv[] )
{
    if( argc == 3) {
        const std::string filein = std::string(argv[1]);
        const std::string fileout = std::string(argv[2]);
        PANGO_ASSERT(filein != fileout);

        pangolin::PacketStreamReader reader(filein);
        pangolin::PacketStreamWriter writer(fileout);

        // Read new JSON from command line
        picojson::value all_properties;

        std::cout << "Reading JSON from input" << std::endl;
        std::cin >> all_properties;
        std::cout << "+ done" << std::endl;

        PANGO_ASSERT(all_properties.size() == reader.Sources().size());

        for(size_t i=0; i < reader.Sources().size(); ++i) {
            const picojson::value& src_json = all_properties[i];
            pangolin::PacketStreamSource src = reader.Sources()[i];

            PANGO_ASSERT(src_json.contains("frame_properties"));
            PANGO_ASSERT(src_json.contains("device_properties"));
            PANGO_ASSERT(src_json["frame_properties"].size() == src.index.size());

            // Ensure the index gets rewritten, and update the device properties.
            src.index.clear();
            src.info["device"] = src_json["device_properties"];
            writer.AddSource(src);
        }

        std::cout << "Writing video with new JSON to '" << fileout << "'" << std::endl;
        try{
            std::vector<char> buffer;

            while(true)
            {
                pangolin::Packet pkt = reader.NextFrame();
                buffer.resize(pkt.BytesRemaining());
                pkt.Stream().read(buffer.data(), buffer.size());

                const picojson::value& new_frame_json = all_properties[pkt.src]["frame_properties"][pkt.sequence_num];
                writer.WriteSourcePacket(pkt.src, buffer.data(), pkt.time, buffer.size(), new_frame_json);
                std::cout << "Frames complete: " << pkt.sequence_num << " / " << reader.Sources()[pkt.src].index.size() << '\r';
                std::cout.flush();
            }
        }catch(const std::runtime_error &)
        {
        }
        std::cout << std::endl << "+ done" << std::endl;

    }else{
        std::cout << "Usage: \n\tPangoJsonTransform file_in.pango file_out.pango" << std::endl;
    }
}
