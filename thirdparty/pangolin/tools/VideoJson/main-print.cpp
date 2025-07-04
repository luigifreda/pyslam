#include <pangolin/pangolin.h>

#include <pangolin/log/packetstream_reader.h>
#include <pangolin/log/packetstream_writer.h>
#include <pangolin/video/video.h>

int main( int argc, char* argv[] )
{
    if( argc == 2) {
        const std::string filename = std::string(argv[1]);
        pangolin::PacketStreamReader reader(filename);

        // Extract JSON
        picojson::value all_properties;

        for(size_t i=0; i < reader.Sources().size(); ++i) {
            picojson::value source_props;

            const pangolin::PacketStreamSource& src = reader.Sources()[i];
            source_props["device_properties"] = src.info["device"];

            // Seek through index, loading frame properties
            for(size_t framenum=0; framenum < src.index.size(); ++framenum) {
                reader.Seek(src.id, framenum);
                pangolin::Packet pkt = reader.NextFrame();
                source_props["frame_properties"].push_back(pkt.meta);
            }

            all_properties.push_back(source_props);
        }

        std::cout << all_properties.serialize(true) << std::endl;
    }else{
        std::cout << "Usage: \n\tPangoJsonPrint filename.pango" << std::endl;
    }
}
