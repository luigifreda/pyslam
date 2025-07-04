/* This file is part of the Pangolin Project.
 * http://github.com/stevenlovegrove/Pangolin
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <fstream>

#include <pangolin/platform.h>

#include <pangolin/log/packetstream_tags.h>
#include <pangolin/utils/file_utils.h>

namespace pangolin
{

class PacketStream: public std::ifstream
{
public:
    PacketStream()
        : _is_pipe(false)
    {
        cclear();
    }

    PacketStream(const std::string& filename)
        : Base(filename.c_str(), std::ios::in | std::ios::binary),
          _is_pipe(IsPipe(filename))
    {
        cclear();
    }

    bool seekable() const
    {
        return is_open() && !_is_pipe;
    }

    void open(const std::string& filename)
    {
        close();
        _is_pipe = IsPipe(filename);
        Base::open(filename.c_str(), std::ios::in | std::ios::binary);
    }

    void close()
    {
        cclear();
        if (Base::is_open()) Base::close();
    }

    void seekg(std::streampos target);

    void seekg(std::streamoff off, std::ios_base::seekdir way);

    std::streampos tellg();

    size_t read(char* target, size_t len);

    char get();

    size_t skip(size_t len);

    size_t readUINT();

    int64_t readTimestamp();

    pangoTagType peekTag();

    pangoTagType readTag();

    pangoTagType readTag(pangoTagType);

    pangoTagType syncToTag();

private:
    using Base = std::ifstream;

    bool _is_pipe;
    pangoTagType _tag;

    // Amount of frame data left to read. Tracks our position within a data block.


    void cclear() {
        _tag = 0;
    }
};


}
