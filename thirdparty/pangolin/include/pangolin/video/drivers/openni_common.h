/* This file is part of the Pangolin Project.
 * http://github.com/stevenlovegrove/Pangolin
 *
 * Copyright (c) 2014 Steven Lovegrove
 *               2015 Richard Newcombe
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

#include <pangolin/video/iostream_operators.h>

namespace pangolin
{

enum OpenNiSensorType
{
    OpenNiUnassigned = -1,
    OpenNiRgb = 0,
    OpenNiIr,
    OpenNiDepth_1mm,
    OpenNiDepth_1mm_Registered,
    OpenNiDepth_100um,
    OpenNiIr8bit,
    OpenNiIr24bit,
    OpenNiIrProj,
    OpenNiIr8bitProj,
    OpenNiGrey
};

struct PANGOLIN_EXPORT OpenNiStreamMode
{
    OpenNiStreamMode(
        OpenNiSensorType sensor_type=OpenNiUnassigned,
        ImageDim dim=ImageDim(640,480), ImageRoi roi=ImageRoi(0,0,0,0), int fps=30, int device=0
    )
        : sensor_type(sensor_type), dim(dim), roi(roi), fps(fps), device(device)
    {

    }

    OpenNiSensorType sensor_type;
    ImageDim dim;
    ImageRoi roi;
    int fps;
    int device;
};

inline OpenNiSensorType openni_sensor(const std::string& str)
{
    if( !str.compare("grey") || !str.compare("gray") ) {
        return OpenNiGrey;
    }else if( !str.compare("rgb") ) {
        return OpenNiRgb;
    }else if( !str.compare("ir") ) {
        return OpenNiIr;
    }else if( !str.compare("depth1mm") || !str.compare("depth") ) {
        return OpenNiDepth_1mm;
    }else if( !str.compare("depth100um") ) {
        return OpenNiDepth_100um;
    }else if( !str.compare("depth_reg") || !str.compare("reg_depth")) {
        return OpenNiDepth_1mm_Registered;
    }else if( !str.compare("ir8") ) {
        return OpenNiIr8bit;
    }else if( !str.compare("ir24") ) {
        return OpenNiIr24bit;
    }else if( !str.compare("ir+") ) {
        return OpenNiIrProj;
    }else if( !str.compare("ir8+") ) {
        return OpenNiIr8bitProj;
    }else if( str.empty() ) {
        return OpenNiUnassigned;
    }else{
        throw pangolin::VideoException("Unknown OpenNi sensor", str );
    }
}

// Find prefix character key
// Given arguments "depth!5:320x240@15", "!:@", would return map
// \0->"depth", !->"5", :->"320x240", @->"15"
inline std::map<char,std::string> GetTokenSplits(const std::string& str, const std::string& tokens)
{
    std::map<char,std::string> splits;

    char last_token = 0;
    size_t  last_start = 0;
    for(unsigned int i=0; i<str.size(); ++i) {
        size_t token_pos = tokens.find(str[i]);
        if(token_pos != std::string::npos) {
            splits[last_token] = str.substr(last_start, i-last_start);
            last_token = tokens[token_pos];
            last_start = i+1;
        }
    }

    if(last_start < str.size()) {
        splits[last_token] = str.substr(last_start);
    }

    return splits;
}

inline std::istream& operator>> (std::istream &is, OpenNiStreamMode& fmt)
{
    std::string str;
    is >> str;

    std::map<char,std::string> splits = GetTokenSplits(str, "!:@#");

    if(splits.count(0)) {
        fmt.sensor_type = openni_sensor(splits[0]);
    }

    if(splits.count('@')) {
        fmt.fps = pangolin::Convert<int,std::string>::Do(splits['@']);
    }

    if(splits.count(':')) {
        fmt.dim = pangolin::Convert<ImageDim,std::string>::Do(splits[':']);
    }

    if(splits.count('!')) {
        fmt.device = pangolin::Convert<int,std::string>::Do(splits['!']);
    }

    if(splits.count('#')) {
        fmt.roi = pangolin::Convert<ImageRoi,std::string>::Do(splits['#']);
    }

    return is;
}

}
