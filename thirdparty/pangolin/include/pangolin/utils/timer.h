/* This file is part of the Pangolin Project.
 * http://github.com/stevenlovegrove/Pangolin
 *
 * Copyright (c) 2011 Steven Lovegrove
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

#include <chrono>
#include <thread>

#include <pangolin/platform.h>

namespace pangolin
{

// These methods exist for backwards compatibility.
// They are deprecated in favour of direct use of std::chrono in C++11

using baseclock = std::chrono::steady_clock;
using basetime = baseclock::time_point;
static_assert(baseclock::is_steady, "baseclock must be steady to be robust against system time settings");

inline basetime TimeNow()
{
    return baseclock::now();
}

inline double Time_s(basetime t)
{
    using namespace std::chrono;
    return duration_cast<seconds>( t.time_since_epoch() ).count();
}

inline int64_t Time_us(basetime t)
{
    using namespace std::chrono;
    return duration_cast<microseconds>( t.time_since_epoch() ).count();
}

inline double TimeDiff_s(basetime start, basetime end)
{
    const baseclock::duration d = end - start;
    return Time_s( basetime() + d);
}

inline int64_t TimeDiff_us(basetime start, basetime end)
{
    const baseclock::duration d = end - start;
    return Time_us( basetime() + d);
}

inline basetime TimeAdd(basetime t1, basetime t2)
{

    return t1 + t2.time_since_epoch();
}

inline double TimeNow_s()
{
    return Time_s(TimeNow());
}

inline int64_t TimeNow_us()
{
    return Time_us(TimeNow());
}

inline basetime WaitUntil(basetime t)
{
    std::this_thread::sleep_until(t);
    return TimeNow();
}

struct Timer
{
    Timer() {
        Reset();
    }
    
    void Reset()
    {
        start = TimeNow();
    }
    
    double Elapsed_s()
    {
        basetime currtime = TimeNow();
        return TimeDiff_s(start,currtime);
    }
    
    basetime start;
};

}
