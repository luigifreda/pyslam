#pragma once

#include <functional>
#include <map>

#include <pangolin/utils/registration.h>

namespace pangolin {

// Based on http://simmesimme.github.io/tutorials/2015/09/20/signal-slot
template <typename... Args>
class Signal
{
public:
    using Id = size_t;
    using Reg = Registration<Id>;

    Signal()
        : current_id_(0)
    {
    }

    Signal(const Signal&) = delete;

    Signal(Signal&&) = default;

    // connects a std::function to the signal. The returned
    // value can be used to disconnect the function again
    Reg Connect(const std::function<void(Args...)>& slot) const {
        slots_.insert(std::make_pair(++current_id_, slot));
        return Reg(current_id_, [this](Id id){ Disconnect(id);});
    }

    // disconnects a previously connected function
    void Disconnect(Id id) const {
        slots_.erase(id);
    }

    // disconnects all previously connected functions
    void DisconnectAll() const {
        slots_.clear();
    }

    // calls all connected functions
    void operator()(Args... p) {
        for(auto it : slots_) {
            it.second(p...);
        }
    }

private:
    mutable std::map<Id, std::function<void(Args...)>> slots_;
    mutable Id current_id_;
};

}
