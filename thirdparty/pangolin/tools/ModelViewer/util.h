#pragma once

template<typename R>
bool is_ready(std::future<R> const& f)
{ return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready; }

inline std::vector<std::string> ExpandGlobOption(const argagg::option_results& opt)
{
    std::vector<std::string> expanded;
    for(const auto& o : opt.all)
    {
        const std::string r = o.as<std::string>();
        pangolin::FilesMatchingWildcard(r, expanded);
    }
    return expanded;
}

template<typename Tout, typename Tin, typename F>
inline std::vector<Tout> TryLoad(const std::vector<Tin>& in, const F& load_func)
{
    std::vector<Tout> loaded;
    for(const Tin& file : in)
    {
        try {
            loaded.emplace_back(load_func(file));
        }catch(const std::exception&) {
        }
    }
    return loaded;
}
