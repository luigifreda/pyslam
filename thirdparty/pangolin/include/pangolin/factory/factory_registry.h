/* This file is part of the Pangolin Project.
 * http://github.com/stevenlovegrove/Pangolin
 *
 * Copyright (c) 2011-2013 Steven Lovegrove
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

#include <memory>
#include <vector>
#include <algorithm>

#include <pangolin/utils/uri.h>

namespace pangolin
{

template<typename T>
struct FactoryInterface
{
    typedef T FactoryItem;

    virtual ~FactoryInterface() = default;
    virtual std::unique_ptr<T> Open(const Uri& uri) = 0;
};

template<typename T>
class FactoryRegistry
{
public:
    // IMPORTANT: Implement for each templated instantiation within a seperate compilation unit.
    static FactoryRegistry<T>& I();

    ~FactoryRegistry()
    {
    }

    void RegisterFactory(std::shared_ptr<FactoryInterface<T>> factory, uint32_t precedence, const std::string& scheme_name )
    {
        FactoryItem item = {precedence, scheme_name, factory};
        factories.push_back( item );
        std::sort(factories.begin(), factories.end());
    }

    void UnregisterFactory(FactoryInterface<T>* factory)
    {
        for( auto i = factories.end()-1; i != factories.begin(); --i)
        {
            if( i->factory.get() == factory ) {
                factories.erase(i);
            }
        }
    }

    void UnregisterAllFactories()
    {
        factories.clear();
    }

    std::unique_ptr<T> Open(const Uri& uri)
    {
        // Iterate over all registered factories in order of precedence.
        for(auto& item : factories) {
            if( item.scheme == uri.scheme) {
                std::unique_ptr<T> video = item.factory->Open(uri);
                if(video) {
                    return video;
                }
            }
        }

        return std::unique_ptr<T>();
    }

private:
    struct FactoryItem
    {
        uint32_t precedence;
        std::string scheme;
        std::shared_ptr<FactoryInterface<T>> factory;

        bool operator<(const FactoryItem& rhs) const {
            return precedence < rhs.precedence;
        }
    };

    // Priority, Factory tuple
    std::vector<FactoryItem> factories;
};

#define PANGOLIN_REGISTER_FACTORY(x) void Register ## x ## Factory()

}
