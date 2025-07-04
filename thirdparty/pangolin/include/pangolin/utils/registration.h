#pragma once

#include <functional>

namespace pangolin {

template<typename TokenType=void>
class Registration
{
public:
    using UnregisterFunc = std::function<void(const TokenType&)>;

    Registration()
        : token()
    {
    }

    Registration(TokenType token, UnregisterFunc unregister)
        : token(token), unregister(unregister)
    {

    }

    // No copy constructor
    Registration(const Registration&) = delete;

    // Default move constructor
    Registration(Registration&& o)
    {
        *this = std::move(o);
    }

    Registration<TokenType>& operator =(Registration<TokenType>&& o)
    {
        token = o.token;
        unregister = std::move(o.unregister);
        o.unregister = nullptr;
        return *this;
    }

    ~Registration()
    {
        Release();
    }

    void Release()
    {
        if(unregister) {
            unregister(token);
            token = TokenType();
        }
    }

    const TokenType& Token()
    {
        return token;
    }

private:
    TokenType token;
    UnregisterFunc unregister;
};

}
