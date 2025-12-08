/*
 * This file is part of PYSLAM
 *
 * Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
 *
 * PYSLAM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PYSLAM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
 */
#pragma once

#include <algorithm>
#include <cstdint>
#include <string>

namespace pyslam {

// ===============================
// Base64 codec
// ===============================

// Small Base64 encoder (no line breaks)
inline std::string base64_encode(const unsigned char *data, size_t len) {
    static const char tbl[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(4 * ((len + 2) / 3));
    size_t i = 0;
    while (i + 3 <= len) {
        uint32_t v =
            (uint32_t(data[i]) << 16) | (uint32_t(data[i + 1]) << 8) | uint32_t(data[i + 2]);
        i += 3;
        out.push_back(tbl[(v >> 18) & 0x3F]);
        out.push_back(tbl[(v >> 12) & 0x3F]);
        out.push_back(tbl[(v >> 6) & 0x3F]);
        out.push_back(tbl[v & 0x3F]);
    }
    if (i < len) {
        uint32_t v = uint32_t(data[i]) << 16;
        if (i + 1 < len)
            v |= uint32_t(data[i + 1]) << 8;
        out.push_back(tbl[(v >> 18) & 0x3F]);
        out.push_back(tbl[(v >> 12) & 0x3F]);
        if (i + 1 < len) {
            out.push_back(tbl[(v >> 6) & 0x3F]);
            out.push_back('=');
        } else {
            out.push_back('=');
            out.push_back('=');
        }
    }
    return out;
}

// Base64 decoder (no line breaks)
inline std::string base64_decode(const std::string &encoded) {
    static const char tbl[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    static int rev_tbl[256];
    static bool initialized = false;

    if (!initialized) {
        std::fill(rev_tbl, rev_tbl + 256, -1);
        for (int i = 0; i < 64; ++i) {
            rev_tbl[static_cast<unsigned char>(tbl[i])] = i;
        }
        initialized = true;
    }

    std::string decoded;
    decoded.reserve(3 * encoded.length() / 4);

    size_t i = 0;
    while (i + 4 <= encoded.size()) {
        unsigned char c0 = encoded[i];
        unsigned char c1 = encoded[i + 1];
        unsigned char c2 = encoded[i + 2];
        unsigned char c3 = encoded[i + 3];

        if (c0 == '=' || c1 == '=')
            break;

        int a = rev_tbl[c0];
        int b = rev_tbl[c1];

        if (c3 == '=' && c2 == '=') {
            // xx== -> 1 byte
            if (a >= 0 && b >= 0) {
                uint32_t v = (a << 18) | (b << 12);
                decoded.push_back(static_cast<char>((v >> 16) & 0xFF));
            }
            i += 4;
            break; // last block
        } else if (c3 == '=') {
            // xxx= -> 2 bytes
            int c = rev_tbl[c2];
            if (a >= 0 && b >= 0 && c >= 0) {
                uint32_t v = (a << 18) | (b << 12) | (c << 6);
                decoded.push_back(static_cast<char>((v >> 16) & 0xFF));
                decoded.push_back(static_cast<char>((v >> 8) & 0xFF));
            }
            i += 4;
            break; // last block
        } else {
            // xxxx -> 3 bytes
            int c = rev_tbl[c2];
            int d = rev_tbl[c3];
            if (a >= 0 && b >= 0 && c >= 0 && d >= 0) {
                uint32_t v = (a << 18) | (b << 12) | (c << 6) | d;
                decoded.push_back(static_cast<char>((v >> 16) & 0xFF));
                decoded.push_back(static_cast<char>((v >> 8) & 0xFF));
                decoded.push_back(static_cast<char>(v & 0xFF));
            }
            i += 4;
        }
    }

    return decoded;
}

} // namespace pyslam