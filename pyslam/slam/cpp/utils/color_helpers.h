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

#include <array>
#include <cstddef>

#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <tuple>

namespace pyslam {

namespace utils {

template <typename T> constexpr bool is_power_of_2(const T value) {
    return (value > 0) && ((value & (value - 1)) == 0);
}

} // namespace utils

class Colors {
  public:
    static constexpr int kMyjetNumColors = 10;
    static constexpr std::array<std::array<float, 3>, kMyjetNumColors> myjet = {
        {{{0.0, 0.0, 0.5}},
         {{0.0, 0.0, 0.99910873}},
         {{0.0, 0.37843137, 1.0}},
         {{0.0, 0.83333333, 1.0}},
         {{0.30044276, 1.0, 0.66729918}},
         {{0.66729918, 1.0, 0.30044276}},
         {{1.0, 0.90123457, 0.0}},
         {{1.0, 0.48002905, 0.0}},
         {{0.99910873, 0.07334786, 0.0}},
         {{0.5, 0.0, 0.0}}}};

    static constexpr std::array<std::array<float, 3>, kMyjetNumColors> my_jet_x_255 = {
        {{{0.0f * 255.0f, 0.0f * 255.0f, 0.5f * 255.0f}},
         {{0.0f * 255.0f, 0.0f * 255.0f, 0.99910873f * 255.0f}},
         {{0.0f * 255.0f, 0.37843137f * 255.0f, 1.0f * 255.0f}},
         {{0.0f * 255.0f, 0.83333333f * 255.0f, 1.0f * 255.0f}},
         {{0.30044276f * 255.0f, 1.0f * 255.0f, 0.66729918f * 255.0f}},
         {{0.66729918f * 255.0f, 1.0f * 255.0f, 0.30044276f * 255.0f}},
         {{1.0f * 255.0f, 0.90123457f * 255.0f, 0.0f * 255.0f}},
         {{1.0f * 255.0f, 0.48002905f * 255.0f, 0.0f * 255.0f}},
         {{0.99910873f * 255.0f, 0.07334786f * 255.0f, 0.0f * 255.0f}},
         {{0.5f * 255.0f, 0.0f * 255.0f, 0.0f * 255.0f}}}};

    static std::array<float, 3> myjet_color(int idx) { return myjet[idx % kMyjetNumColors]; }
    static std::array<float, 3> myjet_color_x_255(int idx) {
        return my_jet_x_255[idx % kMyjetNumColors];
    }
}; // namespace pyslam

class ColorTableGenerator {
  public:
    struct RGB {
        uint8_t r, g, b;
    };

    // Customize table size (keep 64..4096 reasonable)
    static constexpr size_t TABLE_SIZE = 256;            // must be a power of 2
    static constexpr size_t TABLE_MASK = TABLE_SIZE - 1; // For bitwise AND instead of modulo

    static constexpr bool USE_HASH_DISTRIBUTION = false; // set false to use raw id % TABLE_SIZE

    // Singleton accessor
    static const ColorTableGenerator &instance() {
        static const ColorTableGenerator inst;
        return inst;
    }

    // Integer -> color (fast: O(1))
    RGB color_from_int(uint64_t id) const {
        if constexpr (USE_HASH_DISTRIBUTION) {
            uint64_t key = splitmix64(id);
            size_t idx = static_cast<size_t>(key & TABLE_MASK); // Bitwise AND instead of modulo
            return table_[idx];
        } else {
            size_t idx = static_cast<size_t>(id & TABLE_MASK); // Bitwise AND instead of modulo
            return table_[idx];
        }
    }

    // Utility: return hex string like "#RRGGBB"
    std::string to_hex(RGB c) const {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "#%02X%02X%02X", c.r, c.g, c.b);
        return std::string(buf);
    }

    // Utility: pack to 0xRRGGBB
    static uint32_t pack_rgb(RGB c) {
        return (uint32_t(c.r) << 16) | (uint32_t(c.g) << 8) | uint32_t(c.b);
    }

  public:
    // --- mix IDs so nearby integers spread across the table ---
    static uint64_t splitmix64(uint64_t x) {
        // Excellent 64-bit mixer; cheap and reversible enough for our use.
        x += 0x9E3779B97F4A7C15ull;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
        return x ^ (x >> 31);
    }

  private:
    // --- Table generation: well-separated hues via golden-ratio stepping ---
    static std::array<RGB, TABLE_SIZE> generate_table() {
        std::array<RGB, TABLE_SIZE> t{};
        const float PHI = 0.61803398875f; // golden ratio conjugate
        float h = 0.0f;
        for (size_t i = 0; i < TABLE_SIZE; ++i) {
            h = std::fmod(h + PHI, 1.0f);                 // low-discrepancy hues
            auto [r, g, b] = hsv_to_rgb(h, 0.70f, 0.90f); // vivid, readable colors
            t[i] = {r, g, b};
        }
        return t;
    }

    // --- HSV -> RGB helper (returns 8-bit components) ---
    static std::tuple<uint8_t, uint8_t, uint8_t> hsv_to_rgb(float h, float s, float v) {
        float c = v * s;
        float hp = h * 6.0f;
        float x = c * (1.0f - std::fabs(std::fmod(hp, 2.0f) - 1.0f));
        float m = v - c;

        float r = 0, g = 0, b = 0;
        int i = static_cast<int>(hp);
        switch (i % 6) {
        case 0:
            r = c;
            g = x;
            b = 0;
            break;
        case 1:
            r = x;
            g = c;
            b = 0;
            break;
        case 2:
            r = 0;
            g = c;
            b = x;
            break;
        case 3:
            r = 0;
            g = x;
            b = c;
            break;
        case 4:
            r = x;
            g = 0;
            b = c;
            break;
        case 5:
            r = c;
            g = 0;
            b = x;
            break;
        }
        auto to8 = [](float u) { return static_cast<uint8_t>(std::lround((u) * 255.0f)); };
        return {to8(r + m), to8(g + m), to8(b + m)};
    }

    // --- Singleton setup ---
    ColorTableGenerator() : table_(generate_table()) {
        // check if the table is a power of 2
        static_assert(utils::is_power_of_2<size_t>(TABLE_SIZE), "TABLE_SIZE must be a power of 2");
    }
    ColorTableGenerator(const ColorTableGenerator &) = delete;
    ColorTableGenerator &operator=(const ColorTableGenerator &) = delete;

  private:
    const std::array<RGB, TABLE_SIZE> table_;
};

} // namespace pyslam