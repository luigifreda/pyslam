/**
* This file is part of ibow-lcd.
*
* Copyright (C) 2017 Emilio Garcia-Fidalgo <emilio.garcia@uib.es> (University of the Balearic Islands)
*
* ibow-lcd is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ibow-lcd is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ibow-lcd. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef INCLUDE_IBOW_LCD_ISLAND_H_
#define INCLUDE_IBOW_LCD_ISLAND_H_

#include <string>
#include <sstream>

#include "obindex2/BoostArchiver.h"

namespace ibow_lcd {

// Island
struct Island {
  explicit Island(unsigned image_id,
                  double sc,
                  unsigned min_img,
                  unsigned max_img) :
        min_img_id(min_img),
        max_img_id(max_img),
        img_id(image_id),
        score(sc) {}

  unsigned size() {
    return max_img_id - min_img_id + 1;
  }

  bool fits(unsigned image_id) {
    bool response = false;
    if (image_id >= min_img_id && image_id <= max_img_id) {
      response = true;
    }
    return response;
  }

  bool overlaps(const Island& island) const {
    unsigned a1 = min_img_id;
    unsigned a2 = max_img_id;
    unsigned b1 = island.min_img_id;
    unsigned b2 = island.max_img_id;

    return (b1 <= a1 && a1 <= b2) || (a1 <= b1 && b1 <= a2);
  }

  void adjustLimits(const unsigned image_id, unsigned* min, unsigned* max) {
    // If the image is to the right of the island
    if (image_id > max_img_id) {
      if (*min <= max_img_id) {
        *min = max_img_id + 1;
      }
    } else {
      // Otherwise, the image is to the left of the island
      if (*max >= min_img_id) {
        *max = min_img_id - 1;
      }
    }
  }

  void incrementScore(double sc) {
    score += sc;
  }

  void normalizeScore() {
    score = score / size();
  }

  std::string toString() const {
    std::stringstream ss;
    ss << "[" << min_img_id << " - " << max_img_id << "] Score: " << score
       << " | Img Id: " << img_id << std::endl;
    return ss.str();
  }

  bool operator<(const Island& island) const { return score > island.score; }

  unsigned min_img_id;
  unsigned max_img_id;
  unsigned img_id;
  double score;

protected: 
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & min_img_id;
    ar & max_img_id;
    ar & img_id;
    ar & score;
  }
};

}  // namespace ibow_lcd

#endif  // INCLUDE_IBOW_LCD_ISLAND_H_
