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

#ifndef EVALUATION_LCEVALUATOR_H_
#define EVALUATION_LCEVALUATOR_H_

#include <iostream>
#include <fstream>
#include <vector>

#include <boost/filesystem.hpp>
#include <opencv2/features2d.hpp>

#include "ibow-lcd/lcdetector.h"

namespace ibow_lcd {

class LCEvaluator {
 public:
  // Constructor
  LCEvaluator();

  // Methods
  void detectLoops(
    const std::vector<unsigned>& image_ids,
    const std::vector<std::vector<cv::KeyPoint> >& kps,
    const std::vector<cv::Mat>& descs,
    std::vector<LCDetectorResult>* results);
  void detectLoops(
    const std::vector<unsigned>& image_ids,
    const std::vector<std::vector<cv::KeyPoint> >& kps,
    const std::vector<cv::Mat>& descs,
    std::ofstream& out_file);

  inline void setIndexParams(const LCDetectorParams& params) {
    index_params_ = params;
  }

 private:
  LCDetectorParams index_params_;
};

}  // namespace ibow_lcd

#endif  // EVALUATION_LCEVALUATOR_H_
