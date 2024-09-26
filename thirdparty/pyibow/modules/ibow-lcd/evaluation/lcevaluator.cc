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

#include "lcevaluator.h"

namespace ibow_lcd {

LCEvaluator::LCEvaluator() {}

void LCEvaluator::detectLoops(
      const std::vector<unsigned>& image_ids,
      const std::vector<std::vector<cv::KeyPoint> >& kps,
      const std::vector<cv::Mat>& descs,
      std::vector<LCDetectorResult>* results) {
  results->clear();
  unsigned nimages = image_ids.size();

  // Creating the loop closure detector object
  ibow_lcd::LCDetector lcdet(index_params_);

  // Processing the sequence of images
  for (unsigned i = 0; i < nimages; i++) {
    // Processing image i
    // std::cout << "--- Processing image " << i;

    ibow_lcd::LCDetectorResult result;
    lcdet.process(image_ids[i], kps[i], descs[i], &result);

    // if (result.status == LC_DETECTED) {
    //   std::cout << " --- Loop detected!!!: " << result.train_id;
    // }

    results->push_back(result);
    // std::cout << std::endl;
  }
}

void LCEvaluator::detectLoops(
    const std::vector<unsigned>& image_ids,
    const std::vector<std::vector<cv::KeyPoint> >& kps,
    const std::vector<cv::Mat>& descs,
    std::ofstream& out_file) {
  unsigned nimages = image_ids.size();

  // Creating the loop closure detector object
  ibow_lcd::LCDetector lcdet(index_params_);

  // Processing the sequence of images
  for (unsigned i = 0; i < nimages; i++) {
    lcdet.debug(image_ids[i], kps[i], descs[i], out_file);
  }
}

}  // namespace ibow_lcd
