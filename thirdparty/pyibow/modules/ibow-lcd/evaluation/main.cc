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

#include <fstream>
#include <iostream>

#include <boost/filesystem.hpp>
#include <opencv2/features2d.hpp>

#include "lcevaluator.h"
#include "json.hpp"

using json = nlohmann::json;

void getFilenames(const std::string& directory,
                  std::vector<std::string>* filenames) {
    using namespace boost::filesystem;

    filenames->clear();
    path dir(directory);

    // Retrieving, sorting and filtering filenames.
    std::vector<path> entries;
    copy(directory_iterator(dir), directory_iterator(), back_inserter(entries));
    sort(entries.begin(), entries.end());
    for (auto it = entries.begin(); it != entries.end(); it++) {
        std::string ext = it->extension().c_str();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".png" || ext == ".jpg" ||
            ext == ".ppm" || ext == ".jpeg") {
            filenames->push_back(it->string());
        }
    }
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cout << "Incorrect usage. Please, call the program indicating only a ";
    std::cout << "configuration file as a parameter." << std::endl;
    return 0;
  }

  // Reading the indicated JSON configuration
  std::cout << "Parsing configuration file ..." << std::endl;
  std::ifstream config_file(argv[1]);
  json js;
  config_file >> js;

  // Parsing parameters
  std::string config_name = js["config_name"];
  std::cout << "Configuration name: " << config_name << std::endl;

  std::string base_dir = js["base_dir"];
  std::cout << "Base directory: " << base_dir << std::endl;

  std::string results_dir = js["results_dir"];
  std::cout << "Results directory: " << results_dir << std::endl;

  bool debug = js["debug"];
  std::cout << "Debug: " << std::boolalpha << debug << std::endl;

  // Preparing working directory
  std::cout << "Preparing working directory ..." << std::endl;
  boost::filesystem::path res_dir = results_dir + config_name;
  boost::filesystem::remove_all(res_dir);
  boost::filesystem::create_directory(res_dir);
  std::cout << "Working directory ready" << std::endl;

  // Storing information about the dataset
  std::string info_filename = results_dir + config_name + "/info.json";
  std::ofstream info_file(info_filename);
  json info_json;
  info_json["img_dir"] = base_dir + "images/";
  info_json["gt_file"] = base_dir + "groundtruth.mat";
  info_json["coords_file"] = base_dir + "imageCoords.mat";

  // Optional parameters
  if (debug) {
    info_json["p"] = js["p"];
    info_json["min_consecutive_loops"] = js["min_consecutive_loops"];
    info_json["min_inliers"] = js["min_inliers"];
  }

  // Writing information to the info file
  info_file << std::setw(4) << info_json << std::endl;
  info_file.close();

  // Loading image filenames
  std::vector<std::string> filenames;
  getFilenames(base_dir + "images/", &filenames);
  unsigned nimages = filenames.size();
  std::cout << nimages << " images found" << std::endl;

  // Describing the images
  std::vector<unsigned> image_ids;
  std::vector<std::vector<cv::KeyPoint> > kps;
  std::vector<cv::Mat> descs;

  std::cout << "Describing images ..." << std::endl;
  cv::Ptr<cv::Feature2D> detector = cv::ORB::create(1500);  // Default params

  // Processing the sequence of images
  for (unsigned i = 0; i < nimages; i++) {
    // Processing image i

    // Loading and describing the image
    cv::Mat img = cv::imread(filenames[i]);
    std::vector<cv::KeyPoint> tkps;
    detector->detect(img, tkps);
    cv::Mat tdescs;
    detector->compute(img, tkps, tdescs);

    image_ids.push_back(i);
    kps.push_back(tkps);
    descs.push_back(tdescs);
  }

  std::cout << "Images described" << std::endl;

  // Executing the corresponding steps
  ibow_lcd::LCEvaluator eval;

  if (debug) {
    std::cout << "Execution in DEBUG mode ..." << std::endl;

    // Obtaining parameters
    ibow_lcd::LCDetectorParams params;
    params.purge_descriptors = js["purge_descriptors"];
    params.min_feat_apps = js["min_feat_apps"];
    params.nndr = js["nndr"];
    params.nndr_bf = js["nndr_bf"];
    params.ep_dist = js["ep_dist"];
    params.conf_prob = js["conf_prob"];
    params.p = js["p"];
    params.min_score = js["min_score"];
    params.island_size = js["island_size"];
    params.min_inliers = js["min_inliers"];
    params.nframes_after_lc = js["nframes_after_lc"];
    params.min_consecutive_loops = js["min_consecutive_loops"];

    eval.setIndexParams(params);

    // Writing the results to a file
    char output_filename[500];
    sprintf(output_filename, "%s%s/loops.txt", results_dir.c_str(),
                                               config_name.c_str());
    std::ofstream output_file(output_filename);
    eval.detectLoops(image_ids, kps, descs, output_file);
    output_file.close();
  } else {
    unsigned nsteps = js["executions"].size();
    for (unsigned i = 0; i < nsteps; i++) {
      std::cout << "Execution " << i << std::endl;

      // Reading the current configuration
      ibow_lcd::LCDetectorParams params;
      params.purge_descriptors = js["executions"][i]["purge_descriptors"];
      params.min_feat_apps = js["executions"][i]["min_feat_apps"];
      params.nndr = js["executions"][i]["nndr"];
      params.nndr_bf = js["executions"][i]["nndr_bf"];
      params.ep_dist = js["executions"][i]["ep_dist"];
      params.conf_prob = js["executions"][i]["conf_prob"];
      params.p = js["executions"][i]["p"];
      params.min_score = js["executions"][i]["min_score"];
      params.island_size = js["executions"][i]["island_size"];
      params.min_inliers = js["executions"][i]["min_inliers"];
      params.nframes_after_lc = js["executions"][i]["nframes_after_lc"];
      params.min_consecutive_loops = js["executions"][i]["min_consecutive_loops"];

      // Configuring the evaluator
      eval.setIndexParams(params);

        // Executing the process
      std::vector<ibow_lcd::LCDetectorResult> results;
      eval.detectLoops(image_ids, kps, descs, &results);

      // Writing the results to a file
      char output_filename[500];
      sprintf(output_filename, "%s%s/loops_%03d.txt",
                                              results_dir.c_str(),
                                              config_name.c_str(),
                                              i);

      std::ofstream output_file(output_filename);
      for (unsigned j = 0; j < results.size(); j++) {
        ibow_lcd::LCDetectorResult result = results[j];
        output_file << result.query_id << "\t";
        output_file << result.status << "\t";
        output_file << result.train_id << "\t";
        output_file << result.inliers;
        output_file << std::endl;
      }
      output_file.close();
    }
  }

  std::cout << "Evaluation finished" << std::endl;

  return 0;
}
