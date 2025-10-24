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

#include "semantic_labels.h"

namespace pyslam {

std::vector<cv::Vec3b> get_generic_color_map(int num_classes) {
    return generate_hsv_color_map(num_classes);
}

std::vector<cv::Vec3b> generate_hsv_color_map(int n, double s, double v) {
    std::vector<cv::Vec3b> rgb_colors;
    rgb_colors.reserve(n);

    for (int i = 0; i < n; ++i) {
        double h = static_cast<double>(i) / n;

        // Convert HSV to RGB
        double c = v * s;
        double x = c * (1.0 - std::abs(std::fmod(h * 6.0, 2.0) - 1.0));
        double m = v - c;

        double r, g, b;
        if (h < 1.0 / 6.0) {
            r = c;
            g = x;
            b = 0;
        } else if (h < 2.0 / 6.0) {
            r = x;
            g = c;
            b = 0;
        } else if (h < 3.0 / 6.0) {
            r = 0;
            g = c;
            b = x;
        } else if (h < 4.0 / 6.0) {
            r = 0;
            g = x;
            b = c;
        } else if (h < 5.0 / 6.0) {
            r = x;
            g = 0;
            b = c;
        } else {
            r = c;
            g = 0;
            b = x;
        }

        r = (r + m) * 255.0;
        g = (g + m) * 255.0;
        b = (b + m) * 255.0;

        rgb_colors.emplace_back(static_cast<uchar>(std::round(b)),
                                static_cast<uchar>(std::round(g)),
                                static_cast<uchar>(std::round(r)));
    }

    return rgb_colors;
}

std::vector<cv::Vec3b> get_voc_color_map() {
    return {
        cv::Vec3b(0, 0, 0),       // 0=background
        cv::Vec3b(0, 64, 0),      // 1=aeroplane
        cv::Vec3b(0, 128, 0),     // 2=bicycle
        cv::Vec3b(128, 128, 0),   // 3=bird
        cv::Vec3b(0, 0, 128),     // 4=boat
        cv::Vec3b(128, 0, 128),   // 5=bottle
        cv::Vec3b(0, 128, 128),   // 6=bus
        cv::Vec3b(128, 128, 128), // 7=car
        cv::Vec3b(64, 0, 0),      // 8=cat
        cv::Vec3b(192, 0, 0),     // 9=chair
        cv::Vec3b(64, 128, 0),    // 10=cow
        cv::Vec3b(192, 128, 0),   // 11=diningtable
        cv::Vec3b(64, 0, 128),    // 12=dog
        cv::Vec3b(192, 0, 128),   // 13=horse
        cv::Vec3b(64, 128, 128),  // 14=motorbike
        cv::Vec3b(192, 128, 128), // 15=person
        cv::Vec3b(0, 64, 0),      // 16=potted plant
        cv::Vec3b(128, 64, 0),    // 17=sheep
        cv::Vec3b(0, 192, 0),     // 18=sofa
        cv::Vec3b(128, 192, 0),   // 19=train
        cv::Vec3b(0, 64, 128)     // 20=tv/monitor
    };
}

std::vector<std::string> get_voc_labels() {
    return {"background", "aeroplane", "bicycle",   "bird",   "boat",         "bottle",
            "bus",        "car",       "cat",       "chair",  "cow",          "dining table",
            "dog",        "horse",     "motorbike", "person", "potted plant", "sheep",
            "sofa",       "train",     "tv monitor"};
}

std::vector<cv::Vec3b> get_cityscapes_color_map() {
    return {
        cv::Vec3b(128, 64, 128),  // 0=road
        cv::Vec3b(244, 35, 232),  // 1=sidewalk
        cv::Vec3b(70, 70, 70),    // 2=building
        cv::Vec3b(102, 102, 156), // 3=wall
        cv::Vec3b(190, 153, 153), // 4=fence
        cv::Vec3b(153, 153, 153), // 5=pole
        cv::Vec3b(250, 170, 30),  // 6=traffic light
        cv::Vec3b(220, 220, 0),   // 7=traffic sign
        cv::Vec3b(107, 142, 35),  // 8=vegetation
        cv::Vec3b(152, 251, 152), // 9=terrain
        cv::Vec3b(70, 130, 180),  // 10=sky
        cv::Vec3b(220, 20, 60),   // 11=person
        cv::Vec3b(255, 0, 0),     // 12=rider
        cv::Vec3b(0, 0, 142),     // 13=car
        cv::Vec3b(0, 0, 70),      // 14=truck
        cv::Vec3b(0, 60, 100),    // 15=bus
        cv::Vec3b(0, 80, 100),    // 16=train
        cv::Vec3b(0, 0, 230),     // 17=motorcycle
        cv::Vec3b(119, 11, 32)    // 18=bicycle
    };
}

std::vector<std::string> get_cityscapes_labels() {
    return {"road", "sidewalk",      "building",     "wall",       "fence",
            "pole", "traffic light", "traffic sign", "vegetation", "terrain",
            "sky",  "person",        "rider",        "car",        "truck",
            "bus",  "train",         "motorcycle",   "bicycle"};
}

std::vector<cv::Vec3b> get_nyu40_color_map() {
    return {
        cv::Vec3b(0, 0, 0),       // 0=unlabeled
        cv::Vec3b(174, 199, 232), // 1=wall
        cv::Vec3b(152, 223, 138), // 2=floor
        cv::Vec3b(31, 119, 180),  // 3=cabinet
        cv::Vec3b(255, 187, 120), // 4=bed
        cv::Vec3b(188, 189, 34),  // 5=chair
        cv::Vec3b(140, 86, 75),   // 6=sofa
        cv::Vec3b(255, 152, 150), // 7=table
        cv::Vec3b(214, 39, 40),   // 8=door
        cv::Vec3b(197, 176, 213), // 9=window
        cv::Vec3b(148, 103, 189), // 10=bookshelf
        cv::Vec3b(196, 156, 148), // 11=picture
        cv::Vec3b(23, 190, 207),  // 12=counter
        cv::Vec3b(178, 76, 76),   // 13=blinds
        cv::Vec3b(247, 182, 210), // 14=desk
        cv::Vec3b(66, 188, 102),  // 15=shelves
        cv::Vec3b(219, 219, 141), // 16=curtain
        cv::Vec3b(140, 57, 197),  // 17=dresser
        cv::Vec3b(202, 185, 52),  // 18=pillow
        cv::Vec3b(51, 176, 203),  // 19=mirror
        cv::Vec3b(200, 54, 131),  // 20=floormat
        cv::Vec3b(92, 193, 61),   // 21=clothes
        cv::Vec3b(78, 71, 183),   // 22=ceiling
        cv::Vec3b(172, 114, 82),  // 23=books
        cv::Vec3b(255, 127, 14),  // 24=refrigerator
        cv::Vec3b(91, 163, 138),  // 25=television
        cv::Vec3b(153, 98, 156),  // 26=paper
        cv::Vec3b(140, 153, 101), // 27=towel
        cv::Vec3b(158, 218, 229), // 28=showercurtain
        cv::Vec3b(100, 125, 154), // 29=box
        cv::Vec3b(178, 127, 135), // 30=whiteboard
        cv::Vec3b(120, 185, 128), // 31=person
        cv::Vec3b(146, 111, 194), // 32=nightstand
        cv::Vec3b(44, 160, 44),   // 33=toilet
        cv::Vec3b(112, 128, 144), // 34=sink
        cv::Vec3b(96, 207, 209),  // 35=lamp
        cv::Vec3b(227, 119, 194), // 36=bathtub
        cv::Vec3b(213, 92, 176),  // 37=bag
        cv::Vec3b(94, 106, 211),  // 38=otherstructure
        cv::Vec3b(82, 84, 163),   // 39=otherfurniture
        cv::Vec3b(100, 85, 144)   // 40=otherprop
    };
}

std::vector<std::string> get_nyu40_labels() {
    return {"unlabeled",
            "wall",
            "floor",
            "cabinet",
            "bed",
            "chair",
            "sofa",
            "table",
            "door",
            "window",
            "bookshelf",
            "picture",
            "counter",
            "desk",
            "curtain",
            "refrigerator",
            "showercurtain",
            "toilet",
            "sink",
            "bathtub",
            "cloth",
            "bag",
            "other structure",
            "other furniture",
            "other prop"};
}

std::vector<int> get_ade20k_to_scannet40_map() {
    return {
        1,  // wall -> wall
        0,  // building -> unlabeled
        0,  // sky -> unlabeled
        2,  // floor -> floor
        0,  // tree -> unlabeled
        22, // ceiling -> ceiling
        2,  // road -> floor
        4,  // bed -> bed
        9,  // windowpane -> window
        2,  // grass -> floor
        3,  // cabinet -> cabinet
        2,  // sidewalk -> floor
        31, // person -> person (unlabeled)
        0,  // earth -> unlabeled
        8,  // door -> door
        7,  // table -> table
        0,  // mountain -> unlabeled
        40, // plant -> otherprop
        16, // curtain -> curtain
        5,  // chair -> chair
        0,  // car -> unlabeled
        0,  // water -> unlabeled
        11, // painting -> picture
        6,  // sofa -> sofa
        15, // shelf -> shelves
        0,  // house -> unlabeled
        0,  // sea -> unlabeled
        19, // mirror -> mirror
        40, // rug -> otherprop
        2,  // field -> floor
        5,  // armchair -> chair
        5,  // seat -> chair
        0,  // fence -> unlabeled
        14, // desk -> desk
        0,  // rock -> unlabeled
        39, // wardrobe -> otherfurniture
        35, // lamp -> lamp
        19, // bathtub -> bathtub
        38, // railing -> otherstructure
        18, // cushion -> pillow
        0,  // base -> unlabeled
        29, // box -> box
        38, // column -> otherstructure
        40, // signboard -> otherprop
        3,  // chest of drawers -> cabinet
        12, // counter -> counter
        0,  // sand -> unlabeled
        34, // sink -> sink
        0,  // skyscraper -> unlabeled
        38, // fireplace -> otherstructure
        24, // refrigerator -> refrigerator
        0,  // grandstand -> unlabeled
        0,  // path -> unlabeled
        38, // stairs -> otherstructure
        0,  // runway -> unlabeled
        40, // case -> otherprop
        39, // pool table -> otherfurniture
        18, // pillow -> pillow
        8,  // screen door -> door
        38, // stairway -> otherstructure
        0,  // river -> unlabeled
        0,  // bridge -> unlabeled
        10, // bookcase -> bookshelf
        13, // blind -> blinds
        7,  // coffee table -> table
        33, // toilet -> toilet
        40, // flower -> otherprop
        23, // book -> books
        0,  // hill -> unlabeled
        39, // bench -> otherfurniture
        12, // countertop -> counter
        38, // stove -> otherstructure
        0,  // palm tree -> unlabeled
        38, // kitchen island -> otherstructure
        40, // computer -> otherprop
        5,  // swivel chair -> chair
        0,  // boat -> unlabeled
        0,  // bar -> unlabeled
        0,  // arcade machine -> unlabeled
        0,  // hovel -> unlabeled
        0,  // bus -> unlabeled
        27, // towel -> towel
        35, // light -> lamp
        0,  // truck -> unlabeled
        0,  // tower -> unlabeled
        35, // chandelier -> lamp
        19, // awning -> otherfurniture
        35, // streetlight -> lamp
        38, // booth -> otherstructure
        25, // television -> television
        0,  // airplane -> unlabeled
        0,  // dirt track -> unlabeled
        21, // apparel -> clothes
        0,  // pole -> unlabeled
        0,  // land -> unlabeled
        38, // bannister -> otherstructure
        0,  // escalator -> unlabeled
        39, // ottoman -> otherfurniture
        40, // bottle -> otherprop
        0,  // buffet -> unlabeled
        40, // poster -> otherprop
        0,  // stage -> unlabeled
        0,  // van -> unlabeled
        0,  // ship -> unlabeled
        38, // fountain -> otherstructure
        0,  // conveyer belt -> unlabeled
        0,  // canopy -> unlabeled
        39, // washer -> otherfurniture
        40, // toy -> otherprop
        0,  // swimming pool -> unlabeled
        5,  // stool -> chair
        0,  // barrel -> unlabeled
        40, // basket -> otherprop
        0,  // waterfall -> unlabeled
        0,  // tent -> unlabeled
        37, // bag -> bag
        40, // minibike -> otherprop
        0,  // cradle -> unlabeled
        38, // oven -> otherstructure
        40, // ball -> otherprop
        40, // food -> otherprop
        38, // step -> otherstructure
        40, // tank -> otherprop
        0,  // trade name -> unlabeled
        40, // microwave -> otherprop
        40, // pot -> otherprop
        0,  // animal -> unlabeled
        40, // bicycle -> otherprop
        0,  // lake -> unlabeled
        38, // dishwasher -> otherstructure
        40, // screen -> otherprop
        40, // blanket -> otherprop
        40, // sculpture -> otherprop
        21, // hood -> clothes
        0,  // sconce -> unlabeled
        40, // vase -> otherprop
        0,  // traffic light -> unlabeled
        40, // tray -> otherprop
        39, // ashcan -> otherfurniture
        40, // fan -> otherprop
        0,  // pier -> unlabeled
        0,  // crt screen -> unlabeled
        40, // plate -> otherprop
        11, // monitor -> television
        0,  // bulletin board -> unlabeled
        38, // shower -> otherstructure
        0,  // radiator -> otherfurniture (unlabeled)
        40, // glass -> otherprop
        40, // clock -> otherprop
        40  // flag -> otherprop
    };
}

std::vector<cv::Vec3b> get_ade20k_color_map(bool bgr) {
    std::vector<cv::Vec3b> color_map = {
        cv::Vec3b(120, 120, 120), // 0: wall
        cv::Vec3b(180, 120, 120), // 1: building
        cv::Vec3b(6, 230, 230),   // 2: sky
        cv::Vec3b(80, 50, 50),    // 3: floor
        cv::Vec3b(4, 200, 3),     // 4: tree
        cv::Vec3b(120, 120, 80),  // 5: ceiling
        cv::Vec3b(140, 140, 140), // 6: road
        cv::Vec3b(204, 5, 255),   // 7: bed
        cv::Vec3b(230, 230, 230), // 8: windowpane
        cv::Vec3b(4, 250, 7),     // 9: grass
        cv::Vec3b(224, 5, 255),   // 10: cabinet
        cv::Vec3b(235, 255, 7),   // 11: sidewalk
        cv::Vec3b(150, 5, 61),    // 12: person
        cv::Vec3b(120, 120, 70),  // 13: earth
        cv::Vec3b(8, 255, 51),    // 14: door
        cv::Vec3b(255, 6, 82),    // 15: table
        cv::Vec3b(143, 255, 140), // 16: mountain
        cv::Vec3b(204, 255, 4),   // 17: plant
        cv::Vec3b(255, 51, 7),    // 18: curtain
        cv::Vec3b(204, 70, 3),    // 19: chair
        cv::Vec3b(0, 102, 200),   // 20: car
        cv::Vec3b(61, 230, 250),  // 21: water
        cv::Vec3b(255, 6, 51),    // 22: painting
        cv::Vec3b(11, 102, 255),  // 23: sofa
        cv::Vec3b(255, 7, 71),    // 24: shelf
        cv::Vec3b(255, 9, 224),   // 25: house
        cv::Vec3b(9, 7, 230),     // 26: sea
        cv::Vec3b(220, 220, 220), // 27: mirror
        cv::Vec3b(255, 9, 92),    // 28: rug
        cv::Vec3b(112, 9, 255),   // 29: field
        cv::Vec3b(8, 255, 214),   // 30: armchair
        cv::Vec3b(7, 255, 224),   // 31: seat
        cv::Vec3b(255, 184, 6),   // 32: fence
        cv::Vec3b(10, 255, 71),   // 33: desk
        cv::Vec3b(255, 41, 10),   // 34: rock
        cv::Vec3b(7, 255, 255),   // 35: wardrobe
        cv::Vec3b(224, 255, 8),   // 36: lamp
        cv::Vec3b(102, 8, 255),   // 37: bathtub
        cv::Vec3b(255, 61, 6),    // 38: railing
        cv::Vec3b(255, 194, 7),   // 39: cushion
        cv::Vec3b(255, 122, 8),   // 40: base
        cv::Vec3b(0, 255, 20),    // 41: box
        cv::Vec3b(255, 8, 41),    // 42: column
        cv::Vec3b(255, 5, 153),   // 43: signboard
        cv::Vec3b(6, 51, 255),    // 44: chest of drawers
        cv::Vec3b(235, 12, 255),  // 45: counter
        cv::Vec3b(160, 150, 20),  // 46: sand
        cv::Vec3b(0, 163, 255),   // 47: sink
        cv::Vec3b(140, 140, 140), // 48: skyscraper
        cv::Vec3b(250, 10, 15),   // 49: fireplace
        cv::Vec3b(20, 255, 0),    // 50: refrigerator
        cv::Vec3b(31, 255, 0),    // 51: grandstand
        cv::Vec3b(255, 31, 0),    // 52: path
        cv::Vec3b(255, 224, 0),   // 53: stairs
        cv::Vec3b(153, 255, 0),   // 54: runway
        cv::Vec3b(0, 0, 255),     // 55: case
        cv::Vec3b(255, 71, 0),    // 56: pool table
        cv::Vec3b(0, 235, 255),   // 57: pillow
        cv::Vec3b(0, 173, 255),   // 58: screen door
        cv::Vec3b(31, 0, 255),    // 59: stairway
        cv::Vec3b(11, 200, 200),  // 60: river
        cv::Vec3b(255, 82, 0),    // 61: bridge
        cv::Vec3b(0, 255, 245),   // 62: bookcase
        cv::Vec3b(0, 61, 255),    // 63: blind
        cv::Vec3b(0, 255, 112),   // 64: coffee table
        cv::Vec3b(0, 255, 133),   // 65: toilet
        cv::Vec3b(255, 0, 0),     // 66: flower
        cv::Vec3b(255, 163, 0),   // 67: book
        cv::Vec3b(255, 102, 0),   // 68: hill
        cv::Vec3b(194, 255, 0),   // 69: bench
        cv::Vec3b(0, 143, 255),   // 70: countertop
        cv::Vec3b(51, 255, 0),    // 71: stove
        cv::Vec3b(0, 82, 255),    // 72: palm tree
        cv::Vec3b(0, 255, 41),    // 73: kitchen island
        cv::Vec3b(0, 255, 173),   // 74: computer
        cv::Vec3b(10, 0, 255),    // 75: swivel chair
        cv::Vec3b(173, 255, 0),   // 76: boat
        cv::Vec3b(0, 255, 153),   // 77: bar
        cv::Vec3b(255, 92, 0),    // 78: arcade machine
        cv::Vec3b(255, 0, 255),   // 79: hovel
        cv::Vec3b(255, 0, 245),   // 80: bus
        cv::Vec3b(255, 0, 102),   // 81: towel
        cv::Vec3b(255, 173, 0),   // 82: light
        cv::Vec3b(255, 0, 20),    // 83: truck
        cv::Vec3b(255, 184, 184), // 84: tower
        cv::Vec3b(0, 31, 255),    // 85: chandelier
        cv::Vec3b(0, 255, 61),    // 86: awning
        cv::Vec3b(0, 71, 255),    // 87: streetlight
        cv::Vec3b(255, 0, 204),   // 88: booth
        cv::Vec3b(0, 255, 194),   // 89: television
        cv::Vec3b(0, 255, 82),    // 90: airplane
        cv::Vec3b(0, 10, 255),    // 91: dirt track
        cv::Vec3b(0, 112, 255),   // 92: apparel
        cv::Vec3b(51, 0, 255),    // 93: pole
        cv::Vec3b(0, 194, 255),   // 94: land
        cv::Vec3b(0, 122, 255),   // 95: bannister
        cv::Vec3b(0, 255, 163),   // 96: escalator
        cv::Vec3b(255, 153, 0),   // 97: ottoman
        cv::Vec3b(0, 255, 10),    // 98: bottle
        cv::Vec3b(255, 112, 0),   // 99: buffet
        cv::Vec3b(143, 255, 0),   // 100: poster
        cv::Vec3b(82, 0, 255),    // 101: stage
        cv::Vec3b(163, 255, 0),   // 102: van
        cv::Vec3b(255, 235, 0),   // 103: ship
        cv::Vec3b(8, 184, 170),   // 104: fountain
        cv::Vec3b(133, 0, 255),   // 105: conveyer belt
        cv::Vec3b(0, 255, 92),    // 106: canopy
        cv::Vec3b(184, 0, 255),   // 107: washer
        cv::Vec3b(255, 0, 31),    // 108: toy
        cv::Vec3b(0, 184, 255),   // 109: swimming pool
        cv::Vec3b(0, 214, 255),   // 110: stool
        cv::Vec3b(255, 0, 112),   // 111: barrel
        cv::Vec3b(92, 255, 0),    // 112: basket
        cv::Vec3b(0, 224, 255),   // 113: waterfall
        cv::Vec3b(112, 224, 255), // 114: tent
        cv::Vec3b(70, 184, 160),  // 115: bag
        cv::Vec3b(163, 0, 255),   // 116: minibike
        cv::Vec3b(153, 0, 255),   // 117: cradle
        cv::Vec3b(71, 255, 0),    // 118: oven
        cv::Vec3b(255, 0, 163),   // 119: ball
        cv::Vec3b(255, 204, 0),   // 120: food
        cv::Vec3b(255, 0, 143),   // 121: step
        cv::Vec3b(0, 255, 235),   // 122: tank
        cv::Vec3b(133, 255, 0),   // 123: trade name
        cv::Vec3b(255, 0, 235),   // 124: microwave
        cv::Vec3b(245, 0, 255),   // 125: pot
        cv::Vec3b(255, 0, 122),   // 126: animal
        cv::Vec3b(255, 245, 0),   // 127: bicycle
        cv::Vec3b(10, 190, 212),  // 128: lake
        cv::Vec3b(214, 255, 0),   // 129: dishwasher
        cv::Vec3b(0, 204, 255),   // 130: screen
        cv::Vec3b(20, 0, 255),    // 131: blanket
        cv::Vec3b(255, 255, 0),   // 132: sculpture
        cv::Vec3b(0, 153, 255),   // 133: hood
        cv::Vec3b(0, 41, 255),    // 134: sconce
        cv::Vec3b(0, 255, 204),   // 135: vase
        cv::Vec3b(41, 0, 255),    // 136: traffic light
        cv::Vec3b(41, 255, 0),    // 137: tray
        cv::Vec3b(173, 0, 255),   // 138: ashcan
        cv::Vec3b(0, 245, 255),   // 139: fan
        cv::Vec3b(71, 0, 255),    // 140: pier
        cv::Vec3b(122, 0, 255),   // 141: crt screen
        cv::Vec3b(0, 255, 184),   // 142: plate
        cv::Vec3b(0, 92, 255),    // 143: monitor
        cv::Vec3b(184, 255, 0),   // 144: bulletin board
        cv::Vec3b(0, 133, 255),   // 145: shower
        cv::Vec3b(255, 214, 0),   // 146: radiator
        cv::Vec3b(25, 194, 194),  // 147: glass
        cv::Vec3b(102, 255, 0),   // 148: clock
        cv::Vec3b(92, 0, 255)     // 149: flag
    };

    if (bgr) {
        for (auto &color : color_map) {
            std::swap(color[0], color[2]); // Convert RGB to BGR
        }
    }

    return color_map;
}

std::vector<std::string> get_ade20k_labels() {
    return {"wall",
            "building",
            "sky",
            "floor",
            "tree",
            "ceiling",
            "road",
            "bed",
            "windowpane",
            "grass",
            "cabinet",
            "sidewalk",
            "person",
            "earth",
            "door",
            "table",
            "mountain",
            "plant",
            "curtain",
            "chair",
            "car",
            "water",
            "painting",
            "sofa",
            "shelf",
            "house",
            "sea",
            "mirror",
            "rug",
            "field",
            "armchair",
            "seat",
            "fence",
            "desk",
            "rock",
            "wardrobe",
            "lamp",
            "bathtub",
            "railing",
            "cushion",
            "base",
            "box",
            "column",
            "signboard",
            "chest of drawers",
            "counter",
            "sand",
            "sink",
            "skyscraper",
            "fireplace",
            "refrigerator",
            "grandstand",
            "path",
            "stairs",
            "runway",
            "case",
            "pool table",
            "pillow",
            "screen door",
            "stairway",
            "river",
            "bridge",
            "bookcase",
            "blind",
            "coffee table",
            "toilet",
            "flower",
            "book",
            "hill",
            "bench",
            "countertop",
            "stove",
            "palm tree",
            "kitchen island",
            "computer",
            "swivel chair",
            "boat",
            "bar",
            "arcade machine",
            "hovel",
            "bus",
            "towel",
            "light",
            "truck",
            "tower",
            "chandelier",
            "awning",
            "streetlight",
            "booth",
            "television",
            "airplane",
            "dirt track",
            "apparel",
            "pole",
            "land",
            "bannister",
            "escalator",
            "ottoman",
            "bottle",
            "buffet",
            "poster",
            "stage",
            "van",
            "ship",
            "fountain",
            "conveyer belt",
            "canopy",
            "washer",
            "toy",
            "swimming pool",
            "stool",
            "barrel",
            "basket",
            "waterfall",
            "tent",
            "bag",
            "minibike",
            "cradle",
            "oven",
            "ball",
            "food",
            "step",
            "tank",
            "trade name",
            "microwave",
            "pot",
            "animal",
            "bicycle",
            "lake",
            "dishwasher",
            "screen",
            "blanket",
            "sculpture",
            "hood",
            "sconce",
            "vase",
            "traffic light",
            "tray",
            "ashcan",
            "fan",
            "pier",
            "crt screen",
            "plate",
            "monitor",
            "bulletin board",
            "shower",
            "radiator",
            "glass",
            "clock",
            "flag"};
}

} // namespace pyslam