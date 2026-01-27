"""
* This file is part of PYSLAM
*
* Copyright (C) 2025-present David Morilla-Cabello <davidmorillacabello at gmail dot com>
* Copyright (C) 2025-present Luigi Freda <luigi dot freda at gmail dot com>
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
"""

import colorsys
import numpy as np


def get_generic_color_map(num_classes):
    """Generates a color map for generic semantic segmentation

    Args:
        num_classes (int): Number of classes

    Returns:
        np.ndarray: (num_classes, 3) array with RGB values in [0, 255]
    """
    return generate_hsv_color_map(num_classes)


def generate_hsv_color_map(n: int, s=0.65, v=0.95):
    """Generates `n` visually distinct RGB colors using HSV color space.

    Args:
        n (int): Number of colors to generate.
        s (float): Saturation (0-1)
        v (float): Brightness/Value (0-1)

    Returns:
        np.ndarray: (n, 3) array with RGB values in [0, 255]
    """
    hsv_colors = [(i / n, s, v) for i in range(n)]
    rgb_colors = [colorsys.hsv_to_rgb(*hsv) for hsv in hsv_colors]
    rgb_colors = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in rgb_colors]
    return np.array(rgb_colors, dtype=np.uint8)


# ==============================================
# PASCAL VOC
# ==============================================
# https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/


def get_voc_color_map():
    """Load the mapping that associates pascal VOC classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    color_map = np.array(
        [
            [0, 0, 0],  # 0=background
            [0, 64, 0],  # 1=aeroplane # TEMPORAL CHANGE
            [0, 128, 0],  # 2=bicycle
            [128, 128, 0],  # 3=bird
            [0, 0, 128],  # 4=boat
            [128, 0, 128],  # 5=bottle
            [0, 128, 128],  # 6=bus
            [128, 128, 128],  # 7=car
            [64, 0, 0],  # 8=cat
            [192, 0, 0],  # 9=chair
            [64, 128, 0],  # 10=cow
            [192, 128, 0],  # 11=diningtable
            [64, 0, 128],  # 12=dog
            [192, 0, 128],  # 13=horse
            [64, 128, 128],  # 14=motorbike
            [192, 128, 128],  # 15=person
            [0, 64, 0],  # 16=potted plant
            [128, 64, 0],  # 17=sheep
            [0, 192, 0],  # 18=sofa
            [128, 192, 0],  # 19=train
            [0, 64, 128],  # 20=tv/monitor
        ]
    )
    return color_map


def get_voc_labels():
    return [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "dining table",
        "dog",
        "horse",
        "motorbike",
        "person",
        "potted plant",
        "sheep",
        "sofa",
        "train",
        "tv monitor",
    ]


# ==============================================
# CITYSCAPES
# ==============================================
# https://www.cityscapes-dataset.com/


def get_cityscapes_color_map():
    """Load the mapping that associates cityscapes classes with label colors
    Returns:
        np.ndarray with dimensions (19, 3)
    """
    color_map = np.array(
        [
            [128, 64, 128],  # 0=road
            [244, 35, 232],  # 1=sidewalk
            [70, 70, 70],  # 2=building
            [102, 102, 156],  # 3=wall
            [190, 153, 153],  # 4=fence
            [153, 153, 153],  # 5=pole
            [250, 170, 30],  # 6=traffic light
            [220, 220, 0],  # 7=traffic sign
            [107, 142, 35],  # 8=vegetation
            [152, 251, 152],  # 9=terrain
            [70, 130, 180],  # 10=sky
            [220, 20, 60],  # 11=person
            [255, 0, 0],  # 12=rider
            [0, 0, 142],  # 13=car
            [0, 0, 70],  # 14=truck
            [0, 60, 100],  # 15=bus
            [0, 80, 100],  # 16=train
            [0, 0, 230],  # 17=motorcycle
            [119, 11, 32],  # 18=bicycle
        ]
    )
    return color_map


def get_cityscapes_labels():
    return [
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic light",
        "traffic sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
    ]


# ==============================================
# NYU40
# ==============================================
# https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html


def get_nyu40_color_map():
    """
    Load the mapping that associates NYU40 classes with label colors.

    Returns:
        np.ndarray with dimensions (41, 3)
    """
    color_map = np.array(
        [
            [0, 0, 0],  # 0=unlabeled
            [174, 199, 232],  # 1=wall
            [152, 223, 138],  # 2=floor
            [31, 119, 180],  # 3=cabinet
            [255, 187, 120],  # 4=bed
            [188, 189, 34],  # 5=chair
            [140, 86, 75],  # 6=sofa
            [255, 152, 150],  # 7=table
            [214, 39, 40],  # 8=door
            [197, 176, 213],  # 9=window
            [148, 103, 189],  # 10=bookshelf
            [196, 156, 148],  # 11=picture
            [23, 190, 207],  # 12=counter
            [178, 76, 76],  # 13=blinds
            [247, 182, 210],  # 14=desk
            [66, 188, 102],  # 15=shelves
            [219, 219, 141],  # 16=curtain
            [140, 57, 197],  # 17=dresser
            [202, 185, 52],  # 18=pillow
            [51, 176, 203],  # 19=mirror
            [200, 54, 131],  # 20=floormat
            [92, 193, 61],  # 21=clothes
            [78, 71, 183],  # 22=ceiling
            [172, 114, 82],  # 23=books
            [255, 127, 14],  # 24=refrigerator
            [91, 163, 138],  # 25=television
            [153, 98, 156],  # 26=paper
            [140, 153, 101],  # 27=towel
            [158, 218, 229],  # 28=showercurtain
            [100, 125, 154],  # 29=box
            [178, 127, 135],  # 30=whiteboard
            [120, 185, 128],  # 31=person
            [146, 111, 194],  # 32=nightstand
            [44, 160, 44],  # 33=toilet
            [112, 128, 144],  # 34=sink
            [96, 207, 209],  # 35=lamp
            [227, 119, 194],  # 36=bathtub
            [213, 92, 176],  # 37=bag
            [94, 106, 211],  # 38=otherstructure
            [82, 84, 163],  # 39=otherfurniture
            [100, 85, 144],  # 40=otherprop
        ]
    )
    return color_map


def get_nyu40_labels():
    return [
        "unlabeled",
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
        "blinds",
        "desk",
        "shelves",
        "curtain",
        "dresser",
        "pillow",
        "mirror",
        "floormat",
        "clothes",
        "ceiling",
        "books",
        "refrigerator",
        "television",
        "paper",
        "towel",
        "showercurtain",
        "box",
        "whiteboard",
        "person",
        "nightstand",
        "toilet",
        "sink",
        "lamp",
        "bathtub",
        "bag",
        "otherstructure",
        "otherfurniture",
        "otherprop",
    ]


# ==============================================
# ADE20K
# ==============================================
# https://groups.csail.mit.edu/vision/datasets/ADE20K/


def get_ade20k_to_scannet40_map():
    """
    Returns a NumPy array of 150 elements where each index corresponds to an ADE20K class ID,
    and the value is the mapped ScanNet20 class ID. Unmapped classes are assigned to 0 (unlabeled).
    """
    # Define the mapping from ADE20K class IDs to ScanNet20 class IDs
    mapping = np.array(
        [
            1,  # wall -> wall
            0,  # building -> unlabeled
            0,  # sky -> unlabeled
            2,  # floor -> floor
            0,  # tree -> unlabeled
            22,  # ceiling -> ceiling
            2,  # road -> floor
            4,  # bed -> bed
            9,  # windowpane -> window
            2,  # grass -> floor
            3,  # cabinet -> cabinet
            2,  # sidewalk -> floor
            31,  # person -> person (unlabeled)
            0,  # earth -> unlabeled
            8,  # door -> door
            7,  # table -> table
            0,  # mountain -> unlabeled
            40,  # plant -> otherprop
            16,  # curtain -> curtain
            5,  # chair -> chair
            0,  # car -> unlabeled
            0,  # water -> unlabeled
            11,  # painting -> picture
            6,  # sofa -> sofa
            15,  # shelf -> shelves
            0,  # house -> unlabeled
            0,  # sea -> unlabeled
            19,  # mirror -> mirror
            40,  # rug -> otherprop
            2,  # field -> floor
            5,  # armchair -> chair
            5,  # seat -> chair
            0,  # fence -> unlabeled
            14,  # desk -> desk
            0,  # rock -> unlabeled
            39,  # wardrobe -> otherfurniture
            35,  # lamp -> lamp
            19,  # bathtub -> bathtub
            38,  # railing -> otherstructure
            18,  # cushion -> pillow
            0,  # base -> unlabeled
            29,  # box -> box
            38,  # column -> otherstructure
            40,  # signboard -> otherprop
            3,  # chest of drawers -> cabinet
            12,  # counter -> counter
            0,  # sand -> unlabeled
            34,  # sink -> sink
            0,  # skyscraper -> unlabeled
            38,  # fireplace -> otherstructure
            24,  # refrigerator -> refrigerator
            0,  # grandstand -> unlabeled
            0,  # path -> unlabeled
            38,  # stairs -> otherstructure
            0,  # runway -> unlabeled
            40,  # case -> otherprop
            39,  # pool table -> otherfurniture
            18,  # pillow -> pillow
            8,  # screen door -> door
            38,  # stairway -> otherstructure
            0,  # river -> unlabeled
            0,  # bridge -> unlabeled
            10,  # bookcase -> bookshelf
            13,  # blind -> blinds
            7,  # coffee table -> table
            33,  # toilet -> toilet
            40,  # flower -> otherprop
            23,  # book -> books
            0,  # hill -> unlabeled
            39,  # bench -> otherfurniture
            12,  # countertop -> counter
            38,  # stove -> otherstructure
            0,  # palm tree -> unlabeled
            38,  # kitchen island -> otherstructure
            40,  # computer -> otherprop
            5,  # swivel chair -> chair
            0,  # boat -> unlabeled
            0,  # bar -> unlabeled
            0,  # arcade machine -> unlabeled
            0,  # hovel -> unlabeled
            0,  # bus -> unlabeled
            27,  # towel -> towel
            35,  # light -> lamp
            0,  # truck -> unlabeled
            0,  # tower -> unlabeled
            35,  # chandelier -> lamp
            19,  # awning -> otherfurniture
            35,  # streetlight -> lamp
            38,  # booth -> otherstructure
            25,  # television -> television
            0,  # airplane -> unlabeled
            0,  # dirt track -> unlabeled
            21,  # apparel -> clothes
            0,  # pole -> unlabeled
            0,  # land -> unlabeled
            38,  # bannister -> otherstructure
            0,  # escalator -> unlabeled
            39,  # ottoman -> otherfurniture
            40,  # bottle -> otherprop
            0,  # buffet -> unlabeled
            40,  # poster -> otherprop
            0,  # stage -> unlabeled
            0,  # van -> unlabeled
            0,  # ship -> unlabeled
            38,  # fountain -> otherstructure
            0,  # conveyer belt -> unlabeled
            0,  # canopy -> unlabeled
            39,  # washer -> otherfurniture
            40,  # toy -> otherprop
            0,  # swimming pool -> unlabeled
            5,  # stool -> chair
            0,  # barrel -> unlabeled
            40,  # basket -> otherprop
            0,  # waterfall -> unlabeled
            0,  # tent -> unlabeled
            37,  # bag -> bag
            40,  # minibike -> otherprop
            0,  # cradle -> unlabeled
            38,  # oven -> otherstructure
            40,  # ball -> otherprop
            40,  # food -> otherprop
            38,  # step -> otherstructure
            40,  # tank -> otherprop
            0,  # trade name -> unlabeled
            40,  # microwave -> otherprop
            40,  # pot -> otherprop
            0,  # animal -> unlabeled
            40,  # bicycle -> otherprop
            0,  # lake -> unlabeled
            38,  # dishwasher -> otherstructure
            40,  # screen -> otherprop
            40,  # blanket -> otherprop
            40,  # sculpture -> otherprop
            21,  # hood -> clothes
            0,  # sconce -> unlabeled
            40,  # vase -> otherprop
            0,  # traffic light -> unlabeled
            40,  # tray -> otherprop
            39,  # ashcan -> otherfurniture
            40,  # fan -> otherprop
            0,  # pier -> unlabeled
            0,  # crt screen -> unlabeled
            40,  # plate -> otherprop
            11,  # monitor -> television
            0,  # bulletin board -> unlabeled
            38,  # shower -> otherstructure
            0,  # radiator -> otherfurniture (unlabeled)
            40,  # glass -> otherprop
            40,  # clock -> otherprop
            40,  # flag -> otherprop
        ]
    )

    return mapping


def get_ade20k_color_map(bgr=False):
    """
    Returns the ADE20K color map as a NumPy array.

    Args:
        bgr (bool, optional): If True, returns the color map in BGR format
            instead of RGB. Defaults to False (RGB).

    Returns:
        numpy.ndarray: A NumPy array of shape (150, 3) representing the
        ADE20K color map.  Each row represents a color, and the columns
        represent the R, G, and B values (or B, G, R if bgr=True).
    """
    color_map = np.array(
        [
            [120, 120, 120],  # 0:  wall
            [180, 120, 120],  # 1:  building
            [6, 230, 230],  # 2:  sky
            [80, 50, 50],  # 3:  floor
            [4, 200, 3],  # 4:  tree
            [120, 120, 80],  # 5:  ceiling
            [140, 140, 140],  # 6:  road
            [204, 5, 255],  # 7:  bed
            [230, 230, 230],  # 8:  windowpane
            [4, 250, 7],  # 9: grass
            [224, 5, 255],  # 10: cabinet
            [235, 255, 7],  # 11: sidewalk
            [150, 5, 61],  # 12: person
            [120, 120, 70],  # 13: earth
            [8, 255, 51],  # 14: door
            [255, 6, 82],  # 15: table
            [143, 255, 140],  # 16: mountain
            [204, 255, 4],  # 17: plant
            [255, 51, 7],  # 18: curtain
            [204, 70, 3],  # 19: chair
            [0, 102, 200],  # 20: car
            [61, 230, 250],  # 21: water
            [255, 6, 51],  # 22: painting
            [11, 102, 255],  # 23: sofa
            [255, 7, 71],  # 24: shelf
            [255, 9, 224],  # 25: house
            [9, 7, 230],  # 26: sea
            [220, 220, 220],  # 27: mirror
            [255, 9, 92],  # 28: rug
            [112, 9, 255],  # 29: field
            [8, 255, 214],  # 30: armchair
            [7, 255, 224],  # 31: seat
            [255, 184, 6],  # 32: fence
            [10, 255, 71],  # 33: desk
            [255, 41, 10],  # 34: rock
            [7, 255, 255],  # 35: wardrobe
            [224, 255, 8],  # 36: lamp
            [102, 8, 255],  # 37: bathtub
            [255, 61, 6],  # 38: railing
            [255, 194, 7],  # 39: cushion
            [255, 122, 8],  # 40: base
            [0, 255, 20],  # 41: box
            [255, 8, 41],  # 42: column
            [255, 5, 153],  # 43: signboard
            [6, 51, 255],  # 44: chest of drawers
            [235, 12, 255],  # 45: counter
            [160, 150, 20],  # 46: sand
            [0, 163, 255],  # 47: sink
            [140, 140, 140],  # 48: skyscraper
            [250, 10, 15],  # 49: fireplace
            [20, 255, 0],  # 50: refrigerator
            [31, 255, 0],  # 51: grandstand
            [255, 31, 0],  # 52: path
            [255, 224, 0],  # 53: stairs
            [153, 255, 0],  # 54: runway
            [0, 0, 255],  # 55: case
            [255, 71, 0],  # 56: pool table
            [0, 235, 255],  # 57: pillow
            [0, 173, 255],  # 58: screen door
            [31, 0, 255],  # 59: stairway
            [11, 200, 200],  # 60: river
            [255, 82, 0],  # 61: bridge
            [0, 255, 245],  # 62: bookcase
            [0, 61, 255],  # 63: blind
            [0, 255, 112],  # 64: coffee table
            [0, 255, 133],  # 65: toilet
            [255, 0, 0],  # 66: flower
            [255, 163, 0],  # 67: book
            [255, 102, 0],  # 68: hill
            [194, 255, 0],  # 69: bench
            [0, 143, 255],  # 70: countertop
            [51, 255, 0],  # 71: stove
            [0, 82, 255],  # 72: palm tree
            [0, 255, 41],  # 73: kitchen island
            [0, 255, 173],  # 74: computer
            [10, 0, 255],  # 75: swivel chair
            [173, 255, 0],  # 76: boat
            [0, 255, 153],  # 77: bar
            [255, 92, 0],  # 78: arcade machine
            [255, 0, 255],  # 79: hovel
            [255, 0, 245],  # 80: bus
            [255, 0, 102],  # 81: towel
            [255, 173, 0],  # 82: light
            [255, 0, 20],  # 83: truck
            [255, 184, 184],  # 84: tower
            [0, 31, 255],  # 85: chandelier
            [0, 255, 61],  # 86: awning
            [0, 71, 255],  # 87: streetlight
            [255, 0, 204],  # 88: booth
            [0, 255, 194],  # 89: television
            [0, 255, 82],  # 90: airplane
            [0, 10, 255],  # 91: dirt track
            [0, 112, 255],  # 92: apparel
            [51, 0, 255],  # 93: pole
            [0, 194, 255],  # 94: land
            [0, 122, 255],  # 95: bannister
            [0, 255, 163],  # 96: escalator
            [255, 153, 0],  # 97: ottoman
            [0, 255, 10],  # 98: bottle
            [255, 112, 0],  # 99: buffet
            [143, 255, 0],  # 100: poster
            [82, 0, 255],  # 101: stage
            [163, 255, 0],  # 102: van
            [255, 235, 0],  # 103: ship
            [8, 184, 170],  # 104: fountain
            [133, 0, 255],  # 105: conveyer belt
            [0, 255, 92],  # 106: canopy
            [184, 0, 255],  # 107: washer
            [255, 0, 31],  # 108: toy
            [0, 184, 255],  # 109: swimming pool
            [0, 214, 255],  # 110: stool
            [255, 0, 112],  # 111: barrel
            [92, 255, 0],  # 112: basket
            [0, 224, 255],  # 113: waterfall
            [112, 224, 255],  # 114: tent
            [70, 184, 160],  # 115: bag
            [163, 0, 255],  # 116: minibike
            [153, 0, 255],  # 117: cradle
            [71, 255, 0],  # 118: oven
            [255, 0, 163],  # 119: ball
            [255, 204, 0],  # 120: food
            [255, 0, 143],  # 121: step
            [0, 255, 235],  # 122: tank
            [133, 255, 0],  # 123: trade name
            [255, 0, 235],  # 124: microwave
            [245, 0, 255],  # 125: pot
            [255, 0, 122],  # 126: animal
            [255, 245, 0],  # 127: bicycle
            [10, 190, 212],  # 128: lake
            [214, 255, 0],  # 129: dishwasher
            [0, 204, 255],  # 130: screen
            [20, 0, 255],  # 131: blanket
            [255, 255, 0],  # 132: sculpture
            [0, 153, 255],  # 133: hood
            [0, 41, 255],  # 134: sconce
            [0, 255, 204],  # 135: vase
            [41, 0, 255],  # 136: traffic light
            [41, 255, 0],  # 137: tray
            [173, 0, 255],  # 138: ashcan
            [0, 245, 255],  # 139: fan
            [71, 0, 255],  # 140: pier
            [122, 0, 255],  # 141: crt screen
            [0, 255, 184],  # 142: plate
            [0, 92, 255],  # 143: monitor
            [184, 255, 0],  # 144: bulletin board
            [0, 133, 255],  # 145: shower
            [255, 214, 0],  # 146: radiator
            [25, 194, 194],  # 147: glass
            [102, 255, 0],  # 148: clock
            [92, 0, 255],  # 149: flag
        ],
        dtype=np.uint8,
    )

    if bgr:
        color_map = color_map[:, ::-1]  # Convert RGB to BGR
    return color_map


def get_ade20k_labels():
    return [
        "wall",
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
        "flag",
    ]


# ==============================================
# Open-Vocabulary Models (EOV-Seg, Detic, etc.)
# ==============================================


def get_open_vocab_color_map(num_classes=3000):
    """
    Returns a large color map for open-vocabulary semantic segmentation models.

    Open-vocabulary models (like EOV-Seg and Detic) can output category IDs that are much
    larger than standard dataset class counts (e.g., category IDs can be 1203 for LVIS
    or 1432+ for ADE20K), so they need large color maps to avoid color collisions.

    Args:
        num_classes (int): Maximum number of classes to support. Default 3000 to handle
                          large category IDs from open-vocabulary models.

    Returns:
        np.ndarray: A NumPy array of shape (num_classes, 3) representing the
        color map. Each row represents a color, and the columns represent the
        R, G, and B values (0-255).
    """
    return get_generic_color_map(num_classes)
