import random
import numpy as np


class GlColors:
    kRed = np.array([1.0, 0.0, 0.0, 1.0])
    kGreen = np.array([0.0, 1.0, 0.0, 1.0])
    kBlue = np.array([0.0, 0.0, 1.0, 1.0])
    kYellow = np.array([1.0, 1.0, 0.0, 1.0])
    kCyan = np.array([0.0, 1.0, 1.0, 1.0])
    kMagenta = np.array([1.0, 0.0, 1.0, 1.0])
    kWhite = np.array([1.0, 1.0, 1.0, 1.0])
    kBlack = np.array([0.0, 0.0, 0.0, 1.0])
    kGray = np.array([0.5, 0.5, 0.5, 1.0])
    kOrange = np.array([1.0, 0.5, 0.0, 1.0])
    kPurple = np.array([0.5, 0.0, 1.0, 1.0])
    kBrown = np.array([0.5, 0.25, 0.0, 1.0])
    kTeal = np.array([0.0, 0.5, 0.5, 1.0])
    kPink = np.array([1.0, 0.5, 1.0, 1.0])
    kLime = np.array([0.5, 1.0, 0.0, 1.0])
    kIndigo = np.array([0.25, 0.0, 0.5, 1.0])
    kGold = np.array([1.0, 0.75, 0.0, 1.0])
    kMaroon = np.array([0.5, 0.0, 0.25, 1.0])
    kLavender = np.array([0.75, 0.5, 1.0, 1.0])
    kOlive = np.array([0.5, 0.5, 0.0, 1.0])
    kCoral = np.array([1.0, 0.5, 0.5, 1.0])
    kKhaki = np.array([0.75, 0.5, 0.0, 1.0])
    kAqua = np.array([0.0, 1.0, 0.5, 1.0])
    kLimeGreen = np.array([0.5, 1.0, 0.5, 1.0])
    kCrimson = np.array([1.0, 0.5, 0.5, 1.0])
    kTurquoise = np.array([0.0, 1.0, 1.0, 1.0])
    kNavy = np.array([0.0, 0.0, 0.5, 1.0])
    kViolet = np.array([0.5, 0.0, 0.5, 1.0])
    kPinkViolet = np.array([0.5, 0.5, 0.5, 1.0])
    kCyanViolet = np.array([0.5, 0.5, 0.5, 1.0])
    
    _instance = None
    
    def __init__(self):
        # get colors from static fields by iterating over them
        self.colors = [getattr(GlColors, color) for color in dir(GlColors) if isinstance(getattr(GlColors, color), np.ndarray) and color.startswith("k")]
        self.num_colors = len(self.colors)
        
    @staticmethod
    def get_color(i):
        if not GlColors._instance:
            GlColors._instance = GlColors()
        return getattr(GlColors, GlColors._instance.colors[i % GlColors._instance.num_colors])
    
    @staticmethod 
    def get_colors():
        if not GlColors._instance:
            GlColors._instance = GlColors()
        return GlColors._instance.colors

    @staticmethod
    def get_random_color():
        if not GlColors._instance:
            GlColors._instance = GlColors()
        return GlColors._instance.colors[random.randint(0, GlColors._instance.num_colors-1)]