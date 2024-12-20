
from utils_serialization import SerializableEnum, register_class

@register_class
class SlamState(SerializableEnum):
    NO_IMAGES_YET=0,
    NOT_INITIALIZED=1,
    OK=2,
    LOST=3
    RELOCALIZE=4
    INIT_RELOCALIZE=5       # used just for the first relocalization after map reloading

