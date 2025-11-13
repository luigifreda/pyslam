# ------------------------------------------------------------------ #
# ROS2 schemas
# ------------------------------------------------------------------ #

from typing import Dict, Optional
from mcap.reader import make_reader

# Some initial schemas definitions
IMAGE_SCHEMA_NAME = "sensor_msgs/Image"
IMAGE_SCHEMA_TEXT = """MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
===
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
===
MSG: sensor_msgs/Image
std_msgs/Header header
uint32 height
uint32 width
string encoding
uint8 is_bigendian
uint32 step
uint8[] data
"""

POINTCLOUD2_SCHEMA_NAME = "sensor_msgs/PointCloud2"
POINTCLOUD2_SCHEMA_TEXT = """MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
===
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
===
MSG: sensor_msgs/PointField
uint8 INT8=1
uint8 UINT8=2
uint8 INT16=3
uint8 UINT16=4
uint8 INT32=5
uint8 UINT32=6
uint8 FLOAT32=7
uint8 FLOAT64=8
string name
uint32 offset
uint8 datatype
uint32 count
===
MSG: sensor_msgs/PointCloud2
std_msgs/Header header
uint32 height
uint32 width
sensor_msgs/PointField[] fields
bool is_bigendian
uint32 point_step
uint32 row_step
uint8[] data
bool is_dense
"""

TF_SCHEMA_NAME = "tf2_msgs/TFMessage"
TF_SCHEMA_TEXT = """MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
===
MSG: geometry_msgs/Vector3
float64 x
float64 y
float64 z
===
MSG: geometry_msgs/Quaternion
float64 x
float64 y
float64 z
float64 w
===
MSG: geometry_msgs/Transform
geometry_msgs/Vector3 translation
geometry_msgs/Quaternion rotation
===
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
===
MSG: geometry_msgs/TransformStamped
std_msgs/Header header
string child_frame_id
geometry_msgs/Transform transform
===
MSG: tf2_msgs/TFMessage
geometry_msgs/TransformStamped[] transforms
"""


class Ros2Schemas:

    @staticmethod
    def get_schema(schema_name: str):
        lower_schema_name = schema_name.lower()
        if lower_schema_name == "image":
            return {"name": IMAGE_SCHEMA_NAME, "text": IMAGE_SCHEMA_TEXT}
        elif lower_schema_name == "pointcloud2":
            return {
                "name": POINTCLOUD2_SCHEMA_NAME,
                "text": POINTCLOUD2_SCHEMA_TEXT,
            }
        elif lower_schema_name == "tf":
            return {"name": TF_SCHEMA_NAME, "text": TF_SCHEMA_TEXT}
        else:
            raise ValueError(f"Unknown schema name: {schema_name}")
        return {"name": None, "text": None}

    @staticmethod
    def extract_from_mcap(mcap_path: str, schema_name: Optional[str] = None) -> Dict[str, str]:
        """
        Extract message definitions from an existing MCAP file.

        Args:
            mcap_path: Path to the MCAP file
            schema_name: Optional schema name to extract. If None, extracts all schemas.

        Returns:
            Dictionary mapping schema names to their message definition text.
            The message definition text is stored in schema.data as bytes.
        """
        schemas = {}

        with open(mcap_path, "rb") as f:
            reader = make_reader(f)
            seen_schemas = set()

            for schema, channel, message in reader.iter_messages():
                # Skip if we've already seen this schema
                if schema.name in seen_schemas:
                    continue

                # Filter by schema_name if provided
                if schema_name is not None and schema.name != schema_name:
                    continue

                seen_schemas.add(schema.name)

                # Extract the message definition from schema.data
                if schema.data:
                    try:
                        msgdef_text = schema.data.decode("utf-8")
                        schemas[schema.name] = msgdef_text
                    except UnicodeDecodeError:
                        # If it's not UTF-8, try to handle it
                        schemas[schema.name] = schema.data

                # If we found the specific schema we're looking for, we can break early
                if schema_name is not None and schema.name == schema_name:
                    break

        return schemas
