"""
* This file is part of PYSLAM
*
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

import os
import glob
import re

from mcap.reader import make_reader

from typing import Dict, List, Optional, Set, Union
from dataclasses import dataclass, field


def natural_sort_key(s):
    """Sort string using human order (e.g., file2 < file10)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


# -------------------------------------------------------------
# TopicInfo
# -------------------------------------------------------------
@dataclass
class TopicInfo:
    """Information about a single topic."""

    topic_name: str
    message_count: int = 0
    message_types: Set[str] = field(default_factory=set)
    first_timestamp: Optional[int] = None
    last_timestamp: Optional[int] = None
    channels: List[Dict] = field(default_factory=list)


# -------------------------------------------------------------
# McapSummary
# -------------------------------------------------------------
@dataclass
class McapSummary:
    """
    Summary information about MCAP file(s).

    Attributes:
        files: List of MCAP file paths analyzed
        topics: Dictionary mapping topic names to TopicInfo
        total_messages: Total number of messages across all topics
        header: MCAP header information (if available)
        start_time: Earliest message timestamp (nanoseconds)
        end_time: Latest message timestamp (nanoseconds)
        duration_seconds: Duration in seconds (if timestamps available)
    """

    files: List[str] = field(default_factory=list)
    topics: Dict[str, TopicInfo] = field(default_factory=dict)
    total_messages: int = 0
    header: Optional[Dict] = None
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    duration_seconds: Optional[float] = None

    def get_topic_list(self) -> List[str]:
        """Get sorted list of topic names."""
        return sorted(self.topics.keys())

    def get_message_count_per_topic(self) -> Dict[str, int]:
        """Get dictionary of message counts per topic."""
        return {topic: info.message_count for topic, info in self.topics.items()}

    def get_message_types_per_topic(self) -> Dict[str, List[str]]:
        """Get dictionary of message types per topic."""
        return {topic: sorted(list(info.message_types)) for topic, info in self.topics.items()}

    def __repr__(self):
        """String representation of the summary."""
        lines = [
            f"  Files: {len(self.files)}",
            f"  Total messages: {self.total_messages}",
            f"  Topics: {len(self.topics)}",
        ]
        if self.start_time is not None and self.end_time is not None:
            lines.append(f"  Time range: {self.start_time} - {self.end_time}")
            if self.duration_seconds is not None:
                lines.append(f"  Duration: {self.duration_seconds:.2f} seconds")
        lines.append("\n  Topics:")
        for topic_name in sorted(self.topics.keys()):
            info = self.topics[topic_name]
            types_str = ", ".join(sorted(info.message_types)) if info.message_types else "unknown"
            lines.append(f"    {topic_name}: {info.message_count} messages ({types_str})")
        return "\n".join(lines)

    def __str__(self):
        return str(self.__repr__())


def extract_mcap_summary(path: Union[str, List[str]], detect_sequence: bool = True) -> McapSummary:
    """
    Extract summary information from MCAP file(s).

    Args:
        path: Path to MCAP file or directory, or a list of MCAP file paths
        detect_sequence: If True, detect sequence of MCAP files (default: True).
                        Ignored if path is a list.

    Returns:
        McapSummary object containing summary information
    """
    # Handle list of files (already detected)
    if isinstance(path, list):
        files = path
    # Detect files from path string
    elif detect_sequence:
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, "*.mcap")), key=natural_sort_key)
        else:
            dirname = os.path.dirname(path)
            basename = os.path.basename(path)
            root, _ = os.path.splitext(basename)
            pattern1 = glob.glob(os.path.join(dirname, f"{root}*.mcap"))
            match = re.match(r"^(.*?)(\d+)$", root)
            if match:
                base_name = match.group(1)
                pattern2 = glob.glob(os.path.join(dirname, f"{base_name}*.mcap"))
            else:
                prefix = re.split(r"\d+", root)[0]
                pattern2 = glob.glob(os.path.join(dirname, f"{prefix}*.mcap"))
            files = sorted(set(pattern1 + pattern2), key=natural_sort_key)
    else:
        files = [path]

    if not files:
        raise FileNotFoundError(f"No MCAP files found starting from: {path}")

    summary = McapSummary(files=files)
    topics_dict: Dict[str, TopicInfo] = {}

    # Process each file
    for file_path in files:
        with open(file_path, "rb") as f:
            reader = make_reader(f)

            # Try to get header information
            if summary.header is None:
                try:
                    # Get header if available
                    header = getattr(reader, "header", None)
                    if header:
                        summary.header = {
                            "profile": getattr(header, "profile", None),
                            "library": getattr(header, "library", None),
                        }
                except Exception:
                    pass

            # Iterate through messages
            for schema, channel, message in reader.iter_messages():
                topic_name = channel.topic
                schema_name = getattr(schema, "name", "") if schema else "unknown"

                # Initialize topic info if needed
                if topic_name not in topics_dict:
                    topics_dict[topic_name] = TopicInfo(topic_name=topic_name)

                topic_info = topics_dict[topic_name]

                # Update counts and types
                topic_info.message_count += 1
                if schema_name:
                    topic_info.message_types.add(schema_name)

                # Update timestamps
                timestamp = message.log_time
                if topic_info.first_timestamp is None or timestamp < topic_info.first_timestamp:
                    topic_info.first_timestamp = timestamp
                if topic_info.last_timestamp is None or timestamp > topic_info.last_timestamp:
                    topic_info.last_timestamp = timestamp

                # Store channel info (once per topic)
                if len(topic_info.channels) == 0:
                    topic_info.channels.append(
                        {
                            "id": channel.id,
                            "schema_id": channel.schema_id,
                            "topic": channel.topic,
                            "message_encoding": getattr(channel, "message_encoding", None),
                            "metadata": (
                                dict(channel.metadata) if hasattr(channel, "metadata") else {}
                            ),
                        }
                    )

                summary.total_messages += 1

    summary.topics = topics_dict

    # Calculate overall time range
    all_first_times = [
        info.first_timestamp for info in topics_dict.values() if info.first_timestamp is not None
    ]
    all_last_times = [
        info.last_timestamp for info in topics_dict.values() if info.last_timestamp is not None
    ]

    if all_first_times:
        summary.start_time = min(all_first_times)
    if all_last_times:
        summary.end_time = max(all_last_times)

    # Calculate duration (MCAP timestamps are in nanoseconds)
    if summary.start_time is not None and summary.end_time is not None:
        summary.duration_seconds = (summary.end_time - summary.start_time) / 1e9

    return summary
