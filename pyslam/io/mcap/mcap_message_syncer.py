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

from collections import deque
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import threading
import queue as queue_module
import time


class McapMessageSyncer:
    """
    Synchronizes messages from multiple MCAP topics based on timestamps.

    This class handles the synchronization of messages across multiple topics
    by maintaining queues for each topic and finding messages that fall within
    a specified time tolerance (sync_slop).
    """

    def __init__(
        self,
        reader,
        topics: List[str],
        sync_queue_size: int = 10,
        sync_slop: float = 0.05,
    ):
        """
        Initialize the message synchronizer.

        Args:
            reader: McapReader instance
            topics: List of topic names to synchronize
            sync_queue_size: Maximum number of messages to buffer per topic
            sync_slop: Maximum time difference (in seconds) for messages to be considered synchronized
        """
        self.reader = reader
        self.topics = topics
        self.sync_queue_size = sync_queue_size
        self.sync_slop = sync_slop

        # Initialize synchronization queues for each topic
        self.message_queues = {topic: deque(maxlen=sync_queue_size) for topic in topics}

        # Iterator for reading messages
        self.message_iterator = None
        self.eof = False

    def _get_message_timestamp(self, mcap_msg) -> float:
        """Extract timestamp from MCAP ROS2 message in seconds."""
        timestamp_ns = mcap_msg.log_time
        if isinstance(timestamp_ns, datetime):
            # Convert datetime to nanoseconds
            timestamp_ns = int(timestamp_ns.timestamp() * 1e9)
        return timestamp_ns / 1e9

    def _load_messages_into_queues(self):
        """Load messages from MCAP files into synchronization queues."""
        if self.message_iterator is None:
            self.message_iterator = self.reader.iter_ros2(selected_topics=self.topics)

        # Load messages until all queues have at least some messages or all are full
        max_iterations = self.sync_queue_size * len(self.topics)  # Prevent infinite loops
        iterations = 0

        try:
            while iterations < max_iterations:
                # Check if all queues are full
                if all(len(q) >= self.sync_queue_size for q in self.message_queues.values()):
                    break

                path, m = next(self.message_iterator)
                topic = m.channel.topic
                timestamp = self._get_message_timestamp(m)

                # Add to queue if not full
                if len(self.message_queues[topic]) < self.sync_queue_size:
                    self.message_queues[topic].append((timestamp, m))

                iterations += 1
        except StopIteration:
            self.eof = True

    def get_next_synced(
        self,
        required_topics: Optional[List[str]] = None,
        timeout: Optional[float] = None,  # Ignored for sync syncer, kept for API compatibility
    ) -> Optional[Tuple[float, Dict[str, Any]]]:
        """
        Get the next synchronized set of messages.

        Args:
            required_topics: List of topics that must be present in the synchronized set.
                           If None, uses all topics.
            timeout: Ignored for sync syncer (kept for API compatibility with async syncer).

        Returns:
            Tuple of (timestamp, dict of topic -> message) if synchronization found,
            None if no synchronized set available or EOF reached.
        """
        if required_topics is None:
            required_topics = self.topics

        # Try multiple times to find synchronized messages
        max_attempts = 10
        for attempt in range(max_attempts):
            # Ensure queues have some messages
            if all(len(q) == 0 for q in self.message_queues.values()):
                if not self.eof:
                    self._load_messages_into_queues()
                if all(len(q) == 0 for q in self.message_queues.values()):
                    return None

            # Get earliest timestamp from required topics
            earliest_timestamps = []
            for topic in required_topics:
                if len(self.message_queues[topic]) > 0:
                    earliest_timestamps.append(self.message_queues[topic][0][0])

            if not earliest_timestamps:
                # No messages in required topics, try loading more
                if not self.eof:
                    self._load_messages_into_queues()
                    continue
                else:
                    return None

            ref_timestamp = min(earliest_timestamps)

            # Find messages within sync_slop of reference timestamp
            synced = {}
            for topic in self.topics:
                queue = self.message_queues[topic]
                if len(queue) == 0:
                    continue

                # Find best matching message within sync_slop
                best_match = None
                best_idx = -1
                for idx, (ts, msg) in enumerate(queue):
                    if abs(ts - ref_timestamp) <= self.sync_slop:
                        if best_match is None or abs(ts - ref_timestamp) < abs(
                            best_match[0] - ref_timestamp
                        ):
                            best_match = (ts, msg)
                            best_idx = idx

                if best_match is not None:
                    synced[topic] = best_match[1]
                    # Remove matched message and all earlier messages from queue
                    for _ in range(best_idx + 1):
                        if len(queue) > 0:
                            queue.popleft()

            # Check if we have all required topics
            if all(topic in synced for topic in required_topics):
                return (ref_timestamp, synced)

            # Synchronization failed - if we don't have enough messages, load more
            # Check if any required topic queue is empty or has very few messages
            min_queue_size = min(len(self.message_queues[topic]) for topic in required_topics)
            if min_queue_size < 3 and not self.eof:
                # Load more messages and try again
                self._load_messages_into_queues()
                continue

            # If we have messages but can't synchronize, return None
            # (messages might be too far apart in time)
            return None

        # Exhausted attempts
        return None

    def is_eof(self) -> bool:
        """Check if end of file has been reached."""
        return self.eof

    def reset(self):
        """Reset the synchronizer to start from the beginning."""
        self.message_iterator = None
        self.eof = False
        self.message_queues = {topic: deque(maxlen=self.sync_queue_size) for topic in self.topics}


class McapMessageAsyncSyncer:
    """
    Asynchronous version of McapMessageSyncer that uses a background thread
    to continuously read messages and fill queues, avoiding read bursts.

    This class implements a producer-consumer pattern where:
    - Producer thread: Continuously reads messages from MCAP files
    - Consumer (main thread): Retrieves synchronized message sets

    Similar to Ros2BagAsyncReaderATS in C++, this avoids blocking I/O
    operations in the main thread and provides smoother performance.
    """

    def __init__(
        self,
        reader,
        topics: List[str],
        sync_queue_size: int = 10,
        sync_slop: float = 0.05,
        max_read_ahead: int = 100,
    ):
        """
        Initialize the async message synchronizer.

        Args:
            reader: McapReader instance
            topics: List of topic names to synchronize
            sync_queue_size: Maximum number of messages to buffer per topic
            sync_slop: Maximum time difference (in seconds) for messages to be considered synchronized
            max_read_ahead: Maximum number of synchronized message sets to buffer
        """
        self.reader = reader
        self.topics = topics
        self.sync_queue_size = sync_queue_size
        self.sync_slop = sync_slop
        self.max_read_ahead = max_read_ahead

        # Thread-safe queues for each topic (raw messages from MCAP)
        self.message_queues = {topic: deque(maxlen=sync_queue_size) for topic in topics}
        self.queue_locks = {topic: threading.Lock() for topic in topics}
        # Condition variables to wait when queues are full
        self.queue_not_full_cvs = {
            topic: threading.Condition(self.queue_locks[topic]) for topic in topics
        }

        # Queue for synchronized message sets (producer-consumer buffer)
        self.synced_queue = queue_module.Queue(maxsize=max_read_ahead)
        # Condition variable for synced_queue (to wait when full)
        self.synced_queue_lock = threading.Lock()
        self.synced_queue_not_full_cv = threading.Condition(self.synced_queue_lock)

        # Thread control
        self.reader_thread = None
        self.stop_event = threading.Event()
        self.eof = False
        self.eof_lock = threading.Lock()

        # Iterator for reading messages
        self.message_iterator = None

        # Start the background reader thread
        self._start_reader_thread()

    def _get_message_timestamp(self, mcap_msg) -> float:
        """Extract timestamp from MCAP ROS2 message in seconds."""
        timestamp_ns = mcap_msg.log_time
        if isinstance(timestamp_ns, datetime):
            # Convert datetime to nanoseconds
            timestamp_ns = int(timestamp_ns.timestamp() * 1e9)
        return timestamp_ns / 1e9

    def _find_synced_messages(
        self, required_topics: Optional[List[str]] = None
    ) -> Optional[Tuple[float, Dict[str, Any]]]:
        """
        Find synchronized messages from the current message queues.
        This is called by the reader thread to create synchronized sets.

        Args:
            required_topics: List of topics that must be present. If None, uses all topics.

        Note: This method acquires all locks simultaneously to get a consistent
        snapshot of all queues. This prevents race conditions but requires
        careful lock ordering to avoid deadlocks.
        """
        if required_topics is None:
            required_topics = self.topics

        # Acquire all locks in a consistent order (sorted by topic name)
        # to avoid deadlocks
        sorted_topics = sorted(self.topics)
        locks = [self.queue_locks[topic] for topic in sorted_topics]

        # Acquire all locks
        for lock in locks:
            lock.acquire()

        try:
            # Get earliest timestamp from required topics (now we have consistent snapshot)
            earliest_timestamps = []
            for topic in required_topics:
                if len(self.message_queues[topic]) > 0:
                    earliest_timestamps.append(self.message_queues[topic][0][0])

            if not earliest_timestamps:
                return None

            # Ensure we have messages in ALL required topics
            if not all(len(self.message_queues[topic]) > 0 for topic in required_topics):
                return None

            ref_timestamp = min(earliest_timestamps)

            # Find messages within sync_slop of reference timestamp
            synced = {}
            for topic in self.topics:
                queue = self.message_queues[topic]
                if len(queue) == 0:
                    continue

                # Find best matching message within sync_slop
                best_match = None
                best_idx = -1
                for idx, (ts, msg) in enumerate(queue):
                    if abs(ts - ref_timestamp) <= self.sync_slop:
                        if best_match is None or abs(ts - ref_timestamp) < abs(
                            best_match[0] - ref_timestamp
                        ):
                            best_match = (ts, msg)
                            best_idx = idx

                if best_match is not None:
                    synced[topic] = best_match[1]
                    # Remove matched message and all earlier messages from queue
                    removed_count = 0
                    for _ in range(best_idx + 1):
                        if len(queue) > 0:
                            queue.popleft()
                            removed_count += 1

                    # Notify that queue is not full (if we removed messages)
                    if removed_count > 0:
                        # Note: We're still holding the lock, so we can notify
                        # The condition variable is associated with the lock
                        pass  # Will notify after releasing locks

            # Check if we have all required topics
            if all(topic in synced for topic in required_topics):
                # Notify condition variables that queues are not full (while holding locks)
                # Track which topics had messages removed
                topics_with_removals = set()
                for topic in self.topics:
                    if topic in synced:
                        topics_with_removals.add(topic)

                # Notify condition variables while still holding locks
                for topic in topics_with_removals:
                    if topic in self.queue_not_full_cvs:
                        self.queue_not_full_cvs[topic].notify_all()

                return (ref_timestamp, synced)

            # Debug: log why synchronization failed
            if True:  # Set to True for debugging
                missing_topics = [topic for topic in required_topics if topic not in synced]
                queue_sizes = {topic: len(self.message_queues[topic]) for topic in required_topics}
                if missing_topics:
                    print(
                        f"[_find_synced_messages] Failed: missing topics {missing_topics}, "
                        f"queue_sizes: {queue_sizes}, ref_timestamp: {ref_timestamp}"
                    )
                    # Show timestamps of first messages in each queue
                    for topic in required_topics:
                        if len(self.message_queues[topic]) > 0:
                            first_ts = self.message_queues[topic][0][0]
                            diff = abs(first_ts - ref_timestamp)
                            print(
                                f"  {topic}: first_ts={first_ts:.6f}, diff={diff:.6f}, "
                                f"within_slop={diff <= self.sync_slop}"
                            )

            return None
        finally:
            # Release all locks in reverse order
            for lock in reversed(locks):
                lock.release()

    def _reader_thread_function(self):
        """Background thread function that continuously reads and synchronizes messages."""
        if self.message_iterator is None:
            self.message_iterator = self.reader.iter_ros2(selected_topics=self.topics)

        messages_read = 0
        synced_sets_created = 0

        try:
            while not self.stop_event.is_set():
                # Read messages and fill queues
                try:
                    path, m = next(self.message_iterator)
                    topic = m.channel.topic
                    timestamp = self._get_message_timestamp(m)
                    messages_read += 1

                    # Add to topic-specific queue (thread-safe)
                    # Wait if queue is full (producer-consumer pattern)
                    with self.queue_not_full_cvs[topic]:
                        # Wait while queue is full
                        while (
                            len(self.message_queues[topic]) >= self.sync_queue_size
                            and not self.stop_event.is_set()
                        ):
                            self.queue_not_full_cvs[topic].wait(timeout=0.1)

                        if self.stop_event.is_set():
                            break

                        # Add message to queue
                        self.message_queues[topic].append((timestamp, m))
                        # Notify that queue is not empty (for potential consumers)

                    # Try to find synchronized message sets for all topics
                    # _find_synced_messages() will check if we have messages in all queues with proper locking
                    synced_result = self._find_synced_messages(required_topics=self.topics)
                    if synced_result is not None:
                        synced_sets_created += 1
                        # Wait if synced_queue is full (producer-consumer pattern)
                        with self.synced_queue_not_full_cv:
                            # Wait while queue is full
                            while self.synced_queue.full() and not self.stop_event.is_set():
                                self.synced_queue_not_full_cv.wait(timeout=0.1)

                            if self.stop_event.is_set():
                                break

                            # Add synchronized set to queue
                            try:
                                self.synced_queue.put_nowait(synced_result)
                                # Notify that queue is not empty (for consumers)
                                self.synced_queue_not_full_cv.notify_all()
                            except queue_module.Full:
                                # Should not happen due to condition variable wait
                                pass

                except StopIteration:
                    # EOF reached
                    print(
                        f"[McapMessageAsyncSyncer] EOF reached. Messages read: {messages_read}, Synced sets created: {synced_sets_created}"
                    )
                    with self.eof_lock:
                        self.eof = True
                    break
                except Exception as e:
                    print(f"[McapMessageAsyncSyncer] Error in reader thread: {e}")
                    import traceback

                    traceback.print_exc()
                    break

        except Exception as e:
            print(f"[McapMessageAsyncSyncer] Fatal error in reader thread: {e}")
            import traceback

            traceback.print_exc()
            with self.eof_lock:
                self.eof = True

    def _start_reader_thread(self):
        """Start the background reader thread."""
        if self.reader_thread is not None and self.reader_thread.is_alive():
            return  # Thread already running

        self.stop_event.clear()
        with self.eof_lock:
            self.eof = False
        self.reader_thread = threading.Thread(target=self._reader_thread_function, daemon=True)
        self.reader_thread.start()
        print(
            f"[McapMessageAsyncSyncer] Started background reader thread for topics: {self.topics}"
        )
        # Give the thread a moment to start reading messages
        time.sleep(0.01)

    def _stop_reader_thread(self):
        """Stop the background reader thread."""
        if self.reader_thread is None:
            return

        self.stop_event.set()
        if self.reader_thread.is_alive():
            self.reader_thread.join(timeout=1.0)
        self.reader_thread = None

    def get_next_synced(
        self,
        required_topics: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ) -> Optional[Tuple[float, Dict[str, Any]]]:
        """
        Get the next synchronized set of messages.

        Args:
            required_topics: List of topics that must be present in the synchronized set.
                           If None, uses all topics. The synchronized set may contain
                           more topics than required_topics.
            timeout: Optional timeout in seconds. If None, returns immediately (non-blocking).
                    If specified, will wait up to timeout seconds for a message.

        Returns:
            Tuple of (timestamp, dict of topic -> message) if synchronization found,
            None if no synchronized set available or EOF reached.
        """
        if required_topics is None:
            required_topics = self.topics

        # Try to get a synchronized message set from the queue
        try:
            if timeout is None:
                # Non-blocking
                result = self.synced_queue.get_nowait()
                # Notify producer that queue is not full
                with self.synced_queue_not_full_cv:
                    self.synced_queue_not_full_cv.notify_all()
            else:
                # Blocking with timeout
                result = self.synced_queue.get(timeout=timeout)
                # Notify producer that queue is not full
                with self.synced_queue_not_full_cv:
                    self.synced_queue_not_full_cv.notify_all()
        except queue_module.Empty:
            # Queue is empty, check if we're at EOF
            with self.eof_lock:
                if self.eof:
                    return None
            # Not EOF yet, but no synchronized sets available
            # This can happen if synchronization hasn't found matches yet
            # Check queue sizes for debugging
            if False:  # Set to True for debugging
                with self.queue_locks.get(
                    self.topics[0] if self.topics else None, threading.Lock()
                ):
                    queue_sizes = {topic: len(self.message_queues[topic]) for topic in self.topics}
                    print(
                        f"[McapMessageAsyncSyncer] Queue empty, queue sizes: {queue_sizes}, synced_queue size: {self.synced_queue.qsize()}"
                    )
            return None

        if result is None:
            return None

        # Filter result to only include required topics if a subset was requested
        ts, synced = result
        if set(required_topics) != set(self.topics):
            # Check if all required topics are present
            if all(topic in synced for topic in required_topics):
                # Return filtered result with only required topics
                filtered_synced = {topic: synced[topic] for topic in required_topics}
                return (ts, filtered_synced)
            else:
                # Not all required topics present, try again (non-blocking)
                # Put the result back (but this might fail if queue is full)
                try:
                    self.synced_queue.put_nowait(result)
                except queue_module.Full:
                    pass  # Queue is full, drop this result
                return None

        # All topics requested, return as-is
        return result

    def is_eof(self) -> bool:
        """Check if end of file has been reached."""
        with self.eof_lock:
            return self.eof and self.synced_queue.empty()

    def reset(self):
        """Reset the synchronizer to start from the beginning."""
        self._stop_reader_thread()
        self.message_iterator = None
        with self.eof_lock:
            self.eof = False
        self.message_queues = {topic: deque(maxlen=self.sync_queue_size) for topic in self.topics}
        # Clear the synchronized queue
        while not self.synced_queue.empty():
            try:
                self.synced_queue.get_nowait()
            except queue_module.Empty:
                break
        self._start_reader_thread()

    def __del__(self):
        """Cleanup: stop the reader thread."""
        self._stop_reader_thread()
