"""Gate regression: run with python -m unittest discover -s this_directory."""
import threading
import unittest

from pyslam.config import Config  # Configure paths before importing the SLAM packages.
from pyslam.slam.direct_loop_closing import DirectLoopGate


class DirectLoopGateTest(unittest.TestCase):
    def test_release_publishes_after_tracking_becomes_idle(self):
        published = []
        gate = DirectLoopGate(lambda: published.append("published"))
        self.assertEqual(gate.queue_size(), 0)
        self.assertTrue(gate.acquire_tracking())

        def close_loop():
            gate.request_stop()
            gate.wait_idle(timeout=2.0)
            gate.release()

        closer = threading.Thread(target=close_loop)
        closer.start()
        while not gate.is_stop_requested():
            closer.join(0.01)
            self.assertTrue(closer.is_alive())
        self.assertEqual(published, [])
        gate.finish_tracking()
        closer.join(2.0)
        self.assertFalse(closer.is_alive())
        self.assertEqual(published, ["published"])
        self.assertTrue(gate.acquire_tracking())
        gate.finish_tracking()

    def test_wait_idle_does_not_return_on_timeout_while_tracking(self):
        gate = DirectLoopGate(lambda: None)
        self.assertTrue(gate.acquire_tracking())
        returned = threading.Event()

        def close_loop():
            gate.request_stop()
            gate.wait_idle(timeout=0.01, print=lambda *args: None)
            returned.set()

        closer = threading.Thread(target=close_loop)
        closer.start()
        self.assertFalse(returned.wait(0.2))  # Many timeouts elapse; still waiting.
        gate.finish_tracking()
        self.assertTrue(returned.wait(2.0))
        closer.join(2.0)
        gate.release()


if __name__ == "__main__":
    unittest.main()
