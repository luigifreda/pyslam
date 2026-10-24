"""Optional main-thread playback timings, including rendering submissions and pacing."""

import json
from pathlib import Path
import time

import numpy as np


class PlaybackTiming:
    def __init__(self, filename, metadata=None, warmup_frames=30, clock=time.perf_counter):
        self.filename = Path(filename)
        self.metadata = metadata or {}
        self.warmup_frames = warmup_frames
        self.clock = clock
        self.samples = []
        self.current = None

    def begin(self, frame_id):
        self.started = self.previous = self.clock()
        self.current = {"frame_id": frame_id}

    def mark(self, stage):
        if self.current is None:
            return
        now = self.clock()
        self.current[stage] = self.current.get(stage, 0.) + 1000 * (now - self.previous)
        self.previous = now

    def finish(self):
        if self.current is None:
            return
        self.mark("other")
        self.current["total"] = 1000 * (self.previous - self.started)
        self.samples.append(self.current)
        self.current = None

    def discard(self):
        self.current = None

    @staticmethod
    def summarize(samples):
        if not samples:
            return {"frames": 0}
        stages = sorted(set().union(*(row.keys() for row in samples)) - {"frame_id"})
        summary = {}
        for stage in stages:
            values = [row.get(stage, 0.) for row in samples]
            summary[stage] = dict(mean=float(np.mean(values)), median=float(np.median(values)),
                                  p95=float(np.percentile(values, 95)))
        mean_total = summary["total"]["mean"]
        return dict(frames=len(samples), fps=1000 / mean_total if mean_total > 0 else None,
                    stages_ms=summary)

    def write(self):
        report = dict(metadata=self.metadata, warmup_frames=self.warmup_frames,
                      all_frames=self.summarize(self.samples),
                      after_warmup=self.summarize(self.samples[self.warmup_frames:]),
                      samples_ms=self.samples)
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.filename.with_name(self.filename.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(self.filename)
