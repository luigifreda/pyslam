classDiagram
    direction TB
    class LoopClosing {
        -slam
        -last_loop_kf_id
        -loop_detecting
        -loop_consistency_checker
        -is_running
        -stop
        -work_thread
        +__init__(self, slam)
        +start(self)
        +quit(self)
        +add_keyframe(self, keyframe: KeyFrame, img)
        +run(self)
    }

    class LoopConsistencyChecker {
        -consistent_groups
        -covisibility_consistency_th
        -enough_consistent_candidates
        +__init__(self, covisibility_consistency_th=3)
        +clear_consistency_groups(self)
        +check_candidates_consistency(self, current_keyframe, candidate_keyframes)
    }

    class ConsistencyGroup {
        -keyframes
        -consistency
        +__init__(self, keyframes=None, consistency=None)
        +__str__(self)
    }

    class LoopDetectingProcess {
        -loop_detector
        -q_in
        -q_out
        -is_running
        -process
        +__init__(self, slam)
        +quit(self)
        +run(self, q_in, q_out, is_running)
        +loop_detecting(self, q_in, q_out, is_running)
        +add_task(self, task: LoopDetectorTask)
        +pop_output(self)
    }

    LoopClosing o-- LoopConsistencyChecker : "has a"
    LoopClosing o-- LoopDetectingProcess : "has a"
    LoopConsistencyChecker o-- ConsistencyGroup : "manages"