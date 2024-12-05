graph LR
    Slam["Slam<br>_SLAM System_"] --> Tracking
    Slam --> FeatureTracker
    Slam --> LocalMapping["LocalMapping<br>[Thread]"]
    Slam --> LoopClosing["LoopClosing<br>[Thread]"]
    Slam --> Map
    Slam --> Camera
    Slam --> GlobalBundleAdjustment
    Slam --> VolumetricIntegrator["VolumetricIntegrator<br>_3D Volumetric Map_<br>[Process]"]
    
    Tracking --> Initializer
    Tracking --> SLAMDynamicConfig
    Tracking --> MotionModel["MotionModel<br>_Pose Prediction_"]
    
    LoopClosing --> LoopDetectingProcess["LoopDetectingProcess<br>[Process]"]
    LoopClosing --> LoopGroupConsistencyChecker["LoopGroupConsistencyChecker<br>_Loop Cluster Verification_"]
    LoopClosing --> LoopGeometryChecker["LoopGeometryChecker<br>_Geometric Validation_"]
    LoopClosing --> LoopCorrector["LoopCorrector<br>_Apply Corrections_"]
    LoopClosing --> GlobalBundleAdjustment["GlobalBundleAdjustment<br>[Process/Thread]"]
    LoopClosing --> Relocalizer