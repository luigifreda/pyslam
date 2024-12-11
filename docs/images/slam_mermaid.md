graph LR;
    %% Set default styles for all edges
    linkStyle default stroke:#0000FF,stroke-width:1px,font-size:10px;

    %% SLAM System
    classDef system fill:#f9f,stroke:#333,stroke-width:2px,font-size: 12px;
    classDef module fill:#ddf,stroke:#333,stroke-width:2px,font-size:12px;
    classDef component fill:#ddf,stroke:#333,stroke-width:2px,font-size:12px;

    %% Main SLAM System
    class Slam system;
    Slam["Slam<br><span style='font-size:10px;'>_SLAM System_</span>"];

    %% Modules
    class Tracking module;
    class FeatureTracker module;
    class LocalMapping module;
    class LoopClosing module;
    class Map module;
    class Camera module;
    class GlobalBundleAdjustment module;
    class VolumetricIntegrator module;

    Slam -->|*_has-a_*| Tracking;
    Slam -->|*_has-a_*| FeatureTracker;
    Slam -->|*_has-a_*| LocalMapping["LocalMapping<br><span style='font-size:10px;'><b>[Thread]</b></span>"];
    Slam -->|*_has-a_*| LoopClosing["LoopClosing<br><span style='font-size:10px;'><b>[Thread]</b></span>"];
    Slam -->|*_has-a_*| Map;
    Slam -->|*_has-a_*| Camera;
    Slam -->|*_has-a_*| GlobalBundleAdjustment["GlobalBundleAdjustment<br><span style='font-size:10px;'><b>[Process/Thread]</b></span>"];
    Slam -->|*_has-a_*| VolumetricIntegrator["VolumetricIntegrator<br><span style='font-size:10px;'>_3D Volumetric Map_ <b>[Process]</b></span>"];

    %% Tracking Components
    class Initializer component;
    class SLAMDynamicConfig component;
    class MotionModel component;

    Tracking -->|*_has-a_*| Initializer;
    Tracking -->|*_has-a_*| SLAMDynamicConfig;
    Tracking -->|*_has-a_*| MotionModel["MotionModel<br><span style='font-size:10px;'>_Pose Prediction_</span>"];

    %% LoopClosing Components
    class LoopDetectingProcess component;
    class LoopGroupConsistencyChecker component;
    class LoopGeometryChecker component;
    class LoopCorrector component;
    class Relocalizer component;

    LoopClosing -->|*_has-a_*| LoopDetectingProcess["LoopDetectingProcess<br><span style='font-size:10px;'><b>[Process]</b></span>"];
    LoopClosing -->|*_has-a_*| LoopGroupConsistencyChecker["LoopGroupConsistencyChecker<br><span style='font-size:10px;'>_Loop Cluster Verification_</span>"];
    LoopClosing -->|*_has-a_*| LoopGeometryChecker["LoopGeometryChecker<br><span style='font-size:10px;'>_Geometric Validation_</span>"];
    LoopClosing -->|*_has-a_*| LoopCorrector["LoopCorrector<br><span style='font-size:10px;'>_Apply Corrections_</span>"];
    LoopClosing -->|*_has-a_*| GlobalBundleAdjustment;
    LoopClosing -->|*_has-a_*| Relocalizer;

    LoopDetectingProcess -->|*_has-a_*| LoopDetectorBase;

    VolumetricIntegrator -->|*_has-a_*| DepthEstimator;    
