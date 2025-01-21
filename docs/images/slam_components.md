graph LR;
    %% Set default styles for all edges
    linkStyle default stroke:#021526,stroke-width:1px,font-size:12px;

    %% SLAM System
    classDef system fill:#,stroke:#6EACDA,stroke-width:1px,font-size: 12px;
    classDef module fill:#,stroke:#6EACDA,stroke-width:1px,font-size:12px;
    classDef component fill:#,stroke:#6EACDA,stroke-width:1px,font-size:12px;

    %% Main SLAM System
    Slam["Slam<br><span style='font-size:10px;'>_SLAM System_</span>"];
    %%Slam -->|*_has-a_*| Camera;
    Slam -->|*_has-a_*| Tracking;
    %%Slam -->|*_has-a_*| FeatureTracker;
    Slam -->|*_has-a_*| LocalMapping["LocalMapping<br><span style='font-size:10px;'><b>[Thread]</b></span>"];
    Slam -->|*_has-a_*| LoopClosing["LoopClosing<br><span style='font-size:10px;'><b>[Thread]</b></span>"];
    %%Slam -->|*_has-a_*| Map;
    Slam -->|*_has-a_*| VolumetricIntegrator["VolumetricIntegrator<br><span style='font-size:10px;'><b>[Process]</b></span>"];
    %%Slam -->|*_has-a_*| GlobalBundleAdjustment["GlobalBundleAdjustment<br><span style='font-size:10px;'><b>[Process/Thread]</b></span>"];

    Tracking -->|*_has-a_*| Initializer;
    Tracking -->|*_has-a_*| SLAMDynamicConfig;
    Tracking -->|*_has-a_*| MotionModel["MotionModel<br><span style='font-size:10px;'>_Pose Prediction_</span>"];
    Tracking -->|*_has-a_*| FeatureTracker
    %%Tracking -->|*_has-a_*| LocalMapping["LocalMapping<br><span style='font-size:10px;'><b>[Thread]</b></span>"];  
    %%Tracking -->|*_has-a_*| Camera          
    Tracking -->|*_has-a_*| Map     
    %%Tracking -->|*_has-a_*| Relocalizer

    LocalMapping -->|*_has-a_*| Map

    %%VolumetricIntegrator -->|*_has-a_*| Camera;
    VolumetricIntegrator -->|*_has-a_*| DepthEstimator; 

    LoopClosing -->|*_has-a_*| Relocalizer;
    LoopClosing -->|*_has-a_*| Map;
    LoopClosing -->|*_has-a_*| LoopDetectingProcess["LoopDetectingProcess<br><span style='font-size:10px;'><b>[Process]</b></span>"];
    LoopClosing -->|*_has-a_*| LoopGroupConsistencyChecker["LoopGroupConsistencyChecker<br><span style='font-size:10px;'>_Loop Cluster Verification_</span>"];
    LoopClosing -->|*_has-a_*| LoopGeometryChecker["LoopGeometryChecker<br><span style='font-size:10px;'>_Geometric Validation_</span>"];
    LoopClosing -->|*_has-a_*| LoopCorrector["LoopCorrector<br><span style='font-size:10px;'>_Apply Corrections_</span>"];
    LoopClosing -->|*_has-a_*| GlobalBundleAdjustment["GlobalBundleAdjustment<br><span style='font-size:10px;'><b>[Process/Thread]</b></span>"];;

    LoopDetectingProcess -->|*_has-a_*| LoopDetectorBase;

    class Slam system;

    %% Modules
    class Tracking module;
    class FeatureTracker module;
    class LocalMapping module;
    class LoopClosing module;
    class Map module;
    class Camera module;
    class GlobalBundleAdjustment module;
    class VolumetricIntegrator module;

    %% LoopClosing Components
    class LoopDetectingProcess component;
    class LoopGroupConsistencyChecker component;
    class LoopGeometryChecker component;
    class LoopCorrector component;
    class Relocalizer component;
    class LoopDetectorBase component;

    %% Tracking Components
    class Initializer component;
    class SLAMDynamicConfig component;
    class MotionModel component;

    %% Volumetric Integrator Components
    class DepthEstimator component;
