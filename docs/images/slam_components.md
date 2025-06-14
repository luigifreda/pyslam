graph LR;
    %% Set default styles for all edges
    linkStyle default stroke:#021526,stroke-width:1px,font-size:16px;

    %% SLAM System
    classDef system fill:#ECECEC,stroke:#333333,stroke-width:1.5px,font-size:16px;
    classDef moduleTracking fill:#FFD6D6,stroke:#D7263D,stroke-width:1.5px,font-size:16px;
    classDef moduleLocalMapping fill:#D6FFD6,stroke:#218380,stroke-width:1.5px,font-size:16px;
    classDef moduleLoopClosing fill:#D6E4FF,stroke:#00509D,stroke-width:1.5px,font-size:16px;
    classDef moduleVolumetric fill:#FFFACD,stroke:#E1A100,stroke-width:1.5px,font-size:16px;
    classDef moduleSemantic fill:#D1F7C4,stroke:#2E8B57,stroke-width:1.5px,font-size:16px;
    classDef moduleGlobalBA fill:#EAD7F3,stroke:#6A0DAD,stroke-width:1.5px,font-size:16px;
    classDef moduleMap fill:#F1F1F1,stroke:#555555,stroke-width:1.5px,font-size:16px;
    classDef moduleFeatureTracker fill:#FDEEDC,stroke:#FB8500,stroke-width:1.5px,font-size:16px;

    classDef component fill:none,stroke:#6EACDA,stroke-width:1px,font-size:16px;

    %% Main SLAM System
    Slam["Slam<br><span style='font-size:14px;'>_SLAM System_</span>"];
    %%Slam -->|*_has-a_*| Camera;
    Slam -->|*_has-a_*| Tracking;
    %%Slam -->|*_has-a_*| FeatureTracker;
    Slam -->|*_has-a_*| LocalMapping["LocalMapping<br><span style='font-size:14px;'><b>[Thread]</b></span>"];
    Slam -->|*_has-a_*| LoopClosing["LoopClosing<br><span style='font-size:14px;'><b>[Thread]</b></span>"];
    %%Slam -->|*_has-a_*| Map;
    Slam -->|*_has-a_*| VolumetricIntegrator["VolumetricIntegrator<br><span style='font-size:14px;'><b>[Process]</b></span>"];
    %%Slam -->|*_has-a_*| GlobalBundleAdjustment["GlobalBundleAdjustment<br><span style='font-size:14px;'><b>[Process/Thread]</b></span>"];
    Slam -->|*_has-a_*| SemanticMapping["SemanticMappingDense<br><span style='font-size:14px;'><b>[Process]</b></span>"];

    Tracking -->|*_has-a_*| Initializer;
    Tracking -->|*_has-a_*| SLAMDynamicConfig;
    Tracking -->|*_has-a_*| MotionModel["MotionModel<br><span style='font-size:14px;'>_Pose Prediction_</span>"];
    Tracking -->|*_has-a_*| FeatureTracker
    %%Tracking -->|*_has-a_*| LocalMapping["LocalMapping<br><span style='font-size:14px;'><b>[Thread]</b></span>"];  
    %%Tracking -->|*_has-a_*| Camera          
    Tracking -->|*_has-a_*| Map     
    %%Tracking -->|*_has-a_*| Relocalizer

    LocalMapping -->|*_has-a_*| Map

    %%VolumetricIntegrator -->|*_has-a_*| Camera;
    VolumetricIntegrator -->|*_has-a_*| DepthEstimator; 

    SemanticMapping -->|*_has-a_*| SemanticSegmentation; 

    LoopClosing -->|*_has-a_*| Relocalizer;
    LoopClosing -->|*_has-a_*| Map;
    LoopClosing -->|*_has-a_*| LoopDetectingProcess["LoopDetectingProcess<br><span style='font-size:14px;'><b>[Process]</b></span>"];
    LoopClosing -->|*_has-a_*| LoopGroupConsistencyChecker["LoopGroupConsistencyChecker<br><span style='font-size:14px;'>_Loop Cluster Verification_</span>"];
    LoopClosing -->|*_has-a_*| LoopGeometryChecker["LoopGeometryChecker<br><span style='font-size:14px;'>_Geometric Validation_</span>"];
    LoopClosing -->|*_has-a_*| LoopCorrector["LoopCorrector<br><span style='font-size:14px;'>_Apply Corrections_</span>"];
    LoopClosing -->|*_has-a_*| GlobalBundleAdjustment["GlobalBundleAdjustment<br><span style='font-size:14px;'><b>[Process/Thread]</b></span>"];;

    LoopDetectingProcess -->|*_has-a_*| LoopDetectorBase;

    class Slam system;

    %% Modules
    class Tracking moduleTracking;
    class LocalMapping moduleLocalMapping;
    class LoopClosing moduleLoopClosing;
    class VolumetricIntegrator moduleVolumetric;
    class SemanticMapping moduleSemantic;
    class GlobalBundleAdjustment moduleGlobalBA;
    class Map moduleMap;
    class FeatureTracker moduleFeatureTracker;

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

    %% Semantic Mapping Components
    class SemanticSegmentation component;