graph LR;
    %% Set default styles for all edges
    linkStyle default stroke:#0000FF,stroke-width:1px,font-size:10px;

    %% feature_tracker_factory
    classDef factory fill:#f9f,stroke:#333,stroke-width:2px
    class feature_tracker_factory factory;
    
    %% FeatureTracker types
    classDef tracker fill:#f9f,stroke:#333,stroke-width:2px
    class LK tracker;
    class DES_BF tracker;
    class DES_FLANN tracker;
    class XFEAT tracker;
    class LIGHTGLUE tracker;
    class LOFTR tracker;
    
    feature_tracker_factory -->|*tracker_type*| LK;
    feature_tracker_factory -->|*tracker_type*| DES_BF;
    feature_tracker_factory -->|*tracker_type*| DES_FLANN;
    feature_tracker_factory -->|*tracker_type*| XFEAT;
    feature_tracker_factory -->|*tracker_type*| LIGHTGLUE;
    feature_tracker_factory -->|*tracker_type*| LOFTR;
    
    %% Single Tracker class
    classDef singleTracker fill:#f9f,stroke:#333,stroke-width:2px
    class LKFeatureTracker singleTracker;
    class DescriptorFeatureTracker singleTracker;
    class XFeatureTracker singleTracker;
    class LightGlueFeatureTracker singleTracker;
    class LoFTRFeatureTracker singleTracker;
    
    LK -->|*creates*| LKFeatureTracker;
    DES_BF -->|*creates*| DescriptorFeatureTracker;
    DES_FLANN -->|*creates*| DescriptorFeatureTracker;
    XFEAT -->|*creates*| XFeatureTracker;
    LIGHTGLUE -->|*creates*| LightGlueFeatureTracker;
    LOFTR -->|*creates*| LoFTRFeatureTracker;
    
    LKFeatureTracker -->|*_is-a_*| FeatureTracker;
    DescriptorFeatureTracker -->|*_is-a_*| FeatureTracker;
    XFeatureTracker -->|*_is-a_*| FeatureTracker;
    LightGlueFeatureTracker -->|*_is-a_*| FeatureTracker;
    LoFTRFeatureTracker -->|*_is-a_*| FeatureTracker;
    
    %% FeatureTracker relationships
    classDef featureTracker fill:#f9f,stroke:#333,stroke-width:2px
    class FeatureTracker featureTracker;
    class FeatureManager featureTracker;
    class FeatureDetector featureTracker;
    class FeatureDescriptor featureTracker;
    class PyramidAdaptor featureTracker;
    class BlockAdaptor featureTracker;
    class FeatureMatcher featureTracker;
    
    FeatureTracker -->|*_has-a_*| FeatureManager;
    FeatureTracker -->|*_has-a_*| FeatureDetector;
    FeatureTracker -->|*_has-a_*| FeatureDescriptor;
    FeatureTracker -->|*_has-a_*| PyramidAdaptor;
    FeatureTracker -->|*_has-a_*| BlockAdaptor;
    FeatureTracker -->|*_has-a_*| FeatureMatcher;
    
    %% FeatureManager dependencies
    FeatureManager -->|*_has-a_*| FeatureDetector;
    FeatureManager -->|*_has-a_*| FeatureDescriptor;
    FeatureManager -->|*_has-a_*| PyramidAdaptor;
    FeatureManager -->|*_has-a_*| BlockAdaptor;
