graph LR;
    %% FeatureTrackerFactory
    FeatureTrackerFactory[FeatureTrackerFactory]
    
    %% Single Tracker class
    FeatureTrackerFactory --> FeatureTracker[FeatureTracker]
    
    %% FeatureManager and FeatureMatcher usage
    FeatureTracker --> FeatureManager[FeatureManager]
    
    %% FeatureManager dependencies
    FeatureManager --> FeatureDetector[FeatureDetector]
    FeatureManager --> FeatureDescriptor[FeatureDescriptor]
    FeatureManager --> PyramidAdaptor[PyramidAdaptor]
    FeatureManager --> BlockAdaptor[BlockAdaptor]
    
    %% FeatureMatcher Types
    FeatureTracker --> FeatureMatcher[FeatureMatcher]