graph TD;
    %% FeatureTrackerFactory
    A[FeatureTrackerFactory]
    
    %% Single Tracker class
    A --> B[Tracker]
    
    %% FeatureManager and FeatureMatcher usage
    B --> F[FeatureManager]
    
    %% FeatureManager dependencies
    F --> G[FeatureDetector]
    F --> H[FeatureDescriptor]
    F --> I[PyramidAdaptor]
    F --> J[BlockAdaptor]
    
    %% FeatureMatcher Types
    B --> S[FeatureMatcher]