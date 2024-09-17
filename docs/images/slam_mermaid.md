graph TD;
    %% Main Slam System
    A[Slam]

    %% Submodules
    A --> B[Tracking]
    A --> C[Map]
    A --> D[LocalMapping]
    A --> E[FeatureTracker]
    A --> F[Camera]

    %% Tracking History
    B --> G[TrackingHistory]
    
    %% Groundtruth (optional)
    A --> H[Groundtruth]

    %% Tracking Dependencies
    B --> I[Initializer]
    B --> J[MotionModel]
    B --> K[SLAMDynamicConfig]
    B --> L[Pose Optimization]
    
    %% Map Points Search
    A --> M[MapPoint]
    A --> N[KeyFrame]