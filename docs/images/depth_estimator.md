graph LR;
    %% Set default styles for all edges
    linkStyle default stroke:#021526,stroke-width:1px,font-size:10px;

    classDef factory fill:#,stroke:#6EACDA,stroke-width:1px;
    classDef type fill:#,stroke:#6EACDA,stroke-width:1px;
    classDef estimator_type fill:#,stroke:#6EACDA,stroke-width:1px;
    classDef dependencies fill:#,stroke:#6EACDA,stroke-width:1px;
    classDef depthEstimator fill:#,stroke:#6EACDA,stroke-width:1px;
    classDef component fill:#,stroke:#6EACDA,stroke-width:1px;    

    %% depth_estimator_factory
    depth_estimator_factory -->|*depth_estimator_type*| DEPTH_ANYTHING_V2;
    depth_estimator_factory -->|*depth_estimator_type*| DEPTH_PRO;
    depth_estimator_factory -->|*depth_estimator_type*| DEPTH_SGBM;
    depth_estimator_factory -->|*depth_estimator_type*| DEPTH_RAFT_STEREO;
    depth_estimator_factory -->|*depth_estimator_type*| DEPTH_CRESTEREO;    
    depth_estimator_factory -->|*depth_estimator_type*| DEPTH_CRESTEREO_PYTORCH;    

    %% DepthEstimator types (final classes)
    DEPTH_ANYTHING_V2 -->|*_creates_*| DepthEstimatorDepthAnythingV2;
    DEPTH_PRO -->|*_creates_*| DepthEstimatorDepthPro;
    DEPTH_SGBM -->|*_creates_*| DepthEstimatorSgbm;
    DEPTH_RAFT_STEREO -->|*_creates_*| DepthEstimatorRaftStereo;
    DEPTH_CRESTEREO -->|*_creates_*| DepthEstimatorCrestereo;  
    DEPTH_CRESTEREO_PYTORCH -->|*_creates_*| DepthEstimatorCrestereoPytorch;            

    %% DepthEstimator classes
    DepthEstimator -->|*_has-a_*| camera
    DepthEstimator -->|*_has-a_*| device
    DepthEstimator -->|*_has-a_*| model        

    DepthEstimatorDepthAnythingV2 -->|*_is-a_*| DepthEstimator;
    DepthEstimatorDepthPro -->|*_is-a_*| DepthEstimator;
    DepthEstimatorSgbm -->|*_is-a_*| DepthEstimator;
    DepthEstimatorRaftStereo -->|*_is-a_*| DepthEstimator;
    DepthEstimatorCrestereo -->|*_is-a_*| DepthEstimator;
    DepthEstimatorCrestereoPytorch -->|*_is-a_*| DepthEstimator;

    %% DepthEstimator dependencies
    camera -->|*_is-a_*| Camera;

    class depth_estimator_factory factory;

    class DEPTH_ANYTHING_V2 estimator_type;
    class DEPTH_PRO estimator_type;
    class DEPTH_SGBM estimator_type;
    class DEPTH_RAFT_STEREO estimator_type;
    class DEPTH_CRESTEREO estimator_type;            
    class DEPTH_CRESTEREO_PYTORCH estimator_type;   

    class Camera dependencies;

    class DepthEstimator depthEstimator;
    class DepthEstimatorDepthAnythingV2 depthEstimator;
    class DepthEstimatorDepthPro depthEstimator;
    class DepthEstimatorSgbm depthEstimator;
    class DepthEstimatorRaftStereo depthEstimator;
    class DepthEstimatorCrestereo depthEstimator;
    class DepthEstimatorCrestereoPytorch depthEstimator;  

    class camera component;
    class device component;
    class model component;  