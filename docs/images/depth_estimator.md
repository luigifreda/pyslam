graph LR;
    %% Set default styles for all edges
    linkStyle default stroke:#021526,stroke-width:1px,font-size:10px;

    classDef factory fill:#D1F0FF,stroke:#0077B6,stroke-width:1.5px;
    classDef estimator_type fill:#FFFACD,stroke:#E1A100,stroke-width:1.5px;
    classDef depthEstimator fill:#EAD7F3,stroke:#6A0DAD,stroke-width:1.5px;
    classDef baseEstimator fill:#D6FFD6,stroke:#218380,stroke-width:1.5px;
    classDef dependencies fill:#F1F1F1,stroke:#888888,stroke-width:1.5px;
    classDef component fill:#FFFFFF,stroke:#6EACDA,stroke-width:1px;  

    %% depth_estimator_factory
    depth_estimator_factory -->|*depth_estimator_type*| DEPTH_ANYTHING_V2;
    depth_estimator_factory -->|*depth_estimator_type*| DEPTH_PRO;
    depth_estimator_factory -->|*depth_estimator_type*| DEPTH_SGBM;
    depth_estimator_factory -->|*depth_estimator_type*| DEPTH_RAFT_STEREO;
    depth_estimator_factory -->|*depth_estimator_type*| DEPTH_CRESTEREO_MEGENGINE;    
    depth_estimator_factory -->|*depth_estimator_type*| DEPTH_CRESTEREO_PYTORCH;  
    depth_estimator_factory -->|*depth_estimator_type*| DEPTH_MAST3R;  
    depth_estimator_factory -->|*depth_estimator_type*| DEPTH_MVDUST3R;            

    %% DepthEstimator types (final classes)
    DEPTH_ANYTHING_V2 -->|*_creates_*| DepthEstimatorDepthAnythingV2;
    DEPTH_PRO -->|*_creates_*| DepthEstimatorDepthPro;
    DEPTH_SGBM -->|*_creates_*| DepthEstimatorSgbm;
    DEPTH_RAFT_STEREO -->|*_creates_*| DepthEstimatorRaftStereo;
    DEPTH_CRESTEREO_MEGENGINE -->|*_creates_*| DepthEstimatorCrestereoMegengine;  
    DEPTH_CRESTEREO_PYTORCH -->|*_creates_*| DepthEstimatorCrestereoPytorch;
    DEPTH_MAST3R -->|*_creates_*| DepthEstimatorMast3r;  
    DEPTH_MVDUST3R -->|*_creates_*| DepthEstimatorMvdust3r;                 

    %% DepthEstimator classes
    DepthEstimator -->|*_has-a_*| camera
    DepthEstimator -->|*_has-a_*| device
    DepthEstimator -->|*_has-a_*| model        

    DepthEstimatorDepthAnythingV2 -->|*_is-a_*| DepthEstimator;
    DepthEstimatorDepthPro -->|*_is-a_*| DepthEstimator;
    DepthEstimatorSgbm -->|*_is-a_*| DepthEstimator;
    DepthEstimatorRaftStereo -->|*_is-a_*| DepthEstimator;
    DepthEstimatorCrestereoMegengine -->|*_is-a_*| DepthEstimator;
    DepthEstimatorCrestereoPytorch -->|*_is-a_*| DepthEstimator;
    DepthEstimatorMast3r -->|*_is-a_*| DepthEstimator;
    DepthEstimatorMvdust3r -->|*_is-a_*| DepthEstimator;

    %% DepthEstimator dependencies
    camera -->|*_is-a_*| Camera;

    class depth_estimator_factory factory;

    class DEPTH_ANYTHING_V2 estimator_type;
    class DEPTH_PRO estimator_type;
    class DEPTH_SGBM estimator_type;
    class DEPTH_RAFT_STEREO estimator_type;
    class DEPTH_CRESTEREO_MEGENGINE estimator_type;            
    class DEPTH_CRESTEREO_PYTORCH estimator_type;   
    class DEPTH_MAST3R estimator_type;   
    class DEPTH_MVDUST3R estimator_type;           

    class Camera dependencies;

    class DepthEstimator baseEstimator;
    
    class DepthEstimatorDepthAnythingV2 depthEstimator;
    class DepthEstimatorDepthPro depthEstimator;
    class DepthEstimatorSgbm depthEstimator;
    class DepthEstimatorRaftStereo depthEstimator;
    class DepthEstimatorCrestereoMegengine depthEstimator;
    class DepthEstimatorCrestereoPytorch depthEstimator;  
    class DepthEstimatorMast3r depthEstimator;  
    class DepthEstimatorMvdust3r depthEstimator;          

    class camera component;
    class device component;
    class model component;  