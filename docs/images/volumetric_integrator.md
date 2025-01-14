graph LR;
    %% Set default styles for all edges
    linkStyle default stroke:#6EACDA,stroke-width:1px,font-size:10px;


    classDef factory fill:#,stroke:#6EACDA,stroke-width:1px;
    classDef dependencies fill:#,stroke:#6EACDA,stroke-width:1px;
    classDef components fill:#,stroke:#6EACDA,stroke-width:1px;           
    classDef type fill:#,stroke:#6EACDA,stroke-width:1px;
    classDef volumetric_integrator fill:#,stroke:#6EACDA,stroke-width:1px;

    volumetric_integrator_factory -->|*volumetric_integrator_type*| TSDF;
    volumetric_integrator_factory -->|*volumetric_integrator_type*| GAUSSIAN_SPLATTING; 

    %% types

    TSDF -->|*_creates_*| VolumetricIntegratorTSDF;
    GAUSSIAN_SPLATTING -->|*_creates_*| VolumetricIntegratorGaussianSplatting;        

    VolumetricIntegratorTSDF -->|*_is-a_*| VolumetricIntegratorBase;
    VolumetricIntegratorGaussianSplatting -->|*_is-a_*| VolumetricIntegratorBase;

    %% VolumetricIntegratorBase classes

    VolumetricIntegratorBase -->|*_has-a_*| camera
    VolumetricIntegratorBase -->|*_has-a_*| keyframe_queue
    VolumetricIntegratorBase -->|*_has-a_*| volume        

    camera -->|*_is-a_*| Camera;


    class TSDF type;
    class GAUSSIAN_SPLATTING type;

    class volumetric_integrator_factory factory;

    class VolumetricIntegratorTSDF volumetric_integrator;
    class VolumetricIntegratorGaussianSplatting volumetric_integrator;
    class VolumetricIntegratorBase volumetric_integrator;

    class camera components;
    class keyframe_queue components;
    class volume components;
    
    class Camera dependencies;

    class VolumetricIntegratorBase volumetricIntegrator;