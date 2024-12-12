graph LR;
    %% Set default styles for all edges
    linkStyle default stroke:#0000FF,stroke-width:1px,font-size:10px;
    
    %% loop_detector_factory
    classDef factory fill:#f9f,stroke:#333,stroke-width:2px;
    class loop_detector_factory factory;

    %% Global Descriptor Types
    classDef descriptor fill:#f9f,stroke:#333,stroke-width:2px;
    class DBOW2 descriptor;
    class DBOW3 descriptor;
    class VLAD descriptor;
    class OBINDEX2 descriptor;
    class IBOW descriptor;
    class HDC_DELF descriptor;
    class SAD descriptor;
    class ALEXNET descriptor;
    class NETVLAD descriptor;
    class COSPLACE descriptor;
    class EIGENPLACES descriptor;

    loop_detector_factory -->|*global_descriptor_type*| DBOW2;
    loop_detector_factory -->|*global_descriptor_type*| DBOW3;
    loop_detector_factory -->|*global_descriptor_type*| VLAD;
    loop_detector_factory -->|*global_descriptor_type*| OBINDEX2;
    loop_detector_factory -->|*global_descriptor_type*| IBOW;
    loop_detector_factory -->|*global_descriptor_type*| HDC_DELF;
    loop_detector_factory -->|*global_descriptor_type*| SAD;
    loop_detector_factory -->|*global_descriptor_type*| ALEXNET;
    loop_detector_factory -->|*global_descriptor_type*| NETVLAD;
    loop_detector_factory -->|*global_descriptor_type*| COSPLACE;
    loop_detector_factory -->|*global_descriptor_type*| EIGENPLACES;

    %% LoopDetectorBase hierarchy
    classDef loopDetectorBase fill:#f9f,stroke:#333,stroke-width:2px;
    classDef loopDetectorVprBase fill:#ddf,stroke:#333,stroke-width:2px;
    class LoopDetectorBase loopDetectorBase;
    class LoopDetectorVprBase loopDetectorVprBase;

    DBOW2 -->|*creates*| LoopDetectorDBoW2;
    DBOW3 -->|*creates*| LoopDetectorDBoW3;
    VLAD -->|*creates*| LoopDetectorVlad;
    OBINDEX2 -->|*creates*| LoopDetectorOBIndex2;
    IBOW -->|*creates*| LoopDetectorIBow;

    HDC_DELF -->|*creates*| LoopDetectorHdcDelf;
    SAD -->|*creates*| LoopDetectorSad;
    ALEXNET -->|*creates*| LoopDetectorAlexNet;
    NETVLAD -->|*creates*| LoopDetectorNetVLAD;
    COSPLACE -->|*creates*| LoopDetectorCosPlace;
    EIGENPLACES -->|*creates*| LoopDetectorEigenPlaces;

    %% Hierarchical relationships
    LoopDetectorDBoW2 -->|*_is-a_*| LoopDetectorBase;
    LoopDetectorDBoW3 -->|*_is-a_*| LoopDetectorBase;
    LoopDetectorVlad -->|*_is-a_*| LoopDetectorBase;
    LoopDetectorOBIndex2 -->|*_is-a_*| LoopDetectorBase;
    LoopDetectorIBow -->|*_is-a_*| LoopDetectorBase;

    LoopDetectorHdcDelf -->|*_is-a_*| LoopDetectorVprBase;
    LoopDetectorSad -->|*_is-a_*| LoopDetectorVprBase;
    LoopDetectorAlexNet -->|*_is-a_*| LoopDetectorVprBase;
    LoopDetectorNetVLAD -->|*_is-a_*| LoopDetectorVprBase;
    LoopDetectorCosPlace -->|*_is-a_*| LoopDetectorVprBase;
    LoopDetectorEigenPlaces -->|*_is-a_*| LoopDetectorVprBase;

    LoopDetectorVprBase -->|*_is-a_*| LoopDetectorBase;
