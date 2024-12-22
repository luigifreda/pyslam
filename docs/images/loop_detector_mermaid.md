graph LR;
    %% Set default styles for all edges
    linkStyle default stroke:#021526,stroke-width:1px,font-size:10px;
    
    classDef factory fill:#,stroke:#6EACDA,stroke-width:1px;
    classDef descriptor fill:#,stroke:#6EACDA,stroke-width:1px;
    classDef loopDetectorBase fill:#,stroke:#6EACDA,stroke-width:1px;
    classDef loopDetectorVprBase fill:#,stroke:#6EACDA,stroke-width:1px;

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

    %% loop_detector_factory
    class loop_detector_factory factory;

    %% Global Descriptor Types
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

    %% LoopDetectorBase hierarchy
    class LoopDetectorBase loopDetectorBase;
    class LoopDetectorVprBase loopDetectorVprBase;

    class LoopDetectorDBoW2 loopDetectorBase;
    class LoopDetectorDBoW3 loopDetectorBase; 
    class LoopDetectorVlad loopDetectorBase; 
    class LoopDetectorOBIndex2 loopDetectorBase; 
    class LoopDetectorIBow loopDetectorBase; 

    class LoopDetectorHdcDelf loopDetectorBase; 
    class LoopDetectorSad loopDetectorBase; 
    class LoopDetectorAlexNet loopDetectorBase; 
    class LoopDetectorNetVLAD loopDetectorBase; 
    class LoopDetectorCosPlace loopDetectorBase; 
    class LoopDetectorEigenPlaces loopDetectorBase; 