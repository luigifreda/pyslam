graph LR;
    %% Set default styles for all edges
    linkStyle default stroke:#021526,stroke-width:1px,font-size:10px;
    
    classDef factory fill:#D1F0FF,stroke:#0077B6,stroke-width:1.5px;
    classDef descriptor fill:#FFFACD,stroke:#E1A100,stroke-width:1.5px;
    classDef classicalDetector fill:#EAD7F3,stroke:#6A0DAD,stroke-width:1.5px;
    classDef vprDetector fill:#FFE3F1,stroke:#C71585,stroke-width:1.5px;
    classDef loopDetectorBase fill:#D6FFD6,stroke:#218380,stroke-width:1.5px;
    classDef loopDetectorVprBase fill:#D6FFD6,stroke:#218380,stroke-width:1.5px;

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
    loop_detector_factory -->|*global_descriptor_type*| MEGALOC;

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
    MEGALOC -->|*creates*| LoopDetectorMegaloc;

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
    LoopDetectorMegaloc -->|*_is-a_*| LoopDetectorVprBase;

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
    class MEGALOC descriptor;

    %% LoopDetectorBase hierarchy
    class LoopDetectorBase loopDetectorBase;
    class LoopDetectorVprBase loopDetectorVprBase;

    class LoopDetectorDBoW2 classicalDetector;
    class LoopDetectorDBoW3 classicalDetector; 
    class LoopDetectorVlad classicalDetector; 
    class LoopDetectorOBIndex2 classicalDetector; 
    class LoopDetectorIBow classicalDetector; 

    class LoopDetectorHdcDelf vprDetector; 
    class LoopDetectorSad vprDetector; 
    class LoopDetectorAlexNet vprDetector; 
    class LoopDetectorNetVLAD vprDetector; 
    class LoopDetectorCosPlace vprDetector; 
    class LoopDetectorEigenPlaces vprDetector;
    class LoopDetectorMegaloc vprDetector;