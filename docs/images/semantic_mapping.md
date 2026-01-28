graph LR;
    %% Set default styles for all edges
    linkStyle default stroke:#021526,stroke-width:1px,font-size:14px;

    classDef factory fill:#D1F0FF,stroke:#0077B6,stroke-width:1.5px;
    classDef segmentation fill:#FFE3F1,stroke:#C71585,stroke-width:1.5px;
    classDef mapping fill:#D6FFD6,stroke:#218380,stroke-width:1.5px;

    %% Factory creates different segmentation types
    semantic_segmentation_factory -->|*SemanticSegmentationType*| DEEPLABV3;
    semantic_segmentation_factory -->|*SemanticSegmentationType*| SEGFORMER;
    semantic_segmentation_factory -->|*SemanticSegmentationType*| CLIP;
    semantic_segmentation_factory -->|*SemanticSegmentationType*| DETIC;
    semantic_segmentation_factory -->|*SemanticSegmentationType*| ODISE;
    semantic_segmentation_factory -->|*SemanticSegmentationType*| EOV_SEG;
    semantic_segmentation_factory -->|*SemanticSegmentationType*| RFDETR;
    semantic_segmentation_factory -->|*SemanticSegmentationType*| YOLO;


    %% Segmentation types create corresponding classes
    DEEPLABV3 -->|*creates*| SemanticSegmentationDeepLabV3;
    SEGFORMER -->|*creates*| SemanticSegmentationSegformer;
    CLIP -->|*creates*| SemanticSegmentationCLIP;
    DETIC -->|*creates*| SemanticSegmentationDetic;
    ODISE -->|*creates*| SemanticSegmentationOdise;
    EOV_SEG -->|*creates*| SemanticSegmentationEovSeg;
    RFDETR -->|*creates*| SemanticSegmentationRfDetr;
    YOLO -->|*creates*| SemanticSegmentationYolo;

    %% Each segmentation class --> base
    SemanticSegmentationDeepLabV3 -->|*_is-a_*| SemanticSegmentationBase;
    SemanticSegmentationSegformer -->|*_is-a_*| SemanticSegmentationBase;
    SemanticSegmentationCLIP -->|*_is-a_*| SemanticSegmentationBase;
    SemanticSegmentationDetic -->|*_is-a_*| SemanticSegmentationBase;
    SemanticSegmentationEovSeg -->|*_is-a_*| SemanticSegmentationBase;   
    SemanticSegmentationOdise -->|*_is-a_*| SemanticSegmentationBase;   
    SemanticSegmentationRfDetr -->|*_is-a_*| SemanticSegmentationBase;
    SemanticSegmentationYolo -->|*_is-a_*| SemanticSegmentationBase;

    %% Semantic mapping creation
    semantic_mapping_factory -->|*SemanticMappingType.DENSE*| SemanticMappingDense;
    semantic_mapping_factory -->|*SemanticMappingType.DENSE<br>Parameters.kMoveSemanticSegmentationToProcess=True*| SemanticMappingDenseProcess;
    SemanticMappingDense -->|*_is-a_*| SemanticMappingDenseBase;
    SemanticMappingDenseProcess -->|*_is-a_*| SemanticMappingDenseBase;
    SemanticMappingDenseBase -->|*_is-a_*| SemanticMappingBase;

    %% SemanticMappingDense uses semantic_segmentation_factory
    SemanticMappingDense -->|*uses*| semantic_segmentation_factory;
    SemanticMappingDenseProcess -->|*uses*| semantic_segmentation_factory;

    %% Factory
    class semantic_segmentation_factory factory;
    class semantic_mapping_factory factory;

    %% Segmentation models
    class DEEPLABV3 segmentation;
    class SEGFORMER segmentation;
    class CLIP segmentation;
    class DETIC segmentation;
    class EOV_SEG segmentation;
    class ODISE segmentation;
    class RFDETR segmentation;
    class YOLO segmentation;

    class SemanticSegmentationDeepLabV3 segmentation;
    class SemanticSegmentationSegformer segmentation;
    class SemanticSegmentationCLIP segmentation;
    class SemanticSegmentationDetic segmentation;
    class SemanticSegmentationEovSeg segmentation;
    class SemanticSegmentationOdise segmentation;    
    class SemanticSegmentationRfDetr segmentation;
    class SemanticSegmentationYolo segmentation;

    class SemanticSegmentationBase segmentation;

    %% Mapping classes
    class SemanticMappingBase mapping;
    class SemanticMappingDenseBase mapping;
    class SemanticMappingDense mapping;
    class SemanticMappingDenseProcess mapping;
