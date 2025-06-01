graph LR;
    %% Set default styles for all edges
    linkStyle default stroke:#021526,stroke-width:1px,font-size:10px;

    classDef factory fill:#,stroke:#6EACDA,stroke-width:1px;
    classDef segmentation fill:#,stroke:#6EACDA,stroke-width:1px;
    classDef mapping fill:#,stroke:#6EACDA,stroke-width:1px;

    %% Factory creates different segmentation types
    semantic_segmentation_factory -->|*SemanticSegmentationType*| DEEPLABV3;
    semantic_segmentation_factory -->|*SemanticSegmentationType*| SEGFORMER;
    semantic_segmentation_factory -->|*SemanticSegmentationType*| CLIP;

    %% Segmentation types create corresponding classes
    DEEPLABV3 -->|*creates*| SemanticSegmentationDeepLabV3;
    SEGFORMER -->|*creates*| SemanticSegmentationSegformer;
    CLIP -->|*creates*| SemanticSegmentationCLIP;

    %% Each segmentation class --> base
    SemanticSegmentationDeepLabV3 -->|*_is-a_*| SemanticSegmentationBase;
    SemanticSegmentationSegformer -->|*_is-a_*| SemanticSegmentationBase;
    SemanticSegmentationCLIP -->|*_is-a_*| SemanticSegmentationBase;

    %% Semantic mapping creation
    semantic_mapping_factory -->|*SemanticMappingType.DENSE*| SemanticMappingDense;
    SemanticMappingDense -->|*_is-a_*| SemanticMappingBase;

    %% SemanticMappingDense uses semantic_segmentation_factory
    SemanticMappingDense -->|*uses*| semantic_segmentation_factory;

    %% Factory
    class semantic_segmentation_factory factory;
    class semantic_mapping_factory factory;

    %% Segmentation models
    class DEEPLABV3 segmentation;
    class SEGFORMER segmentation;
    class CLIP segmentation;

    class SemanticSegmentationDeepLabV3 segmentation;
    class SemanticSegmentationSegformer segmentation;
    class SemanticSegmentationCLIP segmentation;
    class SemanticSegmentationBase segmentation;

    %% Mapping classes
    class SemanticMappingBase mapping;
    class SemanticMappingDense mapping;
