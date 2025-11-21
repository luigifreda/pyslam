graph LR;
    %% Set default styles for all edges
    linkStyle default stroke:#021526,stroke-width:1px,font-size:14px;

    classDef factory fill:#D1F0FF,stroke:#0077B6,stroke-width:1.5px;
    classDef base fill:#FFE3F1,stroke:#C71585,stroke-width:1.5px;
    classDef implementation fill:#D6FFD6,stroke:#218380,stroke-width:1.5px;
    classDef result fill:#FFFACD,stroke:#E1A100,stroke-width:1.5px;
    classDef enum fill:#EAD7F3,stroke:#6A0DAD,stroke-width:1.5px;

    %% Factory creates different implementations based on type
    scene_from_views_factory -->|*SceneFromViewsType.DEPTH_ANYTHING_V3*| SceneFromViewsDepthAnythingV3;
    scene_from_views_factory -->|*SceneFromViewsType.MAST3R*| SceneFromViewsMast3r;
    scene_from_views_factory -->|*SceneFromViewsType.MVDUST3R*| SceneFromViewsMvdust3r;
    scene_from_views_factory -->|*SceneFromViewsType.VGGT*| SceneFromViewsVggt;
    scene_from_views_factory -->|*SceneFromViewsType.DUST3R*| SceneFromViewsDust3r;

    %% Factory uses enum
    scene_from_views_factory -->|*uses*| SceneFromViewsType;

    %% All implementations inherit from base
    SceneFromViewsDepthAnythingV3 -->|*_is-a_*| SceneFromViewsBase;
    SceneFromViewsMast3r -->|*_is-a_*| SceneFromViewsBase;
    SceneFromViewsMvdust3r -->|*_is-a_*| SceneFromViewsBase;
    SceneFromViewsVggt -->|*_is-a_*| SceneFromViewsBase;
    SceneFromViewsDust3r -->|*_is-a_*| SceneFromViewsBase;

    %% Base class returns result
    SceneFromViewsBase -->|*returns*| SceneFromViewsResult;

    %% Base class shared pipeline
    SceneFromViewsBase -->|*implements*| reconstruct["reconstruct()<br><span style='font-size:12px;'>_Shared Pipeline_</span><br><span style='font-size:11px;'>preprocess → infer → postprocess</span>"];

    %% Factory
    class scene_from_views_factory factory;

    %% Base class
    class SceneFromViewsBase base;
    class reconstruct base;

    %% Implementations
    class SceneFromViewsDepthAnythingV3 implementation;
    class SceneFromViewsMast3r implementation;
    class SceneFromViewsMvdust3r implementation;
    class SceneFromViewsVggt implementation;
    class SceneFromViewsDust3r implementation;

    %% Result structure
    class SceneFromViewsResult result;

    %% Enum
    class SceneFromViewsType enum;

