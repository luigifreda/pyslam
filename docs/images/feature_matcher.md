graph LR;
    %% Set default styles for all edges
    linkStyle default stroke:#021526,stroke-width:1px,font-size:10px;

    classDef factory fill:#D1F0FF,stroke:#0077B6,stroke-width:1.5px;
    classDef matcher fill:#FFFACD,stroke:#E1A100,stroke-width:1.5px;
    classDef singleMatcher fill:#EAD7F3,stroke:#6A0DAD,stroke-width:1.5px;
    classDef featureMatcher fill:#D6FFD6,stroke:#218380,stroke-width:1.5px;
    classDef component fill:none,stroke:#6EACDA,stroke-width:1px;
    classDef externalLib fill:#F1F1F1,stroke:#888888,stroke-width:1px;


    feature_matcher_factory -->|*matcher_type*| BF;
    feature_matcher_factory -->|*matcher_type*| FLANN;
    feature_matcher_factory -->|*matcher_type*| XFEAT;
    feature_matcher_factory -->|*matcher_type*| LIGHTGLUE;
    feature_matcher_factory -->|*matcher_type*| LOFTR;
        
    BF -->|*creates*| BfFeatureMatcher;
    FLANN -->|*creates*| FlannFeatureMatcher;
    XFEAT -->|*creates*| XFeatMatcher;
    LIGHTGLUE -->|*creates*| LightGlueMatcher;
    LOFTR -->|*creates*| LoFTRMatcher
    MAST3R --> |*creates*| Mast3RMatcher

    
    BfFeatureMatcher -->|*_is-a_*| FeatureMatcher;
    FlannFeatureMatcher -->|*_is-a_*| FeatureMatcher;
    XFeatMatcher -->|*_is-a_*| FeatureMatcher;
    LightGlueMatcher -->|*_is-a_*| FeatureMatcher;
    LoFTRMatcher -->|*_is-a_*| FeatureMatcher;
    Mast3RMatcher -->|*_is-a_*| FeatureMatcher;
    
    %% FeatureMatcher relationships    
    FeatureMatcher -->|*_has-a_*| matcher;    
    FeatureMatcher -->|*_has-a_*| matcher_type;
    FeatureMatcher -->|*_has-a_*| detector_type;
    FeatureMatcher -->|*_has-a_*| descriptor_type;
    FeatureMatcher -->|*_has-a_*| ratio_test;
    FeatureMatcher -->|*_has-a_*| norm_type;

    %% Feature dependencies
    matcher -->|*_is-a_*| cv2.BFMatcher;
    matcher -->|*_is-a_*| cv2.FlannBasedMatcher;
    matcher -->|*_is-a_*| xfeat.XFeat;
    matcher -->|*_is-a_*| lightglue.LightGlue;
    matcher -->|*_is-a_*| kornia.LoFTR;

    %% Single Matcher class
    class BfFeatureMatcher singleMatcher;
    class FlannFeatureMatcher singleMatcher;
    class XFeatMatcher singleMatcher;
    class LightGlueMatcher singleMatcher;
    class LoFTRMatcher singleMatcher;
    class Mast3RMatcher singleMatcher;

    %% feature_matcher_factory
    class feature_matcher_factory factory;
    
    %% FeatureMatcher types
    class BF matcher;
    class FLANN matcher;
    class XFEAT matcher;
    class LIGHTGLUE matcher;
    class LOFTR matcher;
    class MAST3R matcher;

    class FeatureMatcher featureMatcher;
    class Feature featureMatcher;
    class Descriptor featureMatcher;
    class DistanceMetric featureMatcher;
    class RatioTest featureMatcher;

    class matcher component;
    class matcher_type component;
    class detector_type component;
    class descriptor_type component;
    class ratio_test component;
    class norm_type component;

    class cv2.BFMatcher externalLib;
    class cv2.FlannBasedMatcher externalLib;
    class xfeat.XFeat externalLib;
    class lightglue.LightGlue externalLib;
    class kornia.LoFTR externalLib;