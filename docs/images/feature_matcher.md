graph LR;
    %% Set default styles for all edges
    linkStyle default stroke:#0000FF,stroke-width:1px,font-size:10px;

    %% feature_matcher_factory
    classDef factory fill:#f9f,stroke:#333,stroke-width:2px
    class feature_matcher_factory factory;
    
    %% FeatureMatcher types
    classDef matcher fill:#f9f,stroke:#333,stroke-width:2px
    class BF matcher;
    class FLANN matcher;
    class XFEAT matcher;
    class LIGHTGLUE matcher;
    class LOFTR matcher;
    
    feature_matcher_factory -->|*matcher_type*| BF;
    feature_matcher_factory -->|*matcher_type*| FLANN;
    feature_matcher_factory -->|*matcher_type*| XFEAT;
    feature_matcher_factory -->|*matcher_type*| LIGHTGLUE;
    feature_matcher_factory -->|*matcher_type*| LOFTR;
    
    %% Single Matcher class
    classDef singleMatcher fill:#f9f,stroke:#333,stroke-width:2px
    class BfFeatureMatcher singleMatcher;
    class FlannFeatureMatcher singleMatcher;
    class XFeatMatcher singleMatcher;
    class LightGlueMatcher singleMatcher;
    class LoFTRMatcher singleMatcher;
    
    BF -->|*creates*| BfFeatureMatcher;
    FLANN -->|*creates*| FlannFeatureMatcher;
    XFEAT -->|*creates*| XFeatMatcher;
    LIGHTGLUE -->|*creates*| LightGlueMatcher;
    LOFTR -->|*creates*| LoFTRMatcher
    
    BfFeatureMatcher -->|*_is-a_*| FeatureMatcher;
    FlannFeatureMatcher -->|*_is-a_*| FeatureMatcher;
    XFeatMatcher -->|*_is-a_*| FeatureMatcher;
    LightGlueMatcher -->|*_is-a_*| FeatureMatcher;
    LoFTRMatcher -->|*_is-a_*| FeatureMatcher;
    
    %% FeatureMatcher relationships
    classDef featureMatcher fill:#f9f,stroke:#333,stroke-width:2px
    class FeatureMatcher featureMatcher;
    class Feature featureMatcher;
    class Descriptor featureMatcher;
    class DistanceMetric featureMatcher;
    class RatioTest featureMatcher;
    
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