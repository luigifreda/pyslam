graph LR
    linkStyle default stroke:#021526,stroke-width:2px,font-size:12px;

    classDef moduleTracking fill:#FFD6D6,stroke:#D7263D,stroke-width:2px,font-weight:bold,font-size:24px;
    classDef moduleLocalMapping fill:#D6FFD6,stroke:#218380,stroke-width:2px,font-weight:bold,font-size:24px;
    classDef moduleLoopClosing fill:#D6E4FF,stroke:#00509D,stroke-width:2px,font-weight:bold,font-size:24px;
    classDef modulePlaceRecognition fill:#D6E4FF,stroke:#00509D,stroke-width:2px,font-weight:bold,font-size:24px;
    classDef moduleVolumetric fill:#FFFACD,stroke:#E1A100,stroke-width:2px,font-weight:bold,font-size:24px;
    classDef moduleGlobalBA fill:#EAD7F3,stroke:#6A0DAD,stroke-width:2px,font-weight:bold,font-size:24px;
    classDef moduleMap fill:#ECECEC,stroke:#333333,stroke-width:2px,font-weight:bold,font-size:24px;
    classDef moduleSemantic fill:#D1F7C4,stroke:#2E8B57,stroke-width:2px,font-weight:bold,font-size:24px;

    classDef component fill:none,stroke:#6EACDA,stroke-width:1px,font-size:24px;
    classDef data fill:none,stroke:#6EACDA,stroke-width:1px,font-size:24px;

    subgraph Tracking[TRACKING]
        FramePreprocessing[Frame<br>Preprocessing] --> PosePrediction
        PosePrediction[Pose Prediction<br>or Relocalization] --> TrackLocalMap
        TrackLocalMap[Track<br>Local Map] --> NewKeyFrameDecision[New KeyFrame<br>Decision] 
    end
        
    subgraph LocalMapping[LOCAL MAPPING]
        KeyFrameProcessing[KeyFrame<br>Processing] --> MapPointsCulling
        MapPointsCulling[Recent<br>Map Points<br>Culling] --> NewPointsCreation
        NewPointsCreation["New Points Creation<br>(Temporal&nbsp;Triangulation)"] --> MapPointFusion
        MapPointFusion[Map Points<br>Fusion] --> LocalBA
        LocalBA[Local BA] --> LocalKeyFramesCulling[Local Keyframes<br>Culling]
    end
    
    subgraph LoopClosing[LOOP CLOSING]
        subgraph PlaceRecognition["PLACE&nbsp;RECOGNITION"]
            VisualVocabulary[Visual Vocabulary] 
            LoopDetectionDatabase[Loop Detection<br>Database]
        end
        LoopDetection[Loop Detection] --> LoopGroupConsistencyChecking
        LoopGroupConsistencyChecking[Loop Group<br>Consistency Checking] --> LoopGeometryChecking
        LoopGeometryChecking[Loop Geometry<br>Checking]--> LoopCorrection
        LoopCorrection[Loop<br>Correction]  --> PoseGraphOptimization[Pose Graph<br>Optimization]      
    end

    subgraph VolumetricIntegration["VOLUMETRIC&nbsp;INTEGRATION"]
        KeyFrameQueue[KeyFrame Queue] --> DepthPrediction
        DepthPrediction["Depth Prediction<br>(Optional)"] --> VolumetricIntegrator[Volume Integration] 
        VolumetricIntegrator --> DenseMap[Dense Map]
    end    

    subgraph GlobalBA[GLOBAL BA]
        FullBA[Full BA] --> UpdateMap[Update Map] 
    end
    
    subgraph Map[SPARSE MAP]
        MapPoints
        KeyFrames
        CovisibilityGraph[Covisibility<br>Graph]
        SpanningTree[Spanning<br>Tree]
    end

    subgraph SemanticMapping["SEMANTIC&nbsp;MAPPING"]
        SemanticKeyFrameQueue[KeyFrame Queue] --> SemanticSegmentation[Semantic<br>Segmentation]
        SemanticSegmentation --> UpdateKeyFrameSemantics[Update<br>KeyFrame Semantics]
        SemanticSegmentation --> UpdateMapPointSemantics[Update<br>MapPoint Semantics]
        UpdateKeyFrameSemantics --> SemanticMap[Semantic Map]
        UpdateMapPointSemantics --> SemanticMap
    end

    ProcessedKeyframe --> SemanticMapping
    SemanticMapping <--> Map

    Frame(Frame) --> Tracking 
    Tracking --> NewKeyFrame(New<br>Keyframe) --> LocalMapping
    LocalMapping --> ProcessedKeyframe
    ProcessedKeyframe(Processed<br>Keyframe) --> LoopClosing
    ProcessedKeyframe --> VolumetricIntegration
    LoopClosing <--> GlobalBA

    Tracking <--> Map
    LocalMapping <--> Map
    LoopClosing <--> Map 
    GlobalBA <--> Map
    VolumetricIntegration <--> Map
    LoopClosing --> VolumetricIntegration

    class Tracking moduleTracking;
    class LocalMapping moduleLocalMapping;
    class LoopClosing moduleLoopClosing;
    class PlaceRecognition modulePlaceRecognition;
    class VolumetricIntegration moduleVolumetric;
    class GlobalBA moduleGlobalBA;
    class Map moduleMap;
    class SemanticMapping moduleSemantic;

    class FramePreprocessing component;
    class PosePrediction component;
    class TrackLocalMap component;
    class NewKeyFrameDecision component;

    class KeyFrameProcessing component;
    class MapPointsCulling component;
    class NewPointsCreation component;
    class MapPointFusion component;
    class LocalBA component;
    class LocalKeyFramesCulling component;

    class VisualVocabulary component;
    class LoopDetectionDatabase component;
    class LoopDetection component;
    class LoopGroupConsistencyChecking component;
    class LoopGeometryChecking component;
    class LoopCorrection component;
    class PoseGraphOptimization component;

    class KeyFrameQueue component;
    class DepthPrediction component;
    class VolumetricIntegrator component;
    class DenseMap component;

    class FullBA component;
    class UpdateMap component;

    class MapPoints component;
    class KeyFrames component;
    class CovisibilityGraph component;
    class SpanningTree component;

    class SemanticKeyFrameQueue component;
    class SemanticSegmentation component;
    class UpdateKeyFrameSemantics component;
    class UpdateMapPointSemantics component;
    class SemanticMap component;

    class Frame data;
    class NewKeyFrame data;
    class ProcessedKeyframe data;
