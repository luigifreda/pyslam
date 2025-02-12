graph LR
    %% Set default styles for all edges
    linkStyle default stroke:#021526,stroke-width:2px,font-size:12px;

    classDef module fill:none,stroke:#6EACDA,stroke-width:2px,font-weight:bold, font-size:24px;
    classDef component fill:#none,stroke:#6EACDA,stroke-width:1px, font-size:24px;  
    classDef data fill:#none,stroke:#6EACDA,stroke-width:1px, font-size:24px;

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

    class Tracking module;
    class FramePreprocessing component;
    class PosePrediction component;
    class TrackLocalMap component;
    class NewKeyFrameDecision component;

    class LocalMapping module;
    class KeyFrameProcessing component;
    class MapPointsCulling component;
    class NewPointsCreation component;
    class MapPointFusion component;
    class LocalBA component;
    class LocalKeyFramesCulling component;
    
    class LoopClosing module;
    class PlaceRecognition module;
    class VisualVocabulary component;
    class LoopDetectionDatabase component;
    class LoopDetection component;
    class LoopGroupConsistencyChecking component;
    class LoopGeometryChecking component;
    class LoopCorrection component;
    class PoseGraphOptimization component;
    
    class VolumetricIntegration module;
    class KeyFrameQueue component;
    class DepthPrediction component;
    class VolumetricIntegrator component;
    class DenseMap component;
    
    class GlobalBA module;
    class FullBA component;
    class UpdateMap component;
    
    class Map module;
    class MapPoints component;
    class KeyFrames component;
    class CovisibilityGraph component;
    class SpanningTree component;
    
    class Frame data;
    class NewKeyFrame data;
    class ProcessedKeyframe data;
