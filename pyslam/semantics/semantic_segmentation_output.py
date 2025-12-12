class SemanticSegmentationOutput:
    """
    Container for semantic segmentation model inference results.

    Attributes:
        semantics: numpy array of shape (H, W) for LABEL, (H, W, num_classes) for PROBABILITY_VECTOR,
                   or (H, W, D) for FEATURE_VECTOR
        instances: numpy array of shape (H, W) for instance IDs (optional, None if not available)
    """

    def __init__(self, semantics=None, instances=None):
        self.semantics = semantics  # numpy array: (H, W) for LABEL, (H, W, num_classes) for PROBABILITY_VECTOR, or (H, W, D) for FEATURE_VECTOR
        self.instances = instances  # numpy array of shape (H, W) for INSTANCES (optional)
