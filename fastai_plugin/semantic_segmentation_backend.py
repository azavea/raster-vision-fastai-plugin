from rastervision.backend import Backend


class SemanticSegmentationBackend(Backend):
    def __init__(self, config, task_config):
        self.config = config
        self.task_config = task_config

    def process_scene_data(self, scene, data, tmp_dir):
        """Process each scene's training data.

        Args:
            scene: Scene
            data: TrainingData

        Returns:
            backend-specific data-structures consumed by backend's
            process_sceneset_results
        """
        pass

    def process_sceneset_results(self, training_results, validation_results,
                                 tmp_dir):
        """After all scenes have been processed, process the result set.

        Args:
            training_results: dependent on the ml_backend's process_scene_data
            validation_results: dependent on the ml_backend's
                process_scene_data
        """
        pass

    def train(self, tmp_dir):
        """Train a model."""
        pass

    def load_model(self, tmp_dir):
        """Load the model in preparation for one or more prediction calls."""
        pass

    def predict(self, chips, windows, tmp_dir):
        """Return predictions for a chip using model.

        Args:
            chips: [[height, width, channels], ...] numpy array of chips
            windows: List of boxes that are the windows aligned with the chips.

        Return:
            Labels object containing predictions
        """
        pass
