import numpy as np
import config


class DefectInference:
    """Loads a TFLite model and runs inference.
    Falls back to a dummy random classifier when model file is absent (dev/test mode).
    """

    def __init__(self):
        self._dummy = False
        try:
            import tflite_runtime.interpreter as tflite
            self.interpreter = tflite.Interpreter(model_path=config.MODEL_PATH)
        except ImportError:
            try:
                import tensorflow as tf
                self.interpreter = tf.lite.Interpreter(model_path=config.MODEL_PATH)
            except Exception:
                self.interpreter = None

        if self.interpreter is None or not self._model_file_exists():
            print("[Model] No TFLite model found — running in DUMMY mode (random outputs).")
            self._dummy = True
        else:
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print(f"[Model] Loaded: {config.MODEL_PATH}")

        with open(config.LABELS_PATH, "r") as f:
            self.labels = [ln.strip() for ln in f if ln.strip()]
        print(f"[Model] Labels: {self.labels}")

    def _model_file_exists(self) -> bool:
        import os
        return os.path.isfile(config.MODEL_PATH)

    def predict(self, input_tensor: np.ndarray):
        """
        Run inference on preprocessed tensor (1, H, W, 3) float32.
        Returns: (label: str, confidence: float, is_defect: bool)
        """
        if self._dummy:
            scores = np.random.dirichlet(np.ones(len(self.labels)))
        else:
            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            self.interpreter.invoke()
            scores = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        class_idx = int(np.argmax(scores))
        confidence = float(scores[class_idx])
        label = self.labels[class_idx]
        is_defect = (label.upper() == "DEFECT") and (confidence >= config.CONFIDENCE_THRESHOLD)
        return label, confidence, is_defect
