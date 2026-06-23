"""
Example bridge file for server/app/adapters/sam3.py.

Put a copy of this file on the server, for example:

    /opt/sam3/server_sam3_adapter.py

Then configure:

    SAM_GEO_CODE_PATH=/opt/sam3
    SAM_GEO_SAM3_ENTRYPOINT=server_sam3_adapter:create_predictor

Replace the placeholder imports and calls with the actual SAM3 implementation
available on the server.
"""


class Sam3PredictorBridge:
    def __init__(self, model_dir: str, device: str) -> None:
        self.model_dir = model_dir
        self.device = device
        self.model = self._load_model()

    def _load_model(self):
        # Example shape only. Replace with the official/server SAM3 loader.
        #
        # from sam3 import build_sam3
        # model = build_sam3(model_dir=self.model_dir, device=self.device)
        # model.eval()
        # return model
        raise NotImplementedError("Wire this bridge to the server SAM3 code.")

    def predict(self, image, prompt: str, box=None, points=None):
        # Return one of these shapes:
        #
        # {"masks": masks, "scores": scores}
        # masks
        #
        # masks can be a numpy array, torch tensor, PIL image, or list of masks.
        # Each mask should resolve to H x W.
        raise NotImplementedError("Wire this bridge to the server SAM3 predict API.")


def create_predictor(model_dir: str, device: str):
    return Sam3PredictorBridge(model_dir=model_dir, device=device)
