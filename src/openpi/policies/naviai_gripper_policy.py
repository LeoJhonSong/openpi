import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class NaviAIGripperInputs(transforms.DataTransformFn):
    """Converts NaviAI gripper inputs to the model expected format.

    Gripper-family (right-arm only) embodiment, dimension-agnostic: state/action are
    passed through unchanged (tcp_gripper=7).
    Images: realsense_up (base) and right_wrist are active; left_wrist is zeroed and masked.
    """

    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        right_wrist_image = _parse_image(data["observation/right_wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class NaviAIGripperOutputs(transforms.DataTransformFn):
    """Converts model outputs back to NaviAI gripper action format.

    The model pads actions to its internal width (32); slice back to the real
    dimension. action_dim is required and supplied per-config (no mode default).
    """

    action_dim: int

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
