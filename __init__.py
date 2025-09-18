import torch

# --- ToCPU ノード ---
class ToCPUNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "to_cpu"
    CATEGORY = "utils"

    def to_cpu(self, image):
        return (image.cpu(),)


# --- ToGPU ノード ---
class ToGPUNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "to_gpu"
    CATEGORY = "utils"

    def to_gpu(self, image):
        return (image.to("cuda"),)


NODE_CLASS_MAPPINGS = {
    "ToCPU": ToCPUNode,
    "ToGPU": ToGPUNode
}
