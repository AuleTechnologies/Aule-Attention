"""
ComfyUI custom nodes for aule-attention.

Provides nodes to enable aule-attention acceleration in ComfyUI workflows.
Works with any model that uses PyTorch's scaled_dot_product_attention:
- Stable Diffusion 1.5, 2.x
- SDXL
- Flux
- SD3
- Any other diffusion model
"""

import aule


class AuleInstall:
    """
    Enable aule-attention for all models in this workflow.

    Place this node at the start of your workflow. Once executed,
    all subsequent attention operations will use aule-attention
    (Triton on ROCm/CUDA, Vulkan on consumer GPUs).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_TYPES = ()
    FUNCTION = "install"
    CATEGORY = "aule"
    OUTPUT_NODE = True

    def install(self):
        aule.install()
        return ()


class AuleUninstall:
    """
    Disable aule-attention and restore PyTorch's default attention.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_TYPES = ()
    FUNCTION = "uninstall"
    CATEGORY = "aule"
    OUTPUT_NODE = True

    def uninstall(self):
        aule.uninstall()
        return ()


class AuleInfo:
    """
    Display aule-attention backend information.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "info"
    CATEGORY = "aule"

    def info(self):
        backends = aule.get_available_backends()
        info = aule.get_backend_info()

        lines = [
            f"aule-attention v{aule.__version__}",
            f"Available backends: {', '.join(backends)}",
            "",
        ]

        for name, details in info.items():
            if details.get('available'):
                device = details.get('device', 'N/A')
                desc = details.get('description', '')
                lines.append(f"[{name.upper()}] {device}")
                if desc:
                    lines.append(f"  {desc}")

        return ("\n".join(lines),)


class AulePatchModel:
    """
    Apply aule-attention to a specific model.

    Alternative to AuleInstall - patches only this model instead of globally.
    Useful when you want fine-grained control over which models use aule.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "causal": ("BOOLEAN", {"default": False, "label_on": "True (LLM)", "label_off": "False (Diffusion)"}),
                "use_rope": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "aule"

    def patch(self, model, causal, use_rope):
        print(f"Aule: Patching ComfyUI model... {model}")
        
        config = {
            "causal": causal,
            "use_rope": use_rope
        }
        try:
            raw_model = model.model
        except AttributeError:
            raw_model = model # Maybe it is already raw?
            
        aule.patch_model(raw_model, config=config)
        
        return (model,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "AuleInstall": AuleInstall,
    "AuleUninstall": AuleUninstall,
    "AuleInfo": AuleInfo,
    "AulePatchModel": AulePatchModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AuleInstall": "Aule Enable",
    "AuleUninstall": "Aule Disable",
    "AuleInfo": "Aule Info",
    "AulePatchModel": "Aule Patch Model",
}
