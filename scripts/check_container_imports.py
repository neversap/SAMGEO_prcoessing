import importlib


MODULES = [
    "torch",
    "torchvision",
    "einops",
    "sam3",
    "sam3.model_builder",
    "sam3.model.sam3_image_processor",
]


for module_name in MODULES:
    module = importlib.import_module(module_name)
    version = getattr(module, "__version__", "")
    suffix = f" {version}" if version else ""
    print(f"OK {module_name}{suffix}")
