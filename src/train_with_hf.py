#!/usr/bin/env python3
"""
Custom training entry point that registers HuggingFace model before running Modalities.
"""
import sys
from modalities.registry.registry import Registry
from modalities.models.huggingface.huggingface_model import (
    HuggingFacePretrainedModel,
    HuggingFacePretrainedModelConfig
)
from modalities.__main__ import main

# Register the HuggingFace model BEFORE Modalities starts
registry = Registry()
registry.register(
    component_key="model",
    variant_key="huggingface_pretrained",
    component=HuggingFacePretrainedModel,
    component_config=HuggingFacePretrainedModelConfig
)

print("✓ Registered HuggingFacePretrainedModel")

# Now run Modalities normally
if __name__ == "__main__":
    sys.exit(main())
