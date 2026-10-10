"""Workflow graph template loading, validation, and dynamic mutation for ComfyUI Prompt API."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


class WorkflowTemplate:
    """Handles loading, validation, and dynamic mutation of the ComfyUI API workflow graph."""

    def __init__(self, raw_graph: dict[str, Any]) -> None:
        self._validate_graph(raw_graph)
        self._graph = raw_graph

    @classmethod
    def load_from_file(cls, file_path: Path) -> WorkflowTemplate:
        """Load and parse workflow graph from a JSON file."""
        if not file_path.is_file():
            raise FileNotFoundError(f"Workflow file does not exist: {file_path}")
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)

    @staticmethod
    def _validate_graph(graph: dict[str, Any]) -> None:
        """Verify graph adheres to ComfyUI Prompt API schema."""
        if not isinstance(graph, dict) or not graph:
            raise ValueError("Workflow template must be a non-empty dictionary.")
        for node_id, node_def in graph.items():
            if not isinstance(node_def, dict):
                raise TypeError(f"Node '{node_id}' definition must be a dictionary.")
            if "class_type" not in node_def:
                raise ValueError(f"Node '{node_id}' missing required key 'class_type'.")
            if "inputs" not in node_def:
                raise ValueError(f"Node '{node_id}' missing required key 'inputs'.")

    def requires_mask(self) -> bool:
        """Check whether the workflow template expects an external mask image."""
        serialized = json.dumps(self._graph)
        return "{{MASK_IMG}}" in serialized

    def patch(
        self,
        campaign_image_name: str,
        model_image_name: str,
        output_prefix: str,
        mask_image_name: str | None = None,
        seed: int | None = None,
        resolution: int | None = None,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        """Produce a patched clone of the workflow graph with injected asset references and parameters."""
        if self.requires_mask() and mask_image_name is None:
            raise ValueError(
                "Workflow template requires an external inpainting mask ({{MASK_IMG}}), but none was provided or auto-paired."
            )

        patched: dict[str, Any] = copy.deepcopy(self._graph)

        # 1. Update any SaveImage node with the unique output prefix
        for node_def in patched.values():
            if isinstance(node_def, dict) and node_def.get("class_type") == "SaveImage":
                node_def.setdefault("inputs", {})["filename_prefix"] = output_prefix

        # 2. String template substitution
        placeholders: dict[str, str] = {
            "{{CAMPAIGN_IMG}}": campaign_image_name,
            "{{MODEL_IMG}}": model_image_name,
            "{{OUTPUT_PREFIX}}": output_prefix,
        }
        if mask_image_name is not None:
            placeholders["{{MASK_IMG}}"] = mask_image_name

        self._substitute_placeholders_recursive(patched, placeholders)

        # 3. Fallback for legacy graphs without {{CAMPAIGN_IMG}} placeholder
        orig_serialized = json.dumps(self._graph)
        if "{{CAMPAIGN_IMG}}" not in orig_serialized:
            if "1" in patched and patched["1"].get("class_type") == "LoadImage":
                patched["1"].setdefault("inputs", {})["image"] = campaign_image_name

            if (
                "2" in patched
                and patched["2"].get("class_type") == "LoadImage"
                and "3" in patched
                and patched["3"].get("class_type") == "LoadImage"
            ):
                if mask_image_name is not None:
                    patched["2"].setdefault("inputs", {})["image"] = mask_image_name
                patched["3"].setdefault("inputs", {})["image"] = model_image_name
            elif (
                "2" in patched
                and patched["2"].get("class_type") == "LoadImage"
                and patched.get("3", {}).get("class_type") != "LoadImage"
            ):
                patched["2"].setdefault("inputs", {})["image"] = model_image_name

        # 4. Type-safe numeric & prompt overrides at parsed dict level
        if seed is not None:
            for node_def in patched.values():
                if isinstance(node_def, dict) and node_def.get("class_type") in {
                    "KSampler",
                    "KSamplerAdvanced",
                }:
                    node_def.setdefault("inputs", {})["seed"] = int(seed)

        if resolution is not None:
            for node_def in patched.values():
                if (
                    isinstance(node_def, dict)
                    and node_def.get("class_type") == "TextEncodeQwenImage21"
                ):
                    node_def.setdefault("inputs", {})["resolution"] = int(resolution)

        if prompt is not None and prompt.strip():
            user_prompt = prompt.strip()
            for node_def in patched.values():
                if isinstance(node_def, dict):
                    class_type = node_def.get("class_type")
                    if class_type == "TextEncodeQwenImage21":
                        # Preserve Qwen multimodal structural conditioning tokens if user did not provide them
                        final_prompt = user_prompt
                        if "<image1>" not in final_prompt or "<image2>" not in final_prompt:
                            final_prompt = (
                                f"In <image1>, replace the person with the target reference model from <image2>. "
                                f"{user_prompt}"
                            )
                        node_def.setdefault("inputs", {})["prompt"] = final_prompt
                    elif class_type == "CLIPTextEncode":
                        # Only override positive prompt (avoid overwriting negative prompt)
                        current_text = str(node_def.get("inputs", {}).get("text", "")).lower()
                        negative_indicators = [
                            "distorted",
                            "blurry",
                            "artifacts",
                            "bad anatomy",
                            "negative",
                        ]
                        is_negative = any(ind in current_text for ind in negative_indicators)
                        if not is_negative:
                            node_def.setdefault("inputs", {})["text"] = user_prompt

        return patched

    def _substitute_placeholders_recursive(
        self,
        target: Any,
        placeholders: dict[str, str],
    ) -> None:
        """Recursively scan dict/list in-place and replace placeholder string tokens."""
        if isinstance(target, dict):
            for key, val in target.items():
                if isinstance(val, str) and val in placeholders:
                    target[key] = placeholders[val]
                elif isinstance(val, (dict, list)):
                    self._substitute_placeholders_recursive(val, placeholders)
        elif isinstance(target, list):
            for i, val in enumerate(target):
                if isinstance(val, str) and val in placeholders:
                    target[i] = placeholders[val]
                elif isinstance(val, (dict, list)):
                    self._substitute_placeholders_recursive(val, placeholders)
