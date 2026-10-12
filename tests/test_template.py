"""Unit tests for ComfyUI workflow template parsing, validation, and patching."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from batchpersona.template import WorkflowTemplate


def test_workflow_template_validation_success() -> None:
    """Verify valid ComfyUI prompt API graph passes validation."""
    valid_graph = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "test.png"}},
        "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
    }
    template = WorkflowTemplate(valid_graph)
    assert template is not None


def test_workflow_template_validation_failures() -> None:
    """Verify invalid graph schemas fail fast with descriptive ValueErrors."""
    with pytest.raises(ValueError, match="non-empty dictionary"):
        WorkflowTemplate({})

    with pytest.raises(ValueError, match="missing required key 'class_type'"):
        WorkflowTemplate({"1": {"inputs": {}}})

    with pytest.raises(ValueError, match="missing required key 'inputs'"):
        WorkflowTemplate({"1": {"class_type": "LoadImage"}})


def test_workflow_template_patching(tmp_path: Path) -> None:
    """Verify workflow patching replaces node values and placeholders correctly."""
    graph = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "{{CAMPAIGN_IMG}}"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "{{MASK_IMG}}"}},
        "3": {"class_type": "LoadImage", "inputs": {"image": "{{MODEL_IMG}}"}},
        "14": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "{{OUTPUT_PREFIX}}"},
        },
    }
    template = WorkflowTemplate(graph)
    patched = template.patch(
        campaign_image_name="campaign_uploaded.png",
        model_image_name="model_uploaded.png",
        output_prefix="campaign_apac_test",
        mask_image_name="mask_uploaded.png",
    )

    assert patched["1"]["inputs"]["image"] == "campaign_uploaded.png"
    assert patched["2"]["inputs"]["image"] == "mask_uploaded.png"
    assert patched["3"]["inputs"]["image"] == "model_uploaded.png"
    assert patched["14"]["inputs"]["filename_prefix"] == "campaign_apac_test"


def test_production_workflow_json_integrity() -> None:
    """Verify the real workflows/model_swap_api.json file conforms to ComfyUI standards."""
    workflow_path = Path("workflows/model_swap_api.json")
    assert workflow_path.is_file()

    template = WorkflowTemplate.load_from_file(workflow_path)
    patched = template.patch(
        campaign_image_name="campaign.png",
        model_image_name="model.png",
        output_prefix="output_test",
        mask_image_name="mask.png",
    )

    # Invariants from architecture review:
    # 1. Attention mask connected to IPAdapterApply (Node 11)
    assert "11" in patched
    assert patched["11"]["class_type"] == "IPAdapterApply"
    assert patched["11"]["inputs"]["attn_mask"] == ["7", 0]

    # 2. VAEEncodeForInpaint receives pixel and mask (Node 8)
    assert "8" in patched
    assert patched["8"]["class_type"] == "VAEEncodeForInpaint"
    assert patched["8"]["inputs"]["pixels"] == ["1", 0]
    assert patched["8"]["inputs"]["mask"] == ["7", 0]

    # 3. No raw unreplaced {{...}} placeholders in final JSON
    serialized = json.dumps(patched)
    assert re.search(r"\{\{[A-Za-z0-9_]+\}\}", serialized) is None


def test_composite_workflow_json_integrity() -> None:
    """Verify workflows/model_swap_composite_api.json conforms to ComfyUI standards."""
    workflow_path = Path("workflows/model_swap_composite_api.json")
    assert workflow_path.is_file()

    template = WorkflowTemplate.load_from_file(workflow_path)
    patched = template.patch(
        campaign_image_name="campaign.png",
        model_image_name="model.png",
        output_prefix="output_test",
        mask_image_name="mask.png",
    )
    assert patched["11"]["class_type"] == "ImageCompositeMasked"
    assert patched["14"]["class_type"] == "SaveImage"

    serialized = json.dumps(patched)
    assert re.search(r"\{\{[A-Za-z0-9_]+\}\}", serialized) is None


def test_auto_mask_workflow_json_integrity() -> None:
    """Verify workflows/model_swap_auto_mask_api.json conforms to ComfyUI standards."""
    workflow_path = Path("workflows/model_swap_auto_mask_api.json")
    assert workflow_path.is_file()

    template = WorkflowTemplate.load_from_file(workflow_path)
    assert not template.requires_mask()

    patched = template.patch(
        campaign_image_name="campaign.png",
        model_image_name="model.png",
        output_prefix="output_test",
    )
    assert patched["3"]["class_type"] == "LoadBackgroundRemovalModel"
    assert patched["4"]["class_type"] == "RemoveBackground"
    assert patched["5"]["class_type"] == "ImageCompositeMasked"
    assert patched["6"]["class_type"] == "SaveImage"

    serialized = json.dumps(patched)
    assert re.search(r"\{\{[A-Za-z0-9_]+\}\}", serialized) is None


def test_maskless_qwen21_workflow_json_integrity() -> None:
    """Verify workflows/model_swap_qwen21_maskless_api.json conforms to ComfyUI standards."""
    workflow_path = Path("workflows/model_swap_qwen21_maskless_api.json")
    assert workflow_path.is_file()

    template = WorkflowTemplate.load_from_file(workflow_path)
    assert not template.requires_mask()

    patched = template.patch(
        campaign_image_name="campaign.png",
        model_image_name="model.png",
        output_prefix="output_test",
    )
    assert patched["6"]["class_type"] == "TextEncodeQwenImage21"
    assert patched["7"]["class_type"] == "KSampler"
    assert patched["9"]["class_type"] == "SaveImage"

    serialized = json.dumps(patched)
    assert re.search(r"\{\{[A-Za-z0-9_]+\}\}", serialized) is None


def test_workflow_template_non_dict_node_raises_type_error() -> None:
    """Verify TypeError is raised when a node is not a dictionary."""
    with pytest.raises(TypeError, match="definition must be a dictionary"):
        WorkflowTemplate({"1": "not-a-dict"})  # type: ignore[arg-type]


def test_workflow_template_file_not_found(tmp_path: Path) -> None:
    """Verify FileNotFoundError is raised when workflow template path is missing."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        WorkflowTemplate.load_from_file(tmp_path / "nonexistent.json")


def test_workflow_template_missing_required_mask_raises() -> None:
    """Verify patching raises ValueError if template contains {{MASK_IMG}} but mask_image_name is omitted."""
    raw_graph = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "{{CAMPAIGN_IMG}}"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "{{MASK_IMG}}"}},
        "3": {"class_type": "LoadImage", "inputs": {"image": "{{MODEL_IMG}}"}},
        "4": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "{{OUTPUT_PREFIX}}"},
        },
    }
    template = WorkflowTemplate(raw_graph=raw_graph)
    assert template.requires_mask()

    with pytest.raises(ValueError, match="requires an external inpainting mask"):
        template.patch(
            campaign_image_name="c.png",
            model_image_name="m.png",
            output_prefix="out",
            mask_image_name=None,
        )


def test_workflow_template_seed_resolution_prompt_overrides() -> None:
    """Verify type-safe injection of seed, resolution, and prompt with token preservation."""
    sample_graph = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "{{CAMPAIGN_IMG}}"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "{{MODEL_IMG}}"}},
        "3": {
            "class_type": "KSampler",
            "inputs": {"seed": 100, "steps": 20, "cfg": 7.0},
        },
        "4": {
            "class_type": "TextEncodeQwenImage21",
            "inputs": {
                "prompt": "original prompt",
                "resolution": 512,
            },
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "photorealistic portrait, beautiful lighting"},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "blurry, low quality, bad anatomy, artifacts"},
        },
        "7": {"class_type": "SaveImage", "inputs": {"filename_prefix": "{{OUTPUT_PREFIX}}"}},
    }
    template = WorkflowTemplate(sample_graph)
    patched = template.patch(
        campaign_image_name="camp.png",
        model_image_name="model.png",
        output_prefix="out_test",
        seed=999999,
        resolution=1024,
        prompt="Studio lighting, 8k portrait",
    )

    # 1. KSampler seed is typed int
    assert patched["3"]["inputs"]["seed"] == 999999
    assert isinstance(patched["3"]["inputs"]["seed"], int)

    # 2. TextEncodeQwenImage21 resolution is typed int
    assert patched["4"]["inputs"]["resolution"] == 1024
    assert isinstance(patched["4"]["inputs"]["resolution"], int)

    # 3. TextEncodeQwenImage21 prompt preserves <image1> and <image2> tokens
    qwen_prompt = patched["4"]["inputs"]["prompt"]
    assert "<image1>" in qwen_prompt
    assert "<image2>" in qwen_prompt
    assert "Studio lighting, 8k portrait" in qwen_prompt

    # 4. CLIP positive prompt is replaced, but negative prompt is preserved
    assert patched["5"]["inputs"]["text"] == "Studio lighting, 8k portrait"
    assert "blurry, low quality" in patched["6"]["inputs"]["text"]


def test_workflow_template_get_required_node_types() -> None:
    """Verify get_required_node_types extracts all class_type strings from workflow."""
    wf_data = {
        "1": {"class_type": "LoadImage", "inputs": {}},
        "2": {"class_type": "KSampler", "inputs": {}},
        "3": {"class_type": "TextEncodeQwenImage21", "inputs": {}},
        "4": {"class_type": "LoadImage", "inputs": {}},
    }
    template = WorkflowTemplate(wf_data)
    nodes = template.get_required_node_types()
    assert nodes == {"LoadImage", "KSampler", "TextEncodeQwenImage21"}
