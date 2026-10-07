"""Unit and integration test suite for the headless ComfyUI batch swapper pipeline."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from scripts.batch_swapper import (
    BatchSwapper,
    ComfyAPIError,
    ComfyExecutionError,
    ComfyTimeoutError,
    ComfyUIClient,
    JobStatus,
    OutputAsset,
    SwapperConfig,
    WorkflowTemplate,
    parse_cli_args,
)
from scripts.generate_testdata import (
    EAST_ASIA_PALETTE,
    NORDIC_PALETTE,
    SOUTH_AMERICA_PALETTE,
    generate_all_testdata,
    synthesize_campaign_lookbook,
    synthesize_campaign_mask,
    synthesize_model_portrait,
)


class MockWebSocket:
    """Mock WebSocket client simulating ComfyUI message framing and lifecycle."""

    def __init__(self, frames: list[str | bytes]) -> None:
        self._frames = list(frames)
        self.closed = False

    def recv(self) -> str | bytes:
        if not self._frames:
            raise StopIteration("No more mock frames")
        return self._frames.pop(0)

    def close(self) -> None:
        self.closed = True


# ============================================================================
# PHASE 1 TESTS: Synthetic Test Data Generator
# ============================================================================


def test_synthesize_campaign_lookbook() -> None:
    """Verify generated campaign lookbook has expected dimensions, mode, and content."""
    img = synthesize_campaign_lookbook(size=256)
    assert img.size == (256, 256)
    assert img.mode == "RGB"

    # Verify lookbook is not blank or flat
    extrema = img.getextrema()
    # Check that channels have dynamic range
    assert isinstance(extrema, tuple)
    for band_range in extrema:
        if isinstance(band_range, tuple):
            ch_min, ch_max = band_range
            assert ch_max - ch_min > 50


def test_synthesize_campaign_mask() -> None:
    """Verify generated mask is grayscale, contains binary regions and feathered edges."""
    mask = synthesize_campaign_mask(size=256)
    assert mask.size == (256, 256)
    assert mask.mode == "L"

    # Mask should have both 0 (unmasked) and 255 (masked target) regions
    extrema = mask.getextrema()
    assert extrema[0] == 0
    assert extrema[1] == 255


def test_synthesize_model_portraits() -> None:
    """Verify distinct archetype generation works across all palettes."""
    palettes = [
        (EAST_ASIA_PALETTE, "East Asia Female", False),
        (SOUTH_AMERICA_PALETTE, "South America Male", True),
        (NORDIC_PALETTE, "Nordic Female", False),
    ]
    for pal, title, is_masculine in palettes:
        img = synthesize_model_portrait(pal, title, size=256, is_masculine_features=is_masculine)
        assert img.size == (256, 256)
        assert img.mode == "RGB"


def test_generate_all_testdata(tmp_path: Path) -> None:
    """Verify end-to-end file generation writes all required assets with valid formats."""
    files = generate_all_testdata(output_root=tmp_path, size=128)
    assert len(files) == 5

    expected_names = {
        "campaign_summer_lookbook.png",
        "campaign_summer_mask.png",
        "model_east_asia_f01.png",
        "model_south_america_m01.png",
        "model_nordic_f01.png",
    }
    actual_names = {f.name for f in files}
    assert expected_names == actual_names

    for file_path in files:
        assert file_path.is_file()
        assert file_path.stat().st_size > 0
        with Image.open(file_path) as im:
            assert im.size == (128, 128)


# ============================================================================
# PHASE 2 TESTS: ComfyUI Workflow Template
# ============================================================================


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
    import re

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

    import re

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

    import re

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

    import re

    serialized = json.dumps(patched)
    assert re.search(r"\{\{[A-Za-z0-9_]+\}\}", serialized) is None


def test_batch_swapper_auto_pairs_mask(tmp_path: Path) -> None:
    """Verify BatchSwapper automatically detects and pairs <campaign>_mask.png if --mask is omitted."""
    campaign_file = tmp_path / "lookbook.png"
    campaign_file.write_bytes(b"campaign")
    mask_file = tmp_path / "lookbook_mask.png"
    mask_file.write_bytes(b"mask")

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "m.png").write_bytes(b"m")

    wf_file = tmp_path / "wf.json"
    wf_file.write_text(
        json.dumps(
            {
                "1": {
                    "class_type": "LoadImage",
                    "inputs": {"image": "{{CAMPAIGN_IMG}}"},
                },
                "2": {"class_type": "LoadImage", "inputs": {"image": "{{MASK_IMG}}"}},
                "3": {"class_type": "LoadImage", "inputs": {"image": "{{MODEL_IMG}}"}},
                "4": {
                    "class_type": "SaveImage",
                    "inputs": {"filename_prefix": "{{OUTPUT_PREFIX}}"},
                },
            }
        ),
        encoding="utf-8",
    )

    config = SwapperConfig(
        server_address="127.0.0.1:8188",
        campaign_path=campaign_file,
        mask_path=None,  # Not provided explicitly!
        models_dir=models_dir,
        output_dir=tmp_path / "out",
        market_tag="test",
        workflow_path=wf_file,
    )

    mock_client = MagicMock(spec=ComfyUIClient)
    mock_client.upload_image.side_effect = lambda p: f"up_{p.name}"
    mock_client.queue_prompt.return_value = "p-1"
    mock_client.track_execution.return_value = [OutputAsset("out.png", "", "output")]

    swapper = BatchSwapper(config=config, client=mock_client)
    results = swapper.run_batch()

    assert len(results) == 1
    # Verify that the mask file was indeed uploaded via auto-pairing
    uploaded_files = [call[0][0].name for call in mock_client.upload_image.call_args_list]
    assert "lookbook.png" in uploaded_files
    assert "lookbook_mask.png" in uploaded_files


# ============================================================================
# PHASE 3 TESTS: ComfyUI Client & Headless Batch Orchestrator
# ============================================================================


def test_client_upload_image_success(tmp_path: Path) -> None:
    """Verify upload_image issues multipart request and returns ComfyUI asset name."""
    test_img = tmp_path / "sample.png"
    test_img.write_bytes(b"dummy_png_bytes")

    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"name": "sample_uploaded.png"}
    mock_session.post.return_value = mock_resp

    client = ComfyUIClient(server_address="127.0.0.1:8188", session=mock_session)
    uploaded = client.upload_image(test_img)

    assert uploaded == "sample_uploaded.png"
    mock_session.post.assert_called_once()
    assert "upload/image" in mock_session.post.call_args[0][0]


def test_client_upload_image_error(tmp_path: Path) -> None:
    """Verify upload_image raises ComfyAPIError when server returns non-200."""
    test_img = tmp_path / "sample.png"
    test_img.write_bytes(b"dummy_png_bytes")

    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.text = "Bad image format"
    mock_session.post.return_value = mock_resp

    client = ComfyUIClient(server_address="127.0.0.1:8188", session=mock_session)
    with pytest.raises(ComfyAPIError, match="Upload failed"):
        client.upload_image(test_img)


def test_client_queue_prompt_success() -> None:
    """Verify queue_prompt posts payload and extracts prompt_id."""
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"prompt_id": "test-prompt-uuid-123"}
    mock_session.post.return_value = mock_resp

    client = ComfyUIClient(server_address="127.0.0.1:8188", session=mock_session)
    prompt_id = client.queue_prompt(prompt={"node": 1}, client_id="client-456")

    assert prompt_id == "test-prompt-uuid-123"
    mock_session.post.assert_called_once()


def test_client_download_image(tmp_path: Path) -> None:
    """Verify download_image writes content cleanly with atomic rename."""
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_content.return_value = [b"chunk1", b"chunk2"]
    mock_session.get.return_value = mock_resp

    client = ComfyUIClient(server_address="127.0.0.1:8188", session=mock_session)
    asset = OutputAsset(filename="out.png", subfolder="", image_type="output")
    target = tmp_path / "downloaded.png"

    client.download_image(asset, target)

    assert target.is_file()
    assert target.read_bytes() == b"chunk1chunk2"
    assert not (tmp_path / "downloaded.png.part").exists()


def test_client_track_execution_success() -> None:
    """Verify WebSocket listener parses events, ignores binary frames, and tracks outputs."""
    prompt_id = "target-prompt-id"
    frames: list[str | bytes] = [
        # 1. Binary preview frame (should be ignored)
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00",
        # 2. Event for different prompt (should be ignored)
        json.dumps(
            {
                "type": "progress",
                "data": {"prompt_id": "other-id", "value": 1, "max": 10},
            }
        ),
        # 3. Matching progress event
        json.dumps(
            {
                "type": "progress",
                "data": {"prompt_id": prompt_id, "value": 5, "max": 25, "node": "12"},
            }
        ),
        # 4. Executed event with output images
        json.dumps(
            {
                "type": "executed",
                "data": {
                    "prompt_id": prompt_id,
                    "node": "14",
                    "output": {
                        "images": [
                            {
                                "filename": "campaign_latam_m01_00001_.png",
                                "subfolder": "",
                                "type": "output",
                            }
                        ]
                    },
                },
            }
        ),
        # 5. Executing event with node=None (workflow completion signal)
        json.dumps({"type": "executing", "data": {"prompt_id": prompt_id, "node": None}}),
    ]

    mock_ws = MockWebSocket(frames)
    client = ComfyUIClient(
        server_address="127.0.0.1:8188",
        ws_factory=lambda url, timeout: mock_ws,
    )

    progress_records: list[tuple[int, int, str]] = []

    def on_progress(cur: int, total: int, node: str) -> None:
        progress_records.append((cur, total, node))

    outputs = client.track_execution(
        prompt_id=prompt_id,
        client_id="test-client",
        timeout_seconds=5.0,
        on_progress=on_progress,
    )

    assert len(outputs) == 1
    assert outputs[0].filename == "campaign_latam_m01_00001_.png"
    assert len(progress_records) == 1
    assert progress_records[0] == (5, 25, "12")
    assert mock_ws.closed


def test_client_track_execution_error() -> None:
    """Verify WebSocket listener raises ComfyExecutionError on execution_error event."""
    prompt_id = "failing-prompt-id"
    frames: list[str | bytes] = [
        json.dumps(
            {
                "type": "execution_error",
                "data": {
                    "prompt_id": prompt_id,
                    "node_id": "12",
                    "node_type": "KSampler",
                    "exception_message": "CUDA out of memory",
                    "exception_type": "torch.cuda.OutOfMemoryError",
                },
            }
        ),
    ]

    mock_ws = MockWebSocket(frames)
    client = ComfyUIClient(
        server_address="127.0.0.1:8188",
        ws_factory=lambda url, timeout: mock_ws,
    )

    with pytest.raises(ComfyExecutionError, match="CUDA out of memory"):
        client.track_execution(
            prompt_id=prompt_id,
            client_id="test-client",
            timeout_seconds=5.0,
        )


def test_client_track_execution_timeout() -> None:
    """Verify track_execution aborts and raises ComfyTimeoutError if duration exceeded."""
    prompt_id = "hanging-prompt-id"
    frames: list[str | bytes] = [
        json.dumps({"type": "status", "data": {"prompt_id": prompt_id}}),
    ]

    mock_ws = MockWebSocket(frames)
    client = ComfyUIClient(
        server_address="127.0.0.1:8188",
        ws_factory=lambda url, timeout: mock_ws,
    )

    # Use timeout_seconds = -1 to trigger immediate timeout
    with pytest.raises(ComfyTimeoutError, match="timeout limit"):
        client.track_execution(
            prompt_id=prompt_id,
            client_id="test-client",
            timeout_seconds=-0.1,
        )


# ============================================================================
# PHASE 4 TESTS: End-to-End Batch Orchestration Mock Tests
# ============================================================================


def test_batch_swapper_end_to_end_mock(tmp_path: Path) -> None:
    """Test full sequential batch execution across multiple models with mocked server."""
    # Setup directories and mock files
    models_dir = tmp_path / "input_models"
    models_dir.mkdir()
    (models_dir / "model_01.png").write_bytes(b"portrait_01")
    (models_dir / "model_02.png").write_bytes(b"portrait_02")

    campaign_file = tmp_path / "campaign.png"
    campaign_file.write_bytes(b"campaign_bytes")

    mask_file = tmp_path / "mask.png"
    mask_file.write_bytes(b"mask_bytes")

    output_dir = tmp_path / "output"

    workflow_file = tmp_path / "workflow.json"
    workflow_content = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "{{CAMPAIGN_IMG}}"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "{{MASK_IMG}}"}},
        "3": {"class_type": "LoadImage", "inputs": {"image": "{{MODEL_IMG}}"}},
        "14": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "{{OUTPUT_PREFIX}}"},
        },
    }
    workflow_file.write_text(json.dumps(workflow_content), encoding="utf-8")

    config = SwapperConfig(
        server_address="127.0.0.1:8188",
        campaign_path=campaign_file,
        mask_path=mask_file,
        models_dir=models_dir,
        output_dir=output_dir,
        market_tag="emea",
        workflow_path=workflow_file,
        timeout_seconds=10.0,
    )

    # Mock ComfyUI Client
    mock_client = MagicMock(spec=ComfyUIClient)
    mock_client.upload_image.side_effect = lambda path: f"uploaded_{path.name}"
    mock_client.queue_prompt.side_effect = lambda prompt, client_id: f"prompt_{client_id[:8]}"
    mock_client.track_execution.return_value = [
        OutputAsset(filename="out_mock.png", subfolder="", image_type="output")
    ]

    def mock_download(asset: OutputAsset, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"rendered_model_result")

    mock_client.download_image.side_effect = mock_download

    swapper = BatchSwapper(config=config, client=mock_client)
    results = swapper.run_batch()

    assert len(results) == 2
    assert all(r.status == JobStatus.SUCCESS for r in results)
    assert (output_dir / "campaign_emea_model_01.png").is_file()
    assert (output_dir / "campaign_emea_model_02.png").is_file()
    assert mock_client.free_memory.call_count == 2


def test_batch_swapper_handles_partial_failure(tmp_path: Path) -> None:
    """Verify that if one model fails, remaining models in batch continue processing."""
    models_dir = tmp_path / "input_models"
    models_dir.mkdir()
    (models_dir / "model_ok.png").write_bytes(b"portrait_ok")
    (models_dir / "model_fail.png").write_bytes(b"portrait_fail")

    campaign_file = tmp_path / "campaign.png"
    campaign_file.write_bytes(b"campaign_bytes")

    output_dir = tmp_path / "output"
    workflow_file = tmp_path / "workflow.json"
    workflow_file.write_text(
        json.dumps(
            {
                "1": {"class_type": "LoadImage", "inputs": {"image": "c"}},
                "3": {"class_type": "LoadImage", "inputs": {"image": "m"}},
                "14": {"class_type": "SaveImage", "inputs": {"filename_prefix": "o"}},
            }
        ),
        encoding="utf-8",
    )

    config = SwapperConfig(
        server_address="127.0.0.1:8188",
        campaign_path=campaign_file,
        mask_path=None,
        models_dir=models_dir,
        output_dir=output_dir,
        market_tag="apac",
        workflow_path=workflow_file,
    )

    mock_client = MagicMock(spec=ComfyUIClient)
    mock_client.upload_image.return_value = "img.png"
    mock_client.queue_prompt.return_value = "prompt-1"

    def mock_track(prompt_id: str, client_id: str, timeout_seconds: float, on_progress=None):
        # We can distinguish by inspecting the queue_prompt calls or a counter
        if mock_client.track_execution.call_count == 1:
            raise ComfyExecutionError("OOM on model_fail")
        return [OutputAsset(filename="ok.png", subfolder="", image_type="output")]

    mock_client.track_execution.side_effect = mock_track

    def mock_download(asset: OutputAsset, destination: Path) -> None:
        destination.write_bytes(b"ok")

    mock_client.download_image.side_effect = mock_download

    swapper = BatchSwapper(config=config, client=mock_client)
    results = swapper.run_batch()

    assert len(results) == 2
    failed_results = [r for r in results if r.status == JobStatus.FAILED]
    success_results = [r for r in results if r.status == JobStatus.SUCCESS]

    assert len(failed_results) == 1
    assert "OOM on model_fail" in (failed_results[0].error_message or "")
    assert len(success_results) == 1


def test_parse_cli_args() -> None:
    """Verify CLI argument parser defaults and flags."""
    args = parse_cli_args(
        [
            "--campaign",
            "data/campaign.png",
            "--models-dir",
            "data/models",
            "--market-tag",
            "latam",
            "--timeout",
            "120",
        ]
    )
    assert args.campaign == Path("data/campaign.png")
    assert args.models_dir == Path("data/models")
    assert args.market_tag == "latam"
    assert args.timeout == 120.0
    assert args.server == "127.0.0.1:8188"
    assert args.filter == "*"
    assert args.workflow == Path("workflows/model_swap_qwen21_maskless_api.json")

    args_custom = parse_cli_args(
        [
            "--campaign",
            "c.png",
            "--models-dir",
            "m",
            "--filter",
            "*_f*",
        ]
    )
    assert args_custom.filter == "*_f*"


def test_workflow_template_non_dict_node_raises_type_error() -> None:
    """Verify TypeError is raised when a node is not a dictionary."""
    with pytest.raises(TypeError, match="definition must be a dictionary"):
        WorkflowTemplate({"1": "not-a-dict"})  # type: ignore[arg-type]


def test_workflow_template_file_not_found(tmp_path: Path) -> None:
    """Verify FileNotFoundError is raised when workflow template path is missing."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        WorkflowTemplate.load_from_file(tmp_path / "nonexistent.json")


def test_client_upload_image_not_found(tmp_path: Path) -> None:
    """Verify upload_image raises FileNotFoundError for missing input file."""
    client = ComfyUIClient(server_address="127.0.0.1:8188")
    with pytest.raises(FileNotFoundError, match="Input file not found"):
        client.upload_image(tmp_path / "ghost.png")


def test_client_queue_prompt_errors() -> None:
    """Verify queue_prompt raises ComfyAPIError on HTTP failure or missing prompt_id."""
    mock_session = MagicMock()
    # 1. HTTP 500 error
    mock_resp_fail = MagicMock()
    mock_resp_fail.status_code = 500
    mock_resp_fail.text = "Internal Server Error"
    mock_session.post.return_value = mock_resp_fail

    client = ComfyUIClient(server_address="127.0.0.1:8188", session=mock_session)
    with pytest.raises(ComfyAPIError, match="Failed to queue prompt"):
        client.queue_prompt({"1": {}}, client_id="c1")

    # 2. Missing prompt_id in response
    mock_resp_missing = MagicMock()
    mock_resp_missing.status_code = 200
    mock_resp_missing.json.return_value = {"other_key": 123}
    mock_session.post.return_value = mock_resp_missing
    with pytest.raises(ComfyAPIError, match="Missing prompt_id"):
        client.queue_prompt({"1": {}}, client_id="c1")


def test_client_download_image_error(tmp_path: Path) -> None:
    """Verify download_image raises ComfyAPIError when server returns 404."""
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.text = "Not Found"
    mock_session.get.return_value = mock_resp

    client = ComfyUIClient(server_address="127.0.0.1:8188", session=mock_session)
    asset = OutputAsset(filename="missing.png", subfolder="", image_type="output")
    with pytest.raises(ComfyAPIError, match="Failed to download asset"):
        client.download_image(asset, tmp_path / "dest.png")


def test_client_interrupt_and_free() -> None:
    """Verify interrupt and free memory requests execute cleanly."""
    mock_session = MagicMock()
    client = ComfyUIClient(server_address="127.0.0.1:8188", session=mock_session)

    client.interrupt_execution()
    assert mock_session.post.call_count == 1

    client.free_memory()
    assert mock_session.post.call_count == 2


def test_batch_swapper_empty_models_dir(tmp_path: Path) -> None:
    """Verify run_batch returns empty list when models directory has no image files."""
    empty_dir = tmp_path / "empty_models"
    empty_dir.mkdir()
    workflow_file = tmp_path / "wf.json"
    workflow_file.write_text(
        json.dumps({"1": {"class_type": "LoadImage", "inputs": {}}}), encoding="utf-8"
    )

    campaign_file = tmp_path / "c.png"
    campaign_file.write_bytes(b"dummy_campaign")

    config = SwapperConfig(
        server_address="127.0.0.1:8188",
        campaign_path=campaign_file,
        mask_path=None,
        models_dir=empty_dir,
        output_dir=tmp_path / "out",
        market_tag="apac",
        workflow_path=workflow_file,
    )
    swapper = BatchSwapper(config=config, client=MagicMock())
    results = swapper.run_batch()
    assert results == []


def test_batch_swapper_zero_outputs_error(tmp_path: Path) -> None:
    """Verify process_single_model fails if execution outputs zero images."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "m1.png").write_bytes(b"model")

    campaign_file = tmp_path / "c.png"
    campaign_file.write_bytes(b"dummy_campaign")

    workflow_file = tmp_path / "wf.json"
    workflow_file.write_text(
        json.dumps(
            {
                "1": {"class_type": "LoadImage", "inputs": {}},
                "3": {"class_type": "LoadImage", "inputs": {}},
                "14": {"class_type": "SaveImage", "inputs": {}},
            }
        ),
        encoding="utf-8",
    )

    config = SwapperConfig(
        server_address="127.0.0.1:8188",
        campaign_path=campaign_file,
        mask_path=None,
        models_dir=models_dir,
        output_dir=tmp_path / "out",
        market_tag="emea",
        workflow_path=workflow_file,
    )

    mock_client = MagicMock(spec=ComfyUIClient)
    mock_client.upload_image.return_value = "uploaded.png"
    mock_client.queue_prompt.return_value = "prompt-1"
    mock_client.track_execution.return_value = []  # No outputs produced

    swapper = BatchSwapper(config=config, client=mock_client)
    res = swapper.process_single_model(
        model_file=models_dir / "m1.png",
        campaign_asset_name="c.png",
        mask_asset_name=None,
    )
    assert res.status == JobStatus.FAILED
    assert "zero output assets" in (res.error_message or "")


# ============================================================================
# PHASE 5 TESTS: Full-Body Dataset & Resampling Tests
# ============================================================================


def test_fullbody_campaign_assets() -> None:
    """Verify full-body campaign lookbooks have valid dimensions and formats without external mask requirements."""
    campaign_dir = Path("data/input_campaign")
    campaign_images = [
        "campaign_fashion_female.png",
        "campaign_fashion_male.png",
    ]
    for img_name in campaign_images:
        img_path = campaign_dir / img_name
        assert img_path.is_file()

        with Image.open(img_path) as im:
            assert im.size == (896, 1152)
            assert im.mode == "RGB"


def test_fullbody_model_assets() -> None:
    """Verify all diverse multi-ethnic target model assets are full-body 896x1152 images."""
    models_dir = Path("data/input_models")
    expected_models = [
        "model_east_asia_f01.png",
        "model_west_africa_m01.png",
        "model_nordic_f01.png",
        "model_south_america_m01.png",
    ]
    for m_name in expected_models:
        m_path = models_dir / m_name
        assert m_path.is_file()
        with Image.open(m_path) as im:
            assert im.size == (896, 1152)
            assert im.mode == "RGB"


def test_resize_and_save_utility(tmp_path: Path) -> None:
    """Verify resize_and_save standardizes any resolution cleanly to 896x1152."""
    from scripts.prepare_fullbody_dataset import resize_and_save

    src = tmp_path / "arbitrary.png"
    Image.new("RGB", (500, 700), color=(100, 150, 200)).save(src)
    dst = tmp_path / "standardized.png"

    resize_and_save(src, dst)
    assert dst.is_file()
    with Image.open(dst) as im:
        assert im.size == (896, 1152)


def test_swapper_config_validation_failures(tmp_path: Path) -> None:
    """Verify SwapperConfig.__post_init__ catches invalid inputs fail-fast."""
    valid_file = tmp_path / "valid.png"
    valid_file.write_bytes(b"data")
    valid_wf = tmp_path / "wf.json"
    valid_wf.write_bytes(b"{}")
    valid_dir = tmp_path / "models"
    valid_dir.mkdir()

    # 1. Non-existent campaign
    with pytest.raises(FileNotFoundError, match="Campaign image does not exist"):
        SwapperConfig(
            server_address="127.0.0.1:8188",
            campaign_path=tmp_path / "nonexistent.png",
            models_dir=valid_dir,
            output_dir=tmp_path / "out",
            workflow_path=valid_wf,
        )

    # 2. Non-existent models directory
    with pytest.raises(NotADirectoryError, match="Models directory does not exist"):
        SwapperConfig(
            server_address="127.0.0.1:8188",
            campaign_path=valid_file,
            models_dir=tmp_path / "nonexistent_dir",
            output_dir=tmp_path / "out",
            workflow_path=valid_wf,
        )

    # 3. Non-existent workflow template
    with pytest.raises(FileNotFoundError, match="Workflow file does not exist"):
        SwapperConfig(
            server_address="127.0.0.1:8188",
            campaign_path=valid_file,
            models_dir=valid_dir,
            output_dir=tmp_path / "out",
            workflow_path=tmp_path / "nonexistent_wf.json",
        )

    # 4. Non-existent mask when explicitly specified
    with pytest.raises(FileNotFoundError, match="Mask file does not exist"):
        SwapperConfig(
            server_address="127.0.0.1:8188",
            campaign_path=valid_file,
            mask_path=tmp_path / "nonexistent_mask.png",
            models_dir=valid_dir,
            output_dir=tmp_path / "out",
            workflow_path=valid_wf,
        )

    # 5. Invalid non-positive timeout
    with pytest.raises(ValueError, match="Timeout must be positive"):
        SwapperConfig(
            server_address="127.0.0.1:8188",
            campaign_path=valid_file,
            models_dir=valid_dir,
            output_dir=tmp_path / "out",
            workflow_path=valid_wf,
            timeout_seconds=0.0,
        )


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


def test_client_context_manager_and_close() -> None:
    """Verify ComfyUIClient context manager closes the underlying HTTP session cleanly."""
    mock_session = MagicMock()
    with ComfyUIClient(server_address="127.0.0.1:8188", session=mock_session) as client:
        assert client._session is mock_session
    mock_session.close.assert_called_once()
    # Calling close again should invoke close on session
    client.close()
    assert mock_session.close.call_count == 2


def test_client_free_memory_with_unload_models() -> None:
    """Verify free_memory passes unload_models parameter to ComfyUI /free endpoint."""
    client = ComfyUIClient(server_address="127.0.0.1:8188")
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_session.post.return_value = mock_resp
    client._session = mock_session

    client.free_memory(unload_models=True)
    mock_session.post.assert_called_once_with(
        "http://127.0.0.1:8188/free",
        json={"unload_models": True, "free_memory": True},
        timeout=5.0,
    )


def test_render_progress_bar_non_tty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify progress bar logs structured 25% milestone updates in CI / non-TTY environments."""
    campaign_file = tmp_path / "c.png"
    campaign_file.write_bytes(b"data")
    wf_file = tmp_path / "wf.json"
    wf_file.write_text(
        json.dumps({"1": {"class_type": "LoadImage", "inputs": {}}}), encoding="utf-8"
    )
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    config = SwapperConfig(
        server_address="127.0.0.1:8188",
        campaign_path=campaign_file,
        models_dir=models_dir,
        output_dir=tmp_path / "out",
        workflow_path=wf_file,
    )
    swapper = BatchSwapper(config=config, client=MagicMock())

    import sys

    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    with caplog.at_level(logging.INFO):
        # 5 out of 20 is 25.0% -> triggers milestone log
        swapper._render_progress_bar(5, 20, "node_1")
        assert any("25.0%" in record.message for record in caplog.records)


def test_prepare_dataset_full_flow(tmp_path: Path) -> None:
    """Verify process_dataset in scripts.prepare_fullbody_dataset correctly ingests source assets."""
    from scripts.prepare_fullbody_dataset import process_dataset

    # Nonexistent dir returns 0
    assert process_dataset(tmp_path / "nonexistent", tmp_path / "out_data") == 0

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    out_data = tmp_path / "out_data"

    # Create dummy artifact images
    dummy_files = [
        "campaign_editorial_female_1.jpg",
        "campaign_editorial_male_2.png",
        "model_east_asia_f_3.jpg",
        "model_west_africa_m_4.png",
        "model_nordic_f_5.jpg",
        "model_south_america_m_6.png",
    ]
    for name in dummy_files:
        f = artifacts_dir / name
        Image.new("RGB", (400, 600), color=(128, 128, 128)).save(f)

    count = process_dataset(artifacts_dir, out_data)
    assert count == 6

    # Verify standardized outputs
    assert (out_data / "input_campaign" / "campaign_fashion_female.png").is_file()
    assert (out_data / "input_campaign" / "campaign_fashion_male.png").is_file()
    assert (out_data / "input_models" / "model_east_asia_f01.png").is_file()
    assert (out_data / "input_models" / "model_west_africa_m01.png").is_file()
    assert (out_data / "input_models" / "model_nordic_f01.png").is_file()
    assert (out_data / "input_models" / "model_south_america_m01.png").is_file()


def test_prepare_dataset_cli_args() -> None:
    """Verify parse_cli_args in prepare_fullbody_dataset parses paths cleanly."""
    from scripts.prepare_fullbody_dataset import parse_cli_args

    args = parse_cli_args(["--source-dir", "custom/src", "--output-dir", "custom/dst"])
    assert args.source_dir == Path("custom/src")
    assert args.output_dir == Path("custom/dst")


def test_configure_logging() -> None:
    """Verify configure_logging initializes root logger without error."""
    from scripts.batch_swapper import configure_logging

    configure_logging(level=logging.DEBUG)
    assert logging.getLogger().level == logging.DEBUG


def test_batch_swapper_main_zero_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify main() CLI exits with code 2 when zero models match."""
    from scripts.batch_swapper import main as batch_main

    campaign = tmp_path / "c.png"
    campaign.write_bytes(b"data")
    models = tmp_path / "models"
    models.mkdir()
    wf = tmp_path / "wf.json"
    wf.write_text(json.dumps({"1": {"class_type": "LoadImage", "inputs": {}}}), encoding="utf-8")

    test_args = [
        "batch_swapper.py",
        "--campaign",
        str(campaign),
        "--models-dir",
        str(models),
        "--output-dir",
        str(tmp_path / "out"),
        "--workflow",
        str(wf),
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    with pytest.raises(SystemExit) as exc_info:
        batch_main()
    assert exc_info.value.code == 2


def test_generate_testdata_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify scripts.generate_testdata main entrypoint runs smoothly."""
    from scripts.generate_testdata import main as gen_main

    out_dir = tmp_path / "test_data"
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate_testdata.py", "--output-dir", str(out_dir), "--size", "64"],
    )
    gen_main()
    assert (out_dir / "input_campaign" / "campaign_summer_lookbook.png").is_file()


def test_prepare_fullbody_dataset_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify scripts.prepare_fullbody_dataset main entrypoint handles exit codes."""
    from scripts.prepare_fullbody_dataset import main as prep_main

    empty_src = tmp_path / "empty_artifacts"
    empty_src.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_fullbody_dataset.py",
            "--source-dir",
            str(empty_src),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        prep_main()
    assert exc_info.value.code == 1
