"""Unit tests for media_handlers.py module."""
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from PIL import Image
import io
import base64
import json
import os
import shutil
import subprocess
from types import SimpleNamespace
from homeassistant.exceptions import ServiceValidationError
from custom_components.llmvision.media_handlers import MediaProcessor


def _make_jpeg_bytes(color):
    """Create JPEG bytes for a solid-color test image."""
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), color=color).save(buffer, format="JPEG")
    return buffer.getvalue()


class _AsyncResponseContext:
    """Minimal async context manager for aiohttp-style responses."""

    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _StreamingContent:
    """Streaming response content stub."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    async def iter_any(self):
        for chunk in self._chunks:
            yield chunk


class _FakeAsyncWriter:
    """Async file writer compatible with aiofile.async_open usage."""

    def __init__(self, path):
        self._path = path
        self._file = None

    async def __aenter__(self):
        self._file = open(self._path, "wb")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._file is not None:
            self._file.close()
        return False

    async def write(self, data):
        if self._file is None:
            raise RuntimeError("Writer not opened")
        self._file.write(data)


class TestMediaProcessor:
    """Test MediaProcessor class."""

    @pytest.fixture
    def mock_hass(self):
        """Create a mock Home Assistant instance."""
        hass = Mock()
        hass.loop = Mock()
        hass.loop.run_in_executor = AsyncMock()
        hass.states = Mock()
        hass.states.get = Mock()
        hass.config = Mock()
        hass.config.path = Mock(return_value="/mock/path")
        return hass

    @pytest.fixture
    def mock_client(self):
        """Create a mock client."""
        client = Mock()
        client.add_frame = Mock()
        return client

    @pytest.fixture
    def processor(self, mock_hass, mock_client):
        """Create a MediaProcessor instance."""
        with patch('custom_components.llmvision.media_handlers.async_get_clientsession'):
            return MediaProcessor(mock_hass, mock_client)

    def test_init(self, processor, mock_hass, mock_client):
        """Test MediaProcessor initialization."""
        assert processor.hass == mock_hass
        assert processor.client == mock_client
        assert processor.base64_images == []
        assert processor.filenames == []
        assert processor.key_frame == ""

    @pytest.mark.asyncio
    async def test_encode_image(self, processor):
        """Test _encode_image method."""
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='red')

        result = await processor._encode_image(img)

        assert isinstance(result, str)
        assert len(result) > 0
        # Verify it's valid base64
        base64.b64decode(result)

    @pytest.mark.asyncio
    async def test_save_clip_writes_base64_image(self, processor, tmp_path):
        """_save_clip should decode base64 image payloads before writing."""
        processor.hass.loop.run_in_executor.side_effect = (
            lambda _executor, func, *args: func(*args)
        )
        output = tmp_path / "snapshot.jpg"

        with patch("custom_components.llmvision.media_handlers.os.makedirs"):
            await processor._save_clip(
                image_data=base64.b64encode(b"image-bytes").decode("utf-8"),
                image_path=str(output),
            )

        assert output.read_bytes() == b"image-bytes"

    @pytest.mark.asyncio
    async def test_expose_image_uses_frame_path_once(self, processor, tmp_path):
        """_expose_image should derive image data from a file only for the first keyframe."""
        processor.hass.loop.run_in_executor.side_effect = (
            lambda _executor, func, *args: func(*args)
        )
        source = tmp_path / "frame.jpg"
        source.write_bytes(_make_jpeg_bytes("navy"))
        processor._save_clip = AsyncMock()

        with patch("custom_components.llmvision.media_handlers.os.makedirs"):
            await processor._expose_image(
                frame_name="7",
                image_data=None,
                uid="deadbeef",
                frame_path=str(source),
            )
            await processor._expose_image(
                frame_name="8",
                image_data="ignored",
                uid="second",
            )

        assert processor.key_frame.endswith("deadbeef-7.jpg")
        processor._save_clip.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_select_keyframe_index_picks_lowest_similarity(self, processor):
        """_select_keyframe_index should return the most dissimilar candidate."""
        processor.hass.loop.run_in_executor.side_effect = (
            lambda _executor, func, *args: func(*args)
        )
        processor._similarity_score = Mock(side_effect=[0.8, 0.2, 0.5])

        result = await processor._select_keyframe_index(
            _make_jpeg_bytes("black"),
            [
                _make_jpeg_bytes("gray"),
                _make_jpeg_bytes("white"),
                _make_jpeg_bytes("red"),
            ],
        )

        assert result == 1

    def test_convert_to_rgb_rgba(self, processor):
        """Test _convert_to_rgb with RGBA image."""
        img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))

        result = processor._convert_to_rgb(img)

        assert result.mode == 'RGB'

    def test_convert_to_rgb_already_rgb(self, processor):
        """Test _convert_to_rgb with RGB image."""
        img = Image.new('RGB', (100, 100), color='red')

        result = processor._convert_to_rgb(img)

        assert result.mode == 'RGB'

    @pytest.mark.asyncio
    async def test_resize_image_with_img(self, processor):
        """Test resize_image with PIL Image object."""
        # Create a test image
        img = Image.new('RGB', (200, 100), color='blue')

        result = await processor.resize_image(
            target_width=100,
            img=img
        )

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_resize_image_with_image_path(self, processor, tmp_path):
        """resize_image should support file-based inputs."""
        processor.hass.loop.run_in_executor.side_effect = (
            lambda _executor, func, *args: func(*args)
        )
        image_path = tmp_path / "photo.jpg"
        image_path.write_bytes(_make_jpeg_bytes("green"))

        result = await processor.resize_image(
            target_width=8, image_path=str(image_path)
        )

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_resize_image_requires_input(self, processor):
        """resize_image should reject empty inputs."""
        with pytest.raises(ServiceValidationError, match="No image data provided"):
            await processor.resize_image(target_width=10)

    def test_similarity_score(self, processor):
        """Test _similarity_score method."""
        # Create two similar images
        img1 = Image.new('L', (100, 100), color=128)
        img2 = Image.new('L', (100, 100), color=130)

        score = processor._similarity_score(img1, img2)

        assert isinstance(score, float)
        assert 0 <= score <= 1

    @pytest.mark.asyncio
    async def test_fetch_retries_until_success(self, processor):
        """_fetch should retry failed responses before succeeding."""
        first = SimpleNamespace(ok=False, status=500)
        second = SimpleNamespace(
            ok=True, status=200, read=AsyncMock(return_value=b"ok")
        )
        processor.session.get.side_effect = [
            _AsyncResponseContext(first),
            _AsyncResponseContext(second),
        ]

        with patch(
            "custom_components.llmvision.media_handlers.asyncio.sleep",
            AsyncMock(),
        ):
            result = await processor._fetch("http://camera/frame.jpg", retry_delay=0)

        assert result == b"ok"

    @pytest.mark.asyncio
    async def test_fetch_streams_to_target_file(self, processor, tmp_path):
        """_fetch should stream large responses to disk when target_file is provided."""
        target = tmp_path / "clip.mp4"
        response = SimpleNamespace(
            ok=True,
            status=200,
            content=_StreamingContent([b"abc", b"def"]),
        )
        processor.session.get.return_value = _AsyncResponseContext(response)

        with patch(
            "custom_components.llmvision.media_handlers.async_open",
            side_effect=lambda path, mode: _FakeAsyncWriter(path),
        ):
            result = await processor._fetch(
                "http://camera/clip.mp4",
                target_file=str(target),
            )

        assert result is None
        assert target.read_bytes() == b"abcdef"

    @pytest.mark.asyncio
    async def test_fetch_returns_none_after_exhausted_errors(self, processor):
        """_fetch should return None after repeated exceptions."""
        processor.session.get.side_effect = RuntimeError("boom")

        with patch(
            "custom_components.llmvision.media_handlers.asyncio.sleep",
            AsyncMock(),
        ):
            result = await processor._fetch(
                "http://camera/frame.jpg",
                max_retries=2,
                retry_delay=0,
                entity_name="front",
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_record_returns_frames_in_capture_order(self, processor):
        """Selected stream frames should be emitted in capture order."""
        clock = {"now": 0.0}

        async def fake_sleep(delay):
            clock["now"] += delay

        processor.hass.loop.run_in_executor.side_effect = (
            lambda _executor, func, *args: func(*args)
        )
        processor.hass.states.get.return_value = SimpleNamespace(
            attributes={"entity_picture": "/api/camera_proxy/camera.front"}
        )
        frame_iter = iter(
            [
                _make_jpeg_bytes("black"),
                _make_jpeg_bytes("gray"),
                _make_jpeg_bytes("white"),
            ]
        )

        async def fake_fetch(*args, **kwargs):
            frame = next(frame_iter, None)
            if frame is None:
                clock["now"] = 10.0
            return frame

        processor._fetch = AsyncMock(side_effect=fake_fetch)
        processor._similarity_score = Mock(side_effect=[0.9, 0.1])
        processor.resize_image = AsyncMock(
            side_effect=["encoded-0", "encoded-1", "encoded-2"]
        )
        processor._select_keyframe_index = AsyncMock(return_value=0)

        with patch(
            "custom_components.llmvision.media_handlers.get_url",
            return_value="http://ha.local",
        ), patch(
            "custom_components.llmvision.media_handlers.time.time",
            side_effect=lambda: clock["now"],
        ), patch(
            "custom_components.llmvision.media_handlers.asyncio.sleep",
            side_effect=fake_sleep,
        ):
            await processor.record(
                image_entities=["camera.front"],
                duration=2.5,
                max_frames=3,
                target_width=128,
                include_filename=False,
                expose_images=False,
            )

        assert [
            call.kwargs["filename"]
            for call in processor.client.add_frame.call_args_list
        ] == ["camera0-frame-0", "camera0-frame-1", "camera0-frame-2"]
        assert [
            call.kwargs["base64_image"]
            for call in processor.client.add_frame.call_args_list
        ] == ["encoded-0", "encoded-1", "encoded-2"]

    @pytest.mark.asyncio
    async def test_record_raises_when_no_cameras_available(self, processor):
        """record should fail when all cameras are unavailable."""
        clock = {"now": 0.0}

        async def fake_sleep(delay):
            clock["now"] += delay

        processor.hass.states.get.return_value = None

        with patch(
            "custom_components.llmvision.media_handlers.get_url",
            return_value="http://ha.local",
        ), patch(
            "custom_components.llmvision.media_handlers.time.time",
            side_effect=lambda: clock["now"],
        ), patch(
            "custom_components.llmvision.media_handlers.asyncio.sleep",
            side_effect=fake_sleep,
        ), pytest.raises(
            ServiceValidationError, match="No cameras available"
        ):
            await processor.record(
                image_entities=["camera.front"],
                duration=0.1,
                max_frames=1,
                target_width=64,
                include_filename=False,
                expose_images=False,
            )

    @pytest.mark.asyncio
    async def test_add_images_supports_entities_and_paths(self, processor, tmp_path):
        """add_images should add frames from both camera entities and file paths."""
        image_path = tmp_path / "still.jpg"
        image_path.write_bytes(_make_jpeg_bytes("teal"))
        entity_state = SimpleNamespace(
            attributes={
                "entity_picture": "/api/camera_proxy/camera.front",
                "friendly_name": "Front Door",
            }
        )
        processor.hass.states.get.return_value = entity_state
        processor._fetch = AsyncMock(return_value=b"entity-bytes")
        processor.resize_image = AsyncMock(side_effect=["entity-frame", "path-frame"])
        processor._expose_image = AsyncMock()

        with patch(
            "custom_components.llmvision.media_handlers.get_url",
            return_value="http://ha.local",
        ):
            result = await processor.add_images(
                image_entities=["camera.front"],
                image_paths=[str(image_path)],
                target_width=128,
                include_filename=True,
                expose_images=True,
            )

        assert result is processor.client
        assert [
            call.kwargs["filename"]
            for call in processor.client.add_frame.call_args_list
        ] == ["Front Door", "still"]
        assert processor._expose_image.await_count == 2

    @pytest.mark.asyncio
    async def test_add_images_raises_for_missing_file(self, processor):
        """add_images should reject missing local image files."""
        with patch(
            "custom_components.llmvision.media_handlers.get_url",
            return_value="http://ha.local",
        ), pytest.raises(ServiceValidationError, match="does not exist"):
            await processor.add_images(
                image_entities=[],
                image_paths=["/missing/file.jpg"],
                target_width=128,
                include_filename=False,
                expose_images=False,
            )

    @pytest.mark.asyncio
    async def test_add_images_raises_when_all_cameras_unavailable(self, processor):
        """add_images should fail when no camera image can be obtained."""
        processor.hass.states.get.return_value = None

        with patch(
            "custom_components.llmvision.media_handlers.get_url",
            return_value="http://ha.local",
        ), pytest.raises(ServiceValidationError, match="No cameras available"):
            await processor.add_images(
                image_entities=["camera.front"],
                image_paths=[],
                target_width=128,
                include_filename=False,
                expose_images=False,
            )

    @pytest.mark.asyncio
    async def test_add_images_wraps_attribute_errors(self, processor):
        """add_images should surface malformed entity state access as validation errors."""
        processor.hass.states.get.return_value = object()

        with patch(
            "custom_components.llmvision.media_handlers.get_url",
            return_value="http://ha.local",
        ), pytest.raises(ServiceValidationError, match="Error accessing camera entity"):
            await processor.add_images(
                image_entities=["camera.front"],
                image_paths=[],
                target_width=128,
                include_filename=False,
                expose_images=False,
            )

    @pytest.mark.asyncio
    async def test_add_video_returns_selected_frames_in_capture_order(self, processor):
        """Selected video keyframes should be added in capture order."""
        processor.hass.loop.run_in_executor.side_effect = (
            lambda _executor, func, *args: func(*args)
        )
        frame_bytes = [
            _make_jpeg_bytes("black"),
            _make_jpeg_bytes("gray"),
            _make_jpeg_bytes("white"),
        ]
        encoded_by_frame = {
            frame_bytes[0]: "encoded-0",
            frame_bytes[1]: "encoded-1",
            frame_bytes[2]: "encoded-2",
        }

        class FakeStdout:
            def __init__(self, chunks):
                self._chunks = list(chunks)

            async def read(self, _size):
                if self._chunks:
                    return self._chunks.pop(0)
                return b""

        class FakeProcess:
            def __init__(self, payload):
                self.pid = 1234
                self.stdout = FakeStdout([payload, b""])
                self.stderr = None
                self.returncode = 0

            async def wait(self):
                return self.returncode

        processor._similarity_score = Mock(side_effect=[0.9, 0.1])
        processor.resize_image = AsyncMock(
            side_effect=lambda target_width, image_data=None, **kwargs: encoded_by_frame[
                image_data or b""
            ]
        )

        with patch(
            "custom_components.llmvision.media_handlers.asyncio.create_subprocess_exec",
            AsyncMock(return_value=FakeProcess(b"".join(frame_bytes))),
        ):
            await processor.add_video(
                video_path="/tmp/test_clip.mp4",
                base_url="http://ha.local",
                max_frames=3,
                target_width=128,
                include_filename=False,
                expose_images=False,
            )

        assert [
            call.kwargs["base64_image"]
            for call in processor.client.add_frame.call_args_list
        ] == ["encoded-0", "encoded-1", "encoded-2"]

    @pytest.mark.asyncio
    async def test_add_video_signs_api_path_and_exposes_keyframe(self, processor):
        """add_video should sign local API paths and expose the selected keyframe."""
        processor.hass.loop.run_in_executor.side_effect = (
            lambda _executor, func, *args: func(*args)
        )
        frame_bytes = [_make_jpeg_bytes("black")]
        temp_video = "/tmp/signed-video.mp4"

        class FakeStdout:
            def __init__(self, chunks):
                self._chunks = list(chunks)

            async def read(self, _size):
                if self._chunks:
                    return self._chunks.pop(0)
                return b""

        class FakeProcess:
            def __init__(self):
                self.pid = 55
                self.stdout = FakeStdout([b"".join(frame_bytes), b""])
                self.stderr = None
                self.returncode = 0

            async def wait(self):
                return self.returncode

        class FakeTempFile:
            name = temp_video

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        processor.resize_image = AsyncMock(return_value="encoded-frame")
        processor._select_keyframe_index = AsyncMock(return_value=0)
        processor._expose_image = AsyncMock()
        processor._fetch = AsyncMock(return_value=None)

        with patch(
            "custom_components.llmvision.media_handlers.async_sign_path",
            return_value="/api/signed/path?authSig=1",
        ), patch(
            "custom_components.llmvision.media_handlers.tempfile.NamedTemporaryFile",
            return_value=FakeTempFile(),
        ), patch(
            "custom_components.llmvision.media_handlers.os.path.exists",
            side_effect=lambda path: path == temp_video,
        ), patch(
            "custom_components.llmvision.media_handlers.os.path.getsize",
            return_value=10,
        ), patch(
            "custom_components.llmvision.media_handlers.os.remove",
        ), patch(
            "custom_components.llmvision.media_handlers.asyncio.create_subprocess_exec",
            AsyncMock(return_value=FakeProcess()),
        ):
            await processor.add_video(
                video_path="/api/frigate/notifications/test/clip.mp4",
                base_url="http://ha.local",
                max_frames=1,
                target_width=128,
                include_filename=True,
                expose_images=True,
            )

        assert (
            processor.client.add_frame.call_args.kwargs["filename"]
            == "path?authSig=1 (frame 1)"
        )
        processor._expose_image.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_add_video_raises_when_http_download_is_empty(
        self, processor, tmp_path
    ):
        """add_video should fail when the downloaded remote clip is empty."""
        temp_video = tmp_path / "download.mp4"

        class FakeTempFile:
            def __init__(self, path):
                self.name = str(path)

            def __enter__(self):
                temp_video.write_bytes(b"")
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        processor._fetch = AsyncMock(return_value=None)

        with patch(
            "custom_components.llmvision.media_handlers.tempfile.NamedTemporaryFile",
            return_value=FakeTempFile(temp_video),
        ), pytest.raises(ServiceValidationError, match="Failed to fetch video"):
            await processor.add_video(
                video_path="http://example.com/clip.mp4",
                base_url="http://ha.local",
                max_frames=1,
                target_width=128,
                include_filename=False,
                expose_images=False,
            )

    @pytest.mark.asyncio
    async def test_add_video_raises_when_no_frames_are_extracted(self, processor):
        """add_video should fail cleanly if ffmpeg yields no JPEG frames."""

        class FakeStdout:
            async def read(self, _size):
                return b""

        class FakeProcess:
            def __init__(self):
                self.pid = 44
                self.stdout = FakeStdout()
                self.stderr = None
                self.returncode = 0

            async def wait(self):
                return self.returncode

        with patch(
            "custom_components.llmvision.media_handlers.asyncio.create_subprocess_exec",
            AsyncMock(return_value=FakeProcess()),
        ), pytest.raises(ServiceValidationError, match="No frames extracted"):
            await processor.add_video(
                video_path="/tmp/empty.mp4",
                base_url="http://ha.local",
                max_frames=1,
                target_width=128,
                include_filename=False,
                expose_images=False,
            )

    @pytest.mark.asyncio
    async def test_add_video_raises_when_stdout_is_missing(self, processor):
        """add_video should reject ffmpeg processes without a stdout pipe."""

        class FakeProcess:
            def __init__(self):
                self.pid = 77
                self.stdout = None
                self.stderr = None
                self.returncode = 0

            async def wait(self):
                return self.returncode

        with patch(
            "custom_components.llmvision.media_handlers.asyncio.create_subprocess_exec",
            AsyncMock(return_value=FakeProcess()),
        ), pytest.raises(ServiceValidationError, match="stdout not available"):
            await processor.add_video(
                video_path="/tmp/no-stdout.mp4",
                base_url="http://ha.local",
                max_frames=1,
                target_width=128,
                include_filename=False,
                expose_images=False,
            )

    @staticmethod
    def _fake_ffmpeg(payload):
        """ffmpeg process stub that emits payload on stdout, then EOF."""

        class FakeStdout:
            def __init__(self):
                self._chunks = [payload, b""]

            async def read(self, _size):
                return self._chunks.pop(0) if self._chunks else b""

        class FakeProcess:
            pid = 99
            stderr = None
            returncode = 0

            def __init__(self):
                self.stdout = FakeStdout()

            async def wait(self):
                return self.returncode

        return FakeProcess()

    @staticmethod
    def _run_executor_inline(processor):
        processor.hass.loop.run_in_executor.side_effect = (
            lambda _executor, func, *args: func(*args)
        )

    @pytest.mark.asyncio
    async def test_add_video_variable_mode_uses_keyframes(self, processor):
        """Variable mode (the default) extracts I-frames with no frame limit."""
        self._run_executor_inline(processor)
        processor.resize_image = AsyncMock(return_value="encoded")
        exec_mock = AsyncMock(
            return_value=self._fake_ffmpeg(_make_jpeg_bytes("black"))
        )

        with patch(
            "custom_components.llmvision.media_handlers.asyncio.create_subprocess_exec",
            exec_mock,
        ):
            await processor.add_video(
                video_path="/tmp/clip.mp4", base_url="http://ha.local", max_frames=3
            )

        cmd = exec_mock.await_args.args
        assert cmd[cmd.index("-vf") + 1] == "select=eq(pict_type\\,I)"
        assert "-frames:v" not in cmd

    @pytest.mark.asyncio
    async def test_add_video_constant_mode_samples_at_fps_up_to_max_frames(
        self, processor
    ):
        """Constant mode samples at fps, scales, and stops ffmpeg after max_frames."""
        self._run_executor_inline(processor)
        processor.resize_image = AsyncMock(return_value="encoded")
        exec_mock = AsyncMock(
            return_value=self._fake_ffmpeg(_make_jpeg_bytes("black"))
        )

        with patch(
            "custom_components.llmvision.media_handlers.asyncio.create_subprocess_exec",
            exec_mock,
        ):
            await processor.add_video(
                video_path="/tmp/clip.mp4",
                base_url="http://ha.local",
                max_frames=30,
                target_width=640,
                fps_mode="constant",
                fps=0.5,
            )

        cmd = exec_mock.await_args.args
        assert cmd[cmd.index("-vf") + 1] == "fps=0.5,scale='trunc(min(iw,640)/2)*2':-2"
        assert cmd[cmd.index("-frames:v") + 1] == "30"

    @pytest.mark.asyncio
    async def test_add_video_constant_mode_keeps_every_frame_in_order(
        self, processor
    ):
        """Constant mode sends all frames in capture order, without scoring them."""
        self._run_executor_inline(processor)
        colors = ["black", "gray", "white", "red", "blue"]
        frames = [_make_jpeg_bytes(color) for color in colors]
        encoded_by_frame = {frame: color for frame, color in zip(frames, colors)}
        processor._similarity_score = Mock()
        processor.resize_image = AsyncMock(
            side_effect=lambda target_width, image_data=None, **kwargs: encoded_by_frame[
                image_data
            ]
        )

        with patch(
            "custom_components.llmvision.media_handlers.asyncio.create_subprocess_exec",
            AsyncMock(return_value=self._fake_ffmpeg(b"".join(frames))),
        ):
            await processor.add_video(
                video_path="/tmp/clip.mp4",
                base_url="http://ha.local",
                max_frames=4,  # ffmpeg enforces this; the extra frame checks the cap
                fps_mode="constant",
                fps=1,
            )

        processor._similarity_score.assert_not_called()
        assert [
            call.kwargs["base64_image"]
            for call in processor.client.add_frame.call_args_list
        ] == ["black", "gray", "white", "red"]

    @pytest.mark.asyncio
    async def test_add_video_variable_mode_scores_frames_beyond_max_frames(
        self, processor
    ):
        """In variable mode every keyframe is a candidate, not only the first max_frames."""
        self._run_executor_inline(processor)
        colors = ["black", "gray", "white", "red", "blue"]
        frames = [_make_jpeg_bytes(color) for color in colors]
        encoded_by_frame = {frame: color for frame, color in zip(frames, colors)}
        # Frames 1-4 are scored; "blue" (index 4) changed most
        processor._similarity_score = Mock(side_effect=[0.9, 0.8, 0.7, 0.1])
        processor.resize_image = AsyncMock(
            side_effect=lambda target_width, image_data=None, **kwargs: encoded_by_frame[
                image_data
            ]
        )

        # All five frames arrive in one stdout read
        with patch(
            "custom_components.llmvision.media_handlers.asyncio.create_subprocess_exec",
            AsyncMock(return_value=self._fake_ffmpeg(b"".join(frames))),
        ):
            await processor.add_video(
                video_path="/tmp/clip.mp4",
                base_url="http://ha.local",
                max_frames=2,
                target_width=128,
            )

        assert processor._similarity_score.call_count == 4
        assert [
            call.kwargs["base64_image"]
            for call in processor.client.add_frame.call_args_list
        ] == ["black", "blue"]

    @pytest.mark.asyncio
    async def test_add_video_constant_mode_attaches_frames_as_video(self, processor):
        """encode_video turns the extracted frames into a clip, alongside the images."""
        self._run_executor_inline(processor)
        frames = [_make_jpeg_bytes("black"), _make_jpeg_bytes("white")]
        processor.resize_image = AsyncMock(return_value="encoded-frame")
        processor._encode_video = AsyncMock(return_value=b"mp4-bytes")

        with patch(
            "custom_components.llmvision.media_handlers.asyncio.create_subprocess_exec",
            AsyncMock(return_value=self._fake_ffmpeg(b"".join(frames))),
        ):
            await processor.add_video(
                video_path="/media/clips/front_door.mp4",
                base_url="http://ha.local",
                max_frames=10,
                target_width=640,
                include_filename=True,
                fps_mode="constant",
                fps=2,
                encode_video=True,
            )

        # The raw ffmpeg JPEGs, not the resized base64 images
        processor._encode_video.assert_awaited_once_with(jpeg_frames=frames, fps=2)
        processor.client.add_video_clip.assert_called_once_with(
            base64_video=base64.b64encode(b"mp4-bytes").decode("utf-8"),
            filename="front_door",
            fps=2,
        )
        # Images are still added for providers without video support
        assert processor.client.add_frame.call_count == 2

    @pytest.mark.asyncio
    async def test_add_video_variable_mode_never_encodes_video(self, processor):
        """encode_video has no effect in variable mode."""
        self._run_executor_inline(processor)
        processor.resize_image = AsyncMock(return_value="encoded-frame")
        processor._encode_video = AsyncMock(return_value=b"mp4")

        with patch(
            "custom_components.llmvision.media_handlers.asyncio.create_subprocess_exec",
            AsyncMock(return_value=self._fake_ffmpeg(_make_jpeg_bytes("black"))),
        ):
            await processor.add_video(
                video_path="/tmp/clip.mp4",
                base_url="http://ha.local",
                encode_video=True,
            )

        processor._encode_video.assert_not_awaited()
        processor.client.add_video_clip.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_video_skips_clip_when_encoding_fails(self, processor):
        """A failed encode leaves only the images."""
        self._run_executor_inline(processor)
        processor.resize_image = AsyncMock(return_value="encoded-frame")
        processor._encode_video = AsyncMock(return_value=None)

        with patch(
            "custom_components.llmvision.media_handlers.asyncio.create_subprocess_exec",
            AsyncMock(return_value=self._fake_ffmpeg(_make_jpeg_bytes("black"))),
        ):
            await processor.add_video(
                video_path="/tmp/clip.mp4",
                base_url="http://ha.local",
                fps_mode="constant",
                encode_video=True,
            )

        processor.client.add_video_clip.assert_not_called()
        processor.client.add_frame.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_video_skips_clip_over_inline_limit(self, processor):
        """A clip too large to send inline is dropped in favour of the images."""
        self._run_executor_inline(processor)
        processor.resize_image = AsyncMock(return_value="encoded-frame")
        processor._encode_video = AsyncMock(return_value=b"x" * 30)

        with patch(
            "custom_components.llmvision.media_handlers.MAX_INLINE_VIDEO_BYTES", 32
        ), patch(
            "custom_components.llmvision.media_handlers.asyncio.create_subprocess_exec",
            AsyncMock(return_value=self._fake_ffmpeg(_make_jpeg_bytes("black"))),
        ):
            # 30 bytes is 40 characters of base64
            await processor.add_video(
                video_path="/tmp/clip.mp4",
                base_url="http://ha.local",
                fps_mode="constant",
                encode_video=True,
            )

        processor.client.add_video_clip.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_video_removes_downloaded_copy(self, processor, tmp_path):
        """A remote clip's downloaded copy is deleted once frames are taken."""
        self._run_executor_inline(processor)
        download = tmp_path / "download.mp4"

        class FakeTempFile:
            name = str(download)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        async def fake_fetch(_url, target_file=None, **kwargs):
            with open(target_file, "wb") as f:
                f.write(b"video")

        processor._fetch = fake_fetch
        processor._encode_video = AsyncMock(return_value=b"mp4")
        processor.resize_image = AsyncMock(return_value="encoded-frame")

        with patch(
            "custom_components.llmvision.media_handlers.tempfile.NamedTemporaryFile",
            return_value=FakeTempFile(),
        ), patch(
            "custom_components.llmvision.media_handlers.asyncio.create_subprocess_exec",
            AsyncMock(return_value=self._fake_ffmpeg(_make_jpeg_bytes("black"))),
        ):
            await processor.add_video(
                video_path="https://example.com/clip.mp4",
                base_url="http://ha.local",
                fps_mode="constant",
                encode_video=True,
            )

        assert not download.exists()
        processor.client.add_video_clip.assert_called_once()

    @pytest.mark.asyncio
    async def test_encode_video_returns_none_when_ffmpeg_fails(self, processor):
        """A non-zero ffmpeg exit returns None and removes the output file."""

        class FailedProcess:
            returncode = 1

            async def communicate(self, _input=None):
                return b"", b"Unknown encoder 'libx264'"

        created = []
        real_named_temp = __import__("tempfile").NamedTemporaryFile

        def tracking_temp(**kwargs):
            tmp = real_named_temp(**kwargs)
            created.append(tmp.name)
            return tmp

        with patch(
            "custom_components.llmvision.media_handlers.tempfile.NamedTemporaryFile",
            side_effect=tracking_temp,
        ), patch(
            "custom_components.llmvision.media_handlers.asyncio.create_subprocess_exec",
            AsyncMock(return_value=FailedProcess()),
        ):
            result = await processor._encode_video([_make_jpeg_bytes("black")], 1)

        assert result is None
        assert created and not os.path.exists(created[0])

    @pytest.mark.skipif(
        not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
        reason="needs ffmpeg and ffprobe",
    )
    @pytest.mark.asyncio
    async def test_constant_mode_with_real_ffmpeg(self, processor, tmp_path):
        """End to end: frames at fps, cut at max_frames, and the same frames as video."""
        self._run_executor_inline(processor)
        source = tmp_path / "source.mp4"
        # 6 seconds with audio, at an odd size
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "testsrc2=size=1281x721:rate=25:duration=6",
                "-f", "lavfi", "-i", "sine=duration=6",
                "-c:v", "libx264", "-c:a", "aac", "-shortest", str(source),
            ],
            check=True,
        )
        clips = []
        processor.client.add_video_clip = Mock(
            side_effect=lambda **kwargs: clips.append(kwargs)
        )

        await processor.add_video(
            video_path=str(source),
            base_url="http://ha.local",
            max_frames=3,
            target_width=640,
            fps_mode="constant",
            fps=1,
            encode_video=True,
        )

        assert processor.client.add_frame.call_count == 3
        sizes = {
            Image.open(io.BytesIO(base64.b64decode(c.kwargs["base64_image"]))).size
            for c in processor.client.add_frame.call_args_list
        }
        assert sizes == {(640, 360)}

        output = tmp_path / "output.mp4"
        output.write_bytes(base64.b64decode(clips[0]["base64_video"]))
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries",
                "stream=codec_type,width,height,r_frame_rate,nb_frames:format=duration",
                "-of", "json", str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        info = json.loads(probe.stdout)
        assert info["streams"] == [
            {
                "codec_type": "video",
                "width": 640,
                "height": 360,
                "r_frame_rate": "1/1",
                "nb_frames": "3",
            }
        ]
        # The first 3 seconds of the 6-second clip; the rest is dropped
        assert float(info["format"]["duration"]) == pytest.approx(3.0)

    @pytest.mark.asyncio
    async def test_add_videos_passes_mode_fps_and_encode_video(self, processor):
        """add_videos forwards fps_mode, fps and encode_video to each add_video call."""
        processor.add_video = AsyncMock()

        with patch(
            "custom_components.llmvision.media_handlers.get_url",
            return_value="http://ha.local",
        ):
            await processor.add_videos(
                video_paths=["/tmp/local.mp4"],
                event_ids=None,
                max_frames=2,
                target_width=128,
                include_filename=False,
                expose_images=False,
                fps_mode="constant",
                fps=1,
                encode_video=True,
            )

        kwargs = processor.add_video.await_args.kwargs
        assert kwargs["fps_mode"] == "constant"
        assert kwargs["fps"] == 1
        assert kwargs["encode_video"] is True

    @pytest.mark.asyncio
    async def test_add_videos_processes_event_ids(self, processor):
        """add_videos should convert event ids into Frigate clip URLs."""
        processor.add_video = AsyncMock()

        with patch(
            "custom_components.llmvision.media_handlers.get_url",
            return_value="http://ha.local",
        ):
            result = await processor.add_videos(
                video_paths=["/tmp/local.mp4"],
                event_ids=["abc123"],
                max_frames=2,
                target_width=128,
                include_filename=False,
                expose_images=False,
            )

        assert result is processor.client
        assert processor.add_video.await_count == 2
        assert any(
            call.kwargs["video_path"] == "/api/frigate/notifications/abc123/clip.mp4"
            for call in processor.add_video.await_args_list
        )

    @pytest.mark.asyncio
    async def test_add_streams_delegates_to_record(self, processor):
        """add_streams should proxy to record when image entities are present."""
        processor.record = AsyncMock()

        result = await processor.add_streams(
            image_entities=["camera.front"],
            duration=5,
            max_frames=2,
            target_width=128,
            include_filename=False,
            expose_images=False,
        )

        assert result is processor.client
        processor.record.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_add_visual_data_delegates_to_add_images(self, processor):
        """add_visual_data should proxy to add_images."""
        processor.add_images = AsyncMock()

        result = await processor.add_visual_data(
            image_entities=["camera.front"],
            image_paths=["/tmp/still.jpg"],
            target_width=128,
            include_filename=False,
            expose_images=False,
        )

        assert result is processor.client
        processor.add_images.assert_awaited_once()
