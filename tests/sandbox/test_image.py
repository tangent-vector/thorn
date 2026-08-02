"""Tests for :mod:`thorn.sandbox._image`."""

from __future__ import annotations

from pathlib import Path

import pytest

from thorn.sandbox import (
    DEFAULT_SANDBOX_DOCKERFILE,
    FakeOCIRuntimeAdapter,
    SandboxImageMissingError,
    build_default_sandbox_image,
    default_sandbox_image_tag,
    ensure_sandbox_image,
    find_default_sandbox_dockerfile,
)


class TestDefaultTag:
    def test_default_tag_starts_with_thorn_sandbox(self) -> None:
        tag = default_sandbox_image_tag()
        assert tag.startswith("thorn-sandbox:")
        assert ":" in tag


class TestEnsureImage:
    @pytest.mark.asyncio
    async def test_present_returns_silently(self) -> None:
        adapter = FakeOCIRuntimeAdapter(present_images=["t:1"])
        await ensure_sandbox_image(adapter, "t:1")  # no raise

    @pytest.mark.asyncio
    async def test_missing_raises_with_remediation(self) -> None:
        adapter = FakeOCIRuntimeAdapter()
        with pytest.raises(SandboxImageMissingError) as exc_info:
            await ensure_sandbox_image(adapter, "ghost:1")
        msg = str(exc_info.value)
        assert "ghost:1" in msg
        assert "thorn sandbox build" in msg


class TestFindDockerfile:
    def test_finds_in_repo(self) -> None:
        path = find_default_sandbox_dockerfile()
        assert path.is_file()
        assert path.name == DEFAULT_SANDBOX_DOCKERFILE

    def test_raises_when_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            find_default_sandbox_dockerfile(start=tmp_path)


class TestBuild:
    @pytest.mark.asyncio
    async def test_builds_with_defaults(self) -> None:
        adapter = FakeOCIRuntimeAdapter()
        tag = await build_default_sandbox_image(adapter)
        assert tag == default_sandbox_image_tag()
        assert len(adapter.build_calls) == 1
        ctx, dockerfile, used_tag = adapter.build_calls[0]
        assert dockerfile.name == DEFAULT_SANDBOX_DOCKERFILE
        assert used_tag == tag
        # The bundled Dockerfile.sandbox does ``COPY pyproject.toml ./``
        # so the resolved build context must be the source-tree root,
        # not the dockerfile's parent (which is now the wheel-shipped
        # ``_resources/`` directory).
        assert (ctx / "pyproject.toml").is_file(), (
            f"resolved build context {ctx} does not contain pyproject.toml"
        )
        # The image should now be in the cache via the fake adapter.
        assert await adapter.image_exists(tag)

    @pytest.mark.asyncio
    async def test_explicit_tag_and_paths(self, tmp_path: Path) -> None:
        adapter = FakeOCIRuntimeAdapter()
        dockerfile = tmp_path / "Dockerfile.custom"
        dockerfile.write_text("FROM scratch\n")
        ctx = tmp_path
        tag = await build_default_sandbox_image(
            adapter, tag="custom:1", dockerfile=dockerfile, context=ctx,
        )
        assert tag == "custom:1"
        assert adapter.build_calls == [(ctx, dockerfile, "custom:1")]
