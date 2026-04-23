"""Tests for the framed JSON wire protocol used by ``thorn-toolhost``.

Covers:

* Round-tripping every frame type through ``encode_frame`` /
  ``decode_frame`` to lock the on-wire shape.
* Stream I/O: ``read_frame`` / ``write_frame`` against in-memory
  ``asyncio`` stream pairs, including back-to-back writes and
  partial reads.
* Defensive parsing: unknown kinds, missing required fields,
  oversized frames, malformed length prefixes.
"""

from __future__ import annotations

import asyncio
import base64
import json
import struct

import pytest

from thorn.toolhost._protocol import (
    LENGTH_PREFIX_FORMAT,
    LENGTH_PREFIX_SIZE,
    MAX_FRAME_SIZE,
    PROTOCOL_MAJOR,
    PROTOCOL_MINOR,
    FrameKind,
    Heartbeat,
    Hello,
    ProtocolError,
    ToolCallCancel,
    ToolCallChunk,
    ToolCallError,
    ToolCallRequest,
    ToolCallResponse,
    decode_frame,
    encode_frame,
    read_frame,
    write_frame,
)


def _strip_prefix(blob: bytes) -> bytes:
    (length,) = struct.unpack(LENGTH_PREFIX_FORMAT, blob[:LENGTH_PREFIX_SIZE])
    body = blob[LENGTH_PREFIX_SIZE:]
    assert len(body) == length
    return body


def _decode_body(blob: bytes) -> dict:
    return json.loads(_strip_prefix(blob).decode("utf-8"))


class TestFrameRoundTrip:
    def test_hello_round_trip(self):
        frame = Hello(
            protocol_major=PROTOCOL_MAJOR,
            protocol_minor=PROTOCOL_MINOR,
            thorn_version="1.2.3",
            features=["streaming", "cancel"],
            per_agent_state={"home": "/var/agents/x/home", "agent_id": "x"},
        )
        body = _decode_body(encode_frame(frame))
        assert body["kind"] == FrameKind.HELLO.value
        assert body["features"] == ["streaming", "cancel"]
        assert body["per_agent_state"]["agent_id"] == "x"

        round_tripped = decode_frame(_strip_prefix(encode_frame(frame)))
        assert round_tripped == frame

    def test_tool_call_request_round_trip(self):
        frame = ToolCallRequest(
            call_id="c1",
            tool_name="read_file",
            arguments={"path": "/etc/hosts"},
            per_call_context={"session_key": "s1"},
        )
        round_tripped = decode_frame(_strip_prefix(encode_frame(frame)))
        assert round_tripped == frame

    def test_tool_call_response_result_round_trip(self):
        frame = ToolCallResponse(call_id="c1", result="hello world")
        body = _decode_body(encode_frame(frame))
        assert body["result"] == "hello world"
        assert "error" not in body

        round_tripped = decode_frame(_strip_prefix(encode_frame(frame)))
        assert round_tripped == frame

    def test_tool_call_response_error_round_trip(self):
        err = ToolCallError(kind="timeout", message="took too long")
        frame = ToolCallResponse(call_id="c1", error=err)
        body = _decode_body(encode_frame(frame))
        assert body["error"] == {"kind": "timeout", "message": "took too long"}
        assert "result" not in body

        round_tripped = decode_frame(_strip_prefix(encode_frame(frame)))
        assert round_tripped == frame

    def test_tool_call_response_requires_exactly_one_payload(self):
        with pytest.raises(ProtocolError):
            ToolCallResponse(call_id="c1")  # neither result nor error
        with pytest.raises(ProtocolError):
            ToolCallResponse(
                call_id="c1",
                result="x",
                error=ToolCallError(kind="boom", message="m"),
            )

    def test_tool_call_chunk_round_trip(self):
        frame = ToolCallChunk(call_id="c1", stream="stdout", data="line\n")
        round_tripped = decode_frame(_strip_prefix(encode_frame(frame)))
        assert round_tripped == frame

    def test_tool_call_chunk_from_bytes_uses_base64(self):
        raw = b"\x00\x01\x02ascii\xff"
        frame = ToolCallChunk.from_bytes(call_id="c1", stream="structured", data=raw)
        assert frame.data == base64.b64encode(raw).decode("ascii")
        round_tripped = decode_frame(_strip_prefix(encode_frame(frame)))
        assert round_tripped == frame

    def test_tool_call_cancel_round_trip(self):
        frame = ToolCallCancel(call_id="c1")
        round_tripped = decode_frame(_strip_prefix(encode_frame(frame)))
        assert round_tripped == frame

    def test_heartbeat_round_trip(self):
        frame = Heartbeat()
        body = _decode_body(encode_frame(frame))
        assert body == {"kind": FrameKind.HEARTBEAT.value}
        round_tripped = decode_frame(_strip_prefix(encode_frame(frame)))
        assert round_tripped == frame


class TestDecodeFailures:
    def test_unknown_kind_raises(self):
        body = json.dumps({"kind": "definitely_unknown"}).encode("utf-8")
        with pytest.raises(ProtocolError, match="unknown frame kind"):
            decode_frame(body)

    def test_missing_kind_raises(self):
        body = json.dumps({"call_id": "c1"}).encode("utf-8")
        with pytest.raises(ProtocolError, match="missing a string 'kind'"):
            decode_frame(body)

    def test_invalid_json_raises(self):
        with pytest.raises(ProtocolError, match="not valid JSON"):
            decode_frame(b"{not json")

    def test_invalid_utf8_raises(self):
        with pytest.raises(ProtocolError, match="not valid UTF-8"):
            decode_frame(b"\xff\xfe\xff\xfe")

    def test_hello_missing_field_raises(self):
        body = json.dumps(
            {
                "kind": FrameKind.HELLO.value,
                "protocol_major": 1,
                "protocol_minor": 0,
                # thorn_version intentionally absent
            }
        ).encode("utf-8")
        with pytest.raises(ProtocolError, match="missing field 'thorn_version'"):
            decode_frame(body)

    def test_tool_call_response_without_result_or_error_raises(self):
        body = json.dumps(
            {"kind": FrameKind.TOOL_CALL_RESPONSE.value, "call_id": "c1"}
        ).encode("utf-8")
        with pytest.raises(ProtocolError, match="must carry either 'result' or 'error'"):
            decode_frame(body)

    def test_tool_call_response_error_must_be_object(self):
        body = json.dumps(
            {
                "kind": FrameKind.TOOL_CALL_RESPONSE.value,
                "call_id": "c1",
                "error": "boom",
            }
        ).encode("utf-8")
        with pytest.raises(ProtocolError, match="error' must be an object"):
            decode_frame(body)

    def test_unknown_extra_fields_are_tolerated(self):
        body = json.dumps(
            {
                "kind": FrameKind.HEARTBEAT.value,
                "extra": "future_field",
                "another": [1, 2, 3],
            }
        ).encode("utf-8")
        assert decode_frame(body) == Heartbeat()


class TestEncodeBoundaries:
    def test_oversized_frame_raises(self, monkeypatch):
        monkeypatch.setattr("thorn.toolhost._protocol.MAX_FRAME_SIZE", 16)
        big = ToolCallRequest(
            call_id="c1",
            tool_name="grep",
            arguments={"pattern": "x" * 64},
        )
        with pytest.raises(ProtocolError, match="exceeds"):
            encode_frame(big)

    def test_unknown_object_raises(self):
        with pytest.raises(ProtocolError, match="not a known protocol frame"):
            encode_frame(object())  # type: ignore[arg-type]


class TestReadWriteFrame:
    @pytest.mark.asyncio
    async def test_write_then_read_round_trip(self):
        reader, writer = await _stream_pair()
        frame = ToolCallRequest(call_id="c1", tool_name="echo", arguments={"v": 1})
        await write_frame(writer, frame)
        result = await read_frame(reader)
        assert result == frame
        writer.close()

    @pytest.mark.asyncio
    async def test_back_to_back_frames_preserve_order(self):
        reader, writer = await _stream_pair()
        frames = [
            Hello(
                protocol_major=PROTOCOL_MAJOR,
                protocol_minor=PROTOCOL_MINOR,
                thorn_version="x",
            ),
            ToolCallRequest(call_id="c1", tool_name="a"),
            Heartbeat(),
            ToolCallResponse(call_id="c1", result="ok"),
        ]
        for f in frames:
            await write_frame(writer, f)

        decoded = [await read_frame(reader) for _ in frames]
        assert decoded == frames
        writer.close()

    @pytest.mark.asyncio
    async def test_read_propagates_incomplete_read_on_clean_close(self):
        reader, writer = await _stream_pair()
        writer.close()
        with pytest.raises(asyncio.IncompleteReadError):
            await read_frame(reader)

    @pytest.mark.asyncio
    async def test_oversized_length_prefix_rejected(self):
        reader, writer = await _stream_pair()
        prefix = struct.pack(LENGTH_PREFIX_FORMAT, MAX_FRAME_SIZE + 1)
        writer.write(prefix)
        await writer.drain()
        with pytest.raises(ProtocolError, match="exceeds maximum"):
            await read_frame(reader)
        writer.close()

    @pytest.mark.asyncio
    async def test_zero_length_frame_rejected(self):
        reader, writer = await _stream_pair()
        writer.write(struct.pack(LENGTH_PREFIX_FORMAT, 0))
        await writer.drain()
        with pytest.raises(ProtocolError, match="zero-length"):
            await read_frame(reader)
        writer.close()

    @pytest.mark.asyncio
    async def test_truncated_body_raises_incomplete_read(self):
        reader, writer = await _stream_pair()
        body = b'{"kind":"heartbeat"}'
        writer.write(struct.pack(LENGTH_PREFIX_FORMAT, len(body) + 5))
        writer.write(body)
        await writer.drain()
        writer.close()
        with pytest.raises(asyncio.IncompleteReadError):
            await read_frame(reader)


async def _stream_pair() -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Create an in-memory ``(reader, writer)`` pair for protocol tests.

    Uses :func:`os.pipe` wrapped in asyncio's connector helpers so that
    the protocol layer sees the same ``StreamReader`` /
    ``StreamWriter`` shape it would see across a Unix-domain socket.
    """
    import os

    rfd, wfd = os.pipe()
    loop = asyncio.get_event_loop()

    reader = asyncio.StreamReader()
    read_protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: read_protocol, os.fdopen(rfd, "rb", 0))

    write_transport, write_protocol = await loop.connect_write_pipe(
        asyncio.streams.FlowControlMixin, os.fdopen(wfd, "wb", 0)
    )
    writer = asyncio.StreamWriter(
        write_transport, write_protocol, None, loop,
    )
    return reader, writer
