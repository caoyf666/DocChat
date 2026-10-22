import asyncio
import json
import logging
import queue
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import Request


class SSEEvent:
    PROGRESS = "progress"
    DELTA = "delta"
    FINAL = "final"


_task_stream: Dict[str, queue.Queue] = {}


def get_sse_queue(task_id: str) -> Optional[queue.Queue]:
    return _task_stream.get(task_id)


def create_sse_queue(task_id: str) -> queue.Queue:
    q = queue.Queue()
    _task_stream[task_id] = q
    return q


def remove_sse_queue(task_id: str):
    _task_stream.pop(task_id, None)


def _sse_pack(event: str, data: Dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def push_sse_event(task_id: str, event: str, data: Dict[str, Any]):
    stream_queue = get_sse_queue(task_id)
    if stream_queue:
        stream_queue.put({"event": event, "data": data})


async def sse_generator(task_id: str, request: Request) -> AsyncGenerator[str, None]:
    sse_queue = _task_stream.get(task_id)
    if sse_queue is None:
        return

    loop = asyncio.get_event_loop()

    try:
        while True:
            if await request.is_disconnected():
                return
            try:
                msg = await loop.run_in_executor(None, sse_queue.get, True, 1)
                yield _sse_pack(msg.get("event"), msg.get("data"))
            except queue.Empty:
                logging.info("SSE queue is empty, waiting...")
                continue
    except (ConnectionResetError, BrokenPipeError):
        return
    except asyncio.CancelledError:
        raise
    finally:
        remove_sse_queue(task_id)
