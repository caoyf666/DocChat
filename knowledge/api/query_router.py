import asyncio
import os
from typing import Union

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from knowledge.core.deps import get_query_service
from knowledge.core.paths import get_front_page_dir
from knowledge.schema.query_schema import QueryRequest, QueryResponse, StreamSubmitResponse
from knowledge.schema.upload_schema import TaskStatusResponse
from knowledge.service.query_service import QueryService
from knowledge.utils.sse_util import create_sse_queue, sse_generator
from knowledge.utils.task_util import get_task_info


def create_app():
    app = FastAPI(description="掌柜智库查询的应用", version="v1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    page_dir = get_front_page_dir()
    if page_dir and os.path.exists(page_dir):
        app.mount("/front", StaticFiles(directory=page_dir))

    register_router(app)
    return app


def register_router(app: FastAPI):
    @app.get("/")
    def hello_world():
        return {"flag": "success"}

    @app.post("/query")
    async def query(
        request: QueryRequest,
        background_tasks: BackgroundTasks,
        service: QueryService = Depends(get_query_service),
    ) -> Union[StreamSubmitResponse, QueryResponse]:
        session_id = request.session_id or service.generate_session_id()
        task_id = service.generate_task_id()

        if request.is_stream:
            create_sse_queue(task_id=task_id)
            background_tasks.add_task(
                service.run_query_graph,
                session_id=session_id,
                task_id=task_id,
                query=request.query,
                is_stream=request.is_stream,
            )
            return StreamSubmitResponse(message="查询请求已经提交", session_id=session_id, task_id=task_id)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, service.run_query_graph, session_id, task_id, request.query, request.is_stream)
        answer = service.get_task_result(task_id)
        return QueryResponse(message="查询请求已经处理完了", session_id=session_id, answer=answer)

    @app.get("/stream/{task_id}")
    async def stream(task_id: str, request: Request) -> StreamingResponse:
        return StreamingResponse(content=sse_generator(task_id, request), media_type="text/event-stream")

    @app.get("/status/{task_id}", response_model=TaskStatusResponse)
    async def get_task_status(task_id: str):
        return TaskStatusResponse(**get_task_info(task_id))

    @app.get("/history/{session_id}")
    async def get_history(
        session_id: str,
        limit: int = 50,
        service: QueryService = Depends(get_query_service),
    ):
        try:
            items = service.get_history(session_id, limit)
            return {"session_id": session_id, "items": items}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"history error: {e}")

    @app.delete("/history/{session_id}")
    async def clear_chat_history(
        session_id: str,
        service: QueryService = Depends(get_query_service),
    ):
        count = service.clear_history(session_id)
        return {"message": "History cleared", "deleted_count": count}


if __name__ == "__main__":
    uvicorn.run(create_app(), host="0.0.0.0", port=8001)
