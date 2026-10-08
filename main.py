import os
import uuid
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import get_settings, setup_logging
from app.api.routes import router as api_router

logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("Application starting up...")
    yield
    logger.info("Application shutting down...")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Filumart Product Knowledge Assistant",
        description="Grounded RAG Assistant for B2B Product Specifications",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global Exception Handlers per Spec Section 9.4
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        error_messages = []
        for error in exc.errors():
            loc = " -> ".join(str(l) for l in error.get("loc", []))
            msg = error.get("msg", "Invalid value")
            error_messages.append(f"{loc}: {msg}" if loc else msg)
        message = "; ".join(error_messages) if error_messages else "Invalid request payload"
        status_422 = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", status.HTTP_422_UNPROCESSABLE_ENTITY)
        return JSONResponse(
            status_code=status_422,
            content={"error": {"code": "invalid_input", "message": message}},
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        status_422 = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", status.HTTP_422_UNPROCESSABLE_ENTITY)
        code_map = {
            status.HTTP_400_BAD_REQUEST: "bad_request",
            status.HTTP_404_NOT_FOUND: "not_found",
            status_422: "invalid_input",
            status.HTTP_500_INTERNAL_SERVER_ERROR: "internal_error",
            status.HTTP_503_SERVICE_UNAVAILABLE: "service_unavailable",
        }
        code = code_map.get(exc.status_code, "http_error")
        message = str(exc.detail) if exc.detail else "An HTTP error occurred"
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": code, "message": message}},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        request_id = str(uuid.uuid4())
        logger.exception(f"Unhandled server error [request_id={request_id}]: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "An unexpected internal server error occurred. Please try again later.",
                }
            },
        )

    # Include API routes
    app.include_router(api_router)

    # Frontend serving
    frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
    if os.path.exists(frontend_dir):
        app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

        @app.get("/")
        async def serve_index():
            index_path = os.path.join(frontend_dir, "index.html")
            if os.path.exists(index_path):
                return FileResponse(index_path, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
            return {"message": "Filumart Product Knowledge Assistant API is running."}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
