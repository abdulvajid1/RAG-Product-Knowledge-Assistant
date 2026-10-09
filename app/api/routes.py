import os
import json
import logging
from typing import AsyncIterator
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse, FileResponse

from app.config import get_settings
from app.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    ComponentHealth,
    FilterOptionsResponse,
    SourceItem,
)
from app.retrieval.retriever import get_retriever
from app.generation.prompt import PromptBuilder
from app.generation.llm import get_llm_client, LLMError
from app.observability import trace_component

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check reporting vector store, embedder, and LLM configuration."""
    settings = get_settings()

    # Vector store check (verify directory presence and initialized collection)
    vector_path = settings.vector_store_path
    vector_exists = os.path.exists(vector_path)
    vector_status = "ready" if vector_exists else "not_initialized"

    # Embedder check (reports configured model name)
    embedder_status = "configured"

    # LLM configuration check (does not call paid APIs)
    llm_status = "configured" if settings.llm_provider else "unconfigured"

    components = {
        "vector_store": ComponentHealth(
            status=vector_status,
            details={"path": vector_path, "initialized": vector_exists},
        ),
        "embedder": ComponentHealth(
            status=embedder_status,
            details={"model": settings.embedding_model},
        ),
        "llm": ComponentHealth(
            status=llm_status,
            details={"provider": settings.llm_provider, "model": settings.llm_model},
        ),
    }

    overall_status = "healthy" if llm_status != "unconfigured" else "degraded"

    return HealthResponse(
        status=overall_status,
        components=components,
    )


@router.get("/filters", response_model=FilterOptionsResponse)
async def get_filter_options():
    """Return available metadata filters to populate UI dropdowns."""
    return FilterOptionsResponse(
        categories=["Solar Panels", "Safety Shoes", "Industrial Sensors"],
        brands=["SolarMax", "SunPower", "HelioCell", "TerraGrip", "TitanSteel", "SafeStep", "SensorTech", "OptiFlow", "VibraSense"],
        suppliers=[
            "Vikram Solar Energies Pvt Ltd",
            "GreenWatt Technologies",
            "Aditya Power Systems",
            "Acme Industrial Safety",
            "Rheinland Workwear GmbH",
            "Bharat Footwear Mills",
            "SensorTech Automation Ltd",
            "FlowMetrics Europe",
            "InduSensors India Pvt Ltd",
        ],
        countries=["India", "Germany", "USA", "Switzerland"],
        products=[
            "SolarMax 550",
            "SolarMax 600",
            "SunPower Eco 400",
            "HelioCell 750 Bifacial",
            "TerraGrip Pro WorkBoot",
            "TerraGrip Ultra Lite",
            "TitanSteel HeavyDuty 900",
            "SafeStep Eco Runner",
            "SensorTech PT100 Transmitter",
            "OptiFlow Ultrasonic Meter",
            "VibraSense Wireless IMU",
        ],
    )


@router.post("/ask", response_model=AskResponse)
@trace_component(name="Endpoint.ask", run_type="chain")
async def ask_question(request: AskRequest):
    """Non-streaming QA endpoint returning complete answer and source attributions."""
    settings = get_settings()
    status_422 = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", status.HTTP_422_UNPROCESSABLE_ENTITY)
    if len(request.query) > settings.max_query_chars:
        raise HTTPException(
            status_code=status_422,
            detail=f"Query exceeds maximum character limit of {settings.max_query_chars}",
        )

    retriever = get_retriever()
    filters = request.filters.model_dump(exclude_none=True) if request.filters else None

    try:
        retrieval_res = retriever.retrieve(
            query=request.query,
            filters=filters,
            top_k=request.top_k,
        )
    except Exception as e:
        logger.exception(f"Error during retrieval in /ask: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Retrieval service encountered an internal error",
        )

    # Spec 7.1: If no chunk passes threshold, return empty context / not available (do not call LLM with empty context)
    if not retrieval_res.has_context:
        return AskResponse(
            answer="The requested information is not available in the provided knowledge base.",
            sources=[],
        )

    prompt_builder = PromptBuilder()
    messages = prompt_builder.build_messages(
        query=request.query,
        context=retrieval_res.context,
    )

    llm = get_llm_client()
    try:
        answer = await llm.generate(messages)
    except LLMError as e:
        logger.error(f"LLM generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=e.message,
        )
    except Exception as e:
        logger.exception(f"Unexpected generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Language model generation failed",
        )

    return AskResponse(
        answer=answer.strip(),
        sources=retrieval_res.sources,
    )


@router.post("/ask/stream")
async def ask_stream(request: Request, ask_req: AskRequest):
    """SSE streaming endpoint for progressive answer rendering."""
    settings = get_settings()
    status_422 = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", status.HTTP_422_UNPROCESSABLE_ENTITY)
    if len(ask_req.query) > settings.max_query_chars:
        raise HTTPException(
            status_code=status_422,
            detail=f"Query exceeds maximum character limit of {settings.max_query_chars}",
        )

    retriever = get_retriever()
    filters = ask_req.filters.model_dump(exclude_none=True) if ask_req.filters else None

    @trace_component(name="Endpoint.ask_stream", run_type="chain")
    async def event_generator() -> AsyncIterator[str]:
        try:
            # 1. Retrieval
            retrieval_res = retriever.retrieve(
                query=ask_req.query,
                filters=filters,
                top_k=ask_req.top_k,
            )

            # Check if client disconnected during retrieval
            if await request.is_disconnected():
                logger.info("Client disconnected during retrieval. Aborting stream.")
                return

            # Spec 7.1 / 9.3: No-context case
            if not retrieval_res.has_context:
                not_found_token = {
                    "type": "token",
                    "content": "The requested information is not available in the provided knowledge base.",
                }
                yield f"data: {json.dumps(not_found_token)}\n\n"
                sources_event = {"type": "sources", "sources": []}
                yield f"data: {json.dumps(sources_event)}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

            # 2. Build prompt
            prompt_builder = PromptBuilder()
            messages = prompt_builder.build_messages(
                query=ask_req.query,
                context=retrieval_res.context,
            )

            # 3. Stream LLM tokens
            llm = get_llm_client()
            async for token in llm.stream(messages):
                if await request.is_disconnected():
                    logger.info("Client disconnected during streaming. Aborting.")
                    return
                token_event = {"type": "token", "content": token}
                yield f"data: {json.dumps(token_event)}\n\n"

            # 4. Stream sources event per Spec 9.3
            sources_payload = [s.model_dump() for s in retrieval_res.sources]
            sources_event = {"type": "sources", "sources": sources_payload}
            yield f"data: {json.dumps(sources_event)}\n\n"

            # 5. Done event
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except LLMError as e:
            logger.warning(f"LLM streaming error: {e}")
            err_event = {
                "type": "error",
                "message": e.message,
                "code": e.code,
            }
            yield f"data: {json.dumps(err_event)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            logger.exception(f"Unhandled error in streaming endpoint: {e}")
            err_event = {
                "type": "error",
                "message": "An unexpected error occurred while streaming the response.",
                "code": "internal_error",
            }
            yield f"data: {json.dumps(err_event)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
