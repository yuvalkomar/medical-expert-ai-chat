import uvicorn

from backend.app.config import get_settings


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "backend.app.main:app",
        host=settings.server_host,
        port=settings.server_port,
    )
