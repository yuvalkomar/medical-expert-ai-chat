from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import Engine, event
from sqlalchemy.engine import make_url
from sqlmodel import Session, SQLModel, create_engine


class Database:
    """Owns one shared engine and creates short-lived independent sessions."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self._ensure_sqlite_directory()
        connect_args: dict[str, object] = {}
        if database_url.startswith("sqlite"):
            connect_args = {"check_same_thread": False, "timeout": 30}
        self.engine = create_engine(
            database_url,
            connect_args=connect_args,
            pool_pre_ping=True,
        )
        if database_url.startswith("sqlite"):
            self._configure_sqlite(self.engine)

    def _ensure_sqlite_directory(self) -> None:
        url = make_url(self.database_url)
        if url.get_backend_name() != "sqlite" or not url.database:
            return
        if url.database == ":memory:" or url.database.startswith("file:"):
            return
        Path(url.database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _configure_sqlite(engine: Engine) -> None:
        @event.listens_for(engine, "connect")
        def set_sqlite_pragmas(dbapi_connection: object, _: object) -> None:
            cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

    def create_schema(self) -> None:
        SQLModel.metadata.create_all(self.engine)

    def session(self) -> Session:
        return Session(self.engine, expire_on_commit=False)

    def dispose(self) -> None:
        self.engine.dispose()


def session_scope(database: Database) -> Iterator[Session]:
    """Small generator useful as a FastAPI dependency."""
    with database.session() as session:
        yield session

