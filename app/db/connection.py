"""
PostgreSQL connection helpers.

The DB writer keeps a small SQLAlchemy async connection pool alive while the
application runs. Each batch insert creates a short-lived AsyncSession, commits
the work, and closes the session. Closing the session returns its connection to
the pool; the real PostgreSQL connections are closed when the engine is
disposed on application shutdown.
"""

import logging
import os
from pathlib import Path

try:
    import asyncpg
except ImportError:
    asyncpg = None

try:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from sqlalchemy.ext.asyncio import create_async_engine
except ImportError:
    text = None
    async_sessionmaker = None
    create_async_engine = None


SCHEMA_PATH = Path(__file__).with_name("schema.sql")
PROJECT_ROOT = SCHEMA_PATH.parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
POOL_SIZE = 4

engine = None
SessionLocal = None
logger = logging.getLogger(__name__)


def load_env_file():
    """
    Load simple KEY=value pairs from the project .env file.

    Existing environment variables win, so shell-provided values can override
    local defaults.
    """

    if not ENV_PATH.exists():
        return

    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(
            key.strip(),
            value.strip().strip('"').strip("'"),
        )


load_env_file()

DATABASE_URL = os.getenv("DATABASE_URL")


def build_sqlalchemy_url(database_url):
    """
    Convert a normal PostgreSQL URL into SQLAlchemy's asyncpg URL format.

    The .env file can stay readable as:

        postgresql://user:password@host:port/dbname

    SQLAlchemy's async engine needs:

        postgresql+asyncpg://user:password@host:port/dbname
    """

    if not database_url:
        return None

    if database_url.startswith("postgresql+asyncpg://"):
        return database_url

    if database_url.startswith("postgresql://"):
        return database_url.replace(
            "postgresql://",
            "postgresql+asyncpg://",
            1,
        )

    return database_url


def validate_configuration():
    """
    Check whether PostgreSQL writes can be enabled.
    """

    if asyncpg is None:
        logger.warning("asyncpg_missing db_writer_disabled=true")
        return False

    if create_async_engine is None or async_sessionmaker is None or text is None:
        logger.warning("sqlalchemy_missing db_writer_disabled=true")
        return False

    if not DATABASE_URL:
        logger.warning("database_url_missing db_writer_disabled=true")
        return False

    return True


def is_configured():
    """
    Return whether PostgreSQL writes can be attempted.
    """

    return validate_configuration()


async def init_db_engine():
    """
    Create the async SQLAlchemy engine and initialize the DB schema.

    The engine owns the connection pool. With pool_size=4 and max_overflow=0,
    this application can keep up to four PostgreSQL connections open and will
    not create extra overflow connections under load.
    """

    global engine
    global SessionLocal

    if not is_configured():
        return False

    if engine is not None and SessionLocal is not None:
        return True

    engine = create_async_engine(
        build_sqlalchemy_url(DATABASE_URL),
        pool_size=POOL_SIZE,
        max_overflow=0,
        pool_pre_ping=True,
    )

    SessionLocal = async_sessionmaker(
        bind=engine,
        expire_on_commit=False,
    )

    await initialize_schema()
    return True


def get_session():
    """
    Create a short-lived AsyncSession for one database operation.
    """

    if SessionLocal is None:
        raise RuntimeError("Database engine has not been initialized.")

    return SessionLocal()


async def initialize_schema():
    """
    Create the detections table and indexes using the engine pool.
    """

    if engine is None:
        raise RuntimeError("Database engine has not been initialized.")

    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")

    async with engine.begin() as db:
        for statement in schema_sql.split(";"):
            statement = statement.strip()

            if statement:
                await db.execute(text(statement))


async def dispose_db_engine():
    """
    Close all pooled PostgreSQL connections.
    """

    global engine
    global SessionLocal

    if engine is not None:
        await engine.dispose()

    engine = None
    SessionLocal = None
