import os

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

_pool: AsyncConnectionPool | None = None
_checkpointer: AsyncPostgresSaver | None = None


async def init_db():
    global _pool, _checkpointer

    _pool = AsyncConnectionPool(
        conninfo=os.getenv("DATABASE_URL"),
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    await _pool.open()

    _checkpointer = AsyncPostgresSaver(_pool)
    await _checkpointer.setup()

    async with _pool.connection() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS blog_runs (
                run_id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                language TEXT,
                status TEXT NOT NULL,
                blog_title TEXT,
                blog_content TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)


async def close_db():
    if _pool is not None:
        await _pool.close()


def get_checkpointer() -> AsyncPostgresSaver:
    return _checkpointer


async def create_run(run_id: str, topic: str, language: str | None):
    async with _pool.connection() as conn:
        await conn.execute(
            "INSERT INTO blog_runs (run_id, topic, language, status) VALUES (%s, %s, %s, %s)",
            (run_id, topic, language, "pending"),
        )


async def update_run_status(run_id: str, status: str, blog_title: str | None = None, blog_content: str | None = None):
    async with _pool.connection() as conn:
        await conn.execute(
            """UPDATE blog_runs
               SET status = %s,
                   blog_title = COALESCE(%s, blog_title),
                   blog_content = COALESCE(%s, blog_content),
                   updated_at = now()
               WHERE run_id = %s""",
            (status, blog_title, blog_content, run_id),
        )


async def get_run(run_id: str):
    async with _pool.connection() as conn:
        cur = await conn.execute("SELECT * FROM blog_runs WHERE run_id = %s", (run_id,))
        return await cur.fetchone()
