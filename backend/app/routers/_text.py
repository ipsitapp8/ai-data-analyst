"""A request-level guard for text the database cannot hold."""
from __future__ import annotations

from fastapi import HTTPException, Request


async def refuse_nul(request: Request) -> None:
    """Reject a JSON body that contains a NUL character, raw or escaped (\\u0000), with a 422.

    PostgreSQL cannot store NUL in a text column or in JSONB, so a name such as "a\\x00b" made the driver raise and the API answer 500, while SQLite
    stored it without complaint. Doing it once per router covers every name, note and key in the body, including keys inside nested JSON."""
    if request.method in ("POST", "PUT", "PATCH"):
        body = await request.body()
        if b"\x00" in body or b"\\u0000" in body.lower():
            raise HTTPException(422, "Text cannot contain the NUL character (\\u0000).")
