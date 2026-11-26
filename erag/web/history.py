"""Durable browser-scoped conversations with transactional cancellation.

The browser token is a local-user capability, not a replacement for authentication
on a public deployment. Keep the default loopback binding or add authentication.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path


def identifier(value):
    try:
        return uuid.UUID(str(value)).hex
    except (ValueError, TypeError, AttributeError):
        raise ValueError("无效的会话标识，请刷新页面") from None


class ChatStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, title TEXT NOT NULL,
                    kb TEXT, rerank INTEGER NOT NULL, tct INTEGER NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS conversations_owner ON conversations(owner, updated_at);
                CREATE TABLE IF NOT EXISTS turns (
                    id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    question TEXT NOT NULL, answer TEXT, sources TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS turns_conversation ON turns(conversation_id, created_at);
            """)
            if "deleted_at" not in {row[1] for row in db.execute("PRAGMA table_info(conversations)")}:
                db.execute("ALTER TABLE conversations ADD COLUMN deleted_at TEXT")

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            with db:
                yield db
        finally:
            db.close()

    def browser_secret(self):
        path = self.path.with_suffix(".browser-key")
        try:
            with path.open("x", encoding="utf-8") as stream:
                path.chmod(0o600)
                stream.write(secrets.token_hex(32))
        except FileExistsError:
            pass
        return path.read_text()

    def create(self, owner, *, kb=None, rerank=True, tct=False):
        owner, session = identifier(owner), uuid.uuid4().hex
        now = datetime.now(UTC).isoformat()
        with self.connection() as db:
            db.execute(
                "INSERT INTO conversations (id,owner,title,kb,rerank,tct,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?)",
                (session, owner, "新对话", kb, int(rerank), int(tct), now, now),
            )
        return session

    def list(self, owner, search=""):
        with self.connection() as db:
            rows = db.execute(
                "SELECT * FROM conversations WHERE owner=? AND deleted_at IS NULL ORDER BY updated_at DESC, id",
                (identifier(owner),),
            ).fetchall()
        return [dict(row) for row in rows if search.casefold() in row["title"].casefold()]

    def _get(self, db, owner, session):
        row = db.execute(
            "SELECT * FROM conversations WHERE id=? AND owner=? AND deleted_at IS NULL",
            (identifier(session), identifier(owner)),
        ).fetchone()
        if row is None:
            raise ValueError("对话不存在或已被删除")
        return dict(row)

    def get(self, owner, session):
        with self.connection() as db:
            result = self._get(db, owner, session)
            result["turns"] = [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM turns WHERE conversation_id=? ORDER BY created_at,id", (session,)
                )
            ]
        for turn in result["turns"]:
            turn["sources"] = json.loads(turn["sources"])
        return result

    def configure(self, owner, session, *, kb, rerank, tct):
        with self.connection() as db:
            self._get(db, owner, session)
            db.execute(
                "UPDATE conversations SET kb=?,rerank=?,tct=? WHERE id=?",
                (kb, int(rerank), int(tct), session),
            )

    def model_history(self, owner, session):
        return [
            message
            for turn in self.get(owner, session)["turns"]
            if turn["status"] == "complete"
            for message in [
                {"role": "user", "content": turn["question"]},
                {"role": "assistant", "content": turn["answer"]},
            ]
        ]

    def rename(self, owner, session, title):
        title = title.strip()[:80]
        if not title:
            raise ValueError("对话名称不能为空")
        with self.connection() as db:
            self._get(db, owner, session)
            db.execute("UPDATE conversations SET title=? WHERE id=?", (title, session))

    def delete(self, owner, session):
        with self.connection() as db:
            self._get(db, owner, session)
            db.execute(
                "UPDATE turns SET status='stopped' WHERE conversation_id=? AND status='running'", (session,)
            )
            db.execute(
                "UPDATE conversations SET deleted_at=? WHERE id=?", (datetime.now(UTC).isoformat(), session)
            )

    def undo_delete(self, owner):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT id FROM conversations WHERE owner=? AND deleted_at IS NOT NULL ORDER BY deleted_at DESC LIMIT 1",
                (identifier(owner),),
            ).fetchone()
            if row is None:
                return None
            db.execute("UPDATE conversations SET deleted_at=NULL WHERE id=?", (row["id"],))
            return row["id"]

    def begin(self, owner, session, question, *, retry=False):
        turn = uuid.uuid4().hex
        now = datetime.now(UTC).isoformat()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            current = self._get(db, owner, session)
            if db.execute(
                "SELECT 1 FROM turns WHERE conversation_id=? AND status='running'", (session,)
            ).fetchone():
                raise ValueError("当前对话正在生成，请等待或停止生成")
            if retry:
                last = db.execute(
                    "SELECT * FROM turns WHERE conversation_id=? ORDER BY created_at DESC,id DESC LIMIT 1",
                    (session,),
                ).fetchone()
                if last is None:
                    raise ValueError("还没有可以重新生成的问题")
                question = last["question"]
                db.execute("DELETE FROM turns WHERE id=?", (last["id"],))
            question = question.strip()
            if not question:
                raise ValueError("请输入问题")
            db.execute(
                "INSERT INTO turns (id,conversation_id,question,status,created_at) VALUES (?,?,?,?,?)",
                (turn, session, question, "running", now),
            )
            title = question[:28] if current["title"] == "新对话" else current["title"]
            db.execute("UPDATE conversations SET title=?,updated_at=? WHERE id=?", (title, now, session))
        return turn, question

    def finish(self, owner, session, turn, answer, sources, *, error=False):
        with self.connection() as db:
            # A late response may arrive after stop/delete. It must not resurrect the turn.
            try:
                self._get(db, owner, session)
            except ValueError:
                return False
            result = db.execute(
                "UPDATE turns SET answer=?,sources=?,status=? WHERE id=? AND conversation_id=? AND status='running'",
                (
                    answer,
                    json.dumps(sources, ensure_ascii=False),
                    "error" if error else "complete",
                    turn,
                    session,
                ),
            )
            return result.rowcount == 1

    def stop(self, owner, session):
        with self.connection() as db:
            self._get(db, owner, session)
            db.execute(
                "UPDATE turns SET status='stopped' WHERE conversation_id=? AND status='running'", (session,)
            )

    def recover_interrupted(self):
        """Called once at server startup, never on a browser refresh."""
        with self.connection() as db:
            db.execute("UPDATE turns SET status='stopped' WHERE status='running'")

    def export(self, owner, session):
        conversation = self.get(owner, session)
        lines = [f"# {conversation['title']}", "", f"知识库：{conversation['kb'] or '普通对话'}", ""]
        for turn in conversation["turns"]:
            lines.extend(
                ["## 用户", "", turn["question"], "", "## ERAG", "", turn["answer"] or "（已停止生成）", ""]
            )
            for i, source in enumerate(turn["sources"], 1):
                chunk = source["chunk"]
                lines.extend(
                    [
                        f"[{i}] {Path(chunk['source']).name} · {chunk.get('chapter', '')}",
                        "",
                        chunk["content"],
                        "",
                    ]
                )
        directory = self.path.parent / "exports"
        directory.mkdir(exist_ok=True)
        path = directory / f"{identifier(session)}.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        return str(path)
