from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
import sqlite3
import threading
import uuid
from collections import OrderedDict
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.agents.coordinator import analyze_question
from app.services.analytics import dashboard_summary, derive_measures, forecast_summary, generate_insights, load_frame, preprocess_frame, profile_frame

app = FastAPI(title="DataNova AI API", version="1.0.0")
logger = logging.getLogger("datanova.upload")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
@app.middleware("http")
async def no_store_api_responses(request, call_next):
    # API responses are per-user; never let a browser or proxy cache serve them to another session.
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
        response.headers["Vary"] = "Authorization"
    return response

app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
DATABASE_PATH = Path(__file__).parent.parent / "datanova.db"
SESSIONS: dict[str, int] = {}

def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection

def initialize_database() -> None:
    with get_connection() as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
        connection.execute("CREATE TABLE IF NOT EXISTS sessions (token TEXT PRIMARY KEY, user_id INTEGER NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(user_id) REFERENCES users(id))")
        connection.execute("CREATE TABLE IF NOT EXISTS datasets (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, name TEXT NOT NULL, stored_name TEXT UNIQUE NOT NULL, processed_name TEXT, file_size INTEGER NOT NULL, original_filename TEXT, file_path TEXT, processed_file_path TEXT, file_type TEXT, row_count INTEGER, column_count INTEGER, quality_score REAL, status TEXT DEFAULT 'processed', metadata_json TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP, updated_at TEXT DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(user_id) REFERENCES users(id))")
        columns = {row[1] for row in connection.execute("PRAGMA table_info(datasets)").fetchall()}
        migrations = {"processed_name": "TEXT", "original_filename": "TEXT", "file_path": "TEXT", "processed_file_path": "TEXT", "file_type": "TEXT", "row_count": "INTEGER", "column_count": "INTEGER", "quality_score": "REAL", "status": "TEXT DEFAULT 'processed'", "metadata_json": "TEXT", "updated_at": "TEXT"}
        for column, column_type in migrations.items():
            if column not in columns:
                connection.execute(f"ALTER TABLE datasets ADD COLUMN {column} {column_type}")
        connection.commit()

initialize_database()

class QueryRequest(BaseModel):
    question: str
    dataset_id: int | None = None
    dataset_name: str | None = None

class SignupRequest(BaseModel):
    name: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 120_000)
    return f"{salt.hex()}:{digest.hex()}"

def verify_password(password: str, stored: str) -> bool:
    salt_hex, digest_hex = stored.split(":", maxsplit=1)
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), 120_000)
    return hmac.compare_digest(candidate.hex(), digest_hex)

def public_user(row: sqlite3.Row) -> dict[str, str | int]:
    return {"id": row["id"], "name": row["name"], "email": row["email"], "created_at": row["created_at"]}

def authenticated_user(authorization: str | None) -> int:
    token = authorization.removeprefix("Bearer ").strip() if authorization else ""
    user_id = SESSIONS.get(token)
    if user_id is None and token:
        with get_connection() as connection:
            row = connection.execute("SELECT user_id FROM sessions WHERE token = ?", (token,)).fetchone()
        user_id = int(row["user_id"]) if row else None
    if user_id is None:
        raise HTTPException(status_code=401, detail="Sign in to access this workspace.")
    return user_id

READY_STATUSES = ("ready", "processed")
_FRAME_CACHE: OrderedDict[tuple[str, float], pd.DataFrame] = OrderedDict()
_FRAME_CACHE_LOCK = threading.Lock()

def dataset_path(row: sqlite3.Row) -> Path:
    return DATA_DIR / row["stored_name"]

def analysis_path(row: sqlite3.Row) -> Path:
    """Analytics run on the preprocessed CSV (fast to read); the original upload is kept untouched."""
    if row["processed_name"] and (DATA_DIR / row["processed_name"]).exists():
        return DATA_DIR / row["processed_name"]
    return dataset_path(row)

def load_analysis_frame(row: sqlite3.Row) -> pd.DataFrame:
    path = analysis_path(row)
    if not path.exists():
        raise HTTPException(status_code=410, detail="The stored file for this dataset is missing. Upload it again.")
    key = (str(path), path.stat().st_mtime)
    with _FRAME_CACHE_LOCK:
        if key in _FRAME_CACHE:
            _FRAME_CACHE.move_to_end(key)
            return _FRAME_CACHE[key]
    frame = derive_measures(load_frame(path))
    with _FRAME_CACHE_LOCK:
        _FRAME_CACHE[key] = frame
        while len(_FRAME_CACHE) > 4:
            _FRAME_CACHE.popitem(last=False)
    return frame

def dataset_profile(row: sqlite3.Row) -> dict:
    """Profile of the ORIGINAL upload, computed once at upload time and stored in metadata_json."""
    if row["metadata_json"]:
        return json.loads(row["metadata_json"])
    profile = profile_frame(load_frame(dataset_path(row)), int(row["file_size"]))
    with get_connection() as connection:
        connection.execute("UPDATE datasets SET metadata_json = ?, row_count = ?, column_count = ?, quality_score = ? WHERE id = ?", (json.dumps(profile, default=str), profile["rows"], profile["columns"], profile["quality_score"], row["id"]))
        connection.commit()
    return profile

def owned_dataset(dataset_id: int, user_id: int) -> sqlite3.Row:
    with get_connection() as connection:
        row = connection.execute(f"SELECT * FROM datasets WHERE id = ? AND user_id = ? AND status IN ({','.join('?' * len(READY_STATUSES))})", (dataset_id, user_id, *READY_STATUSES)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return row

def dataset_response(row: sqlite3.Row) -> dict:
    profile = dataset_profile(row)
    return {"id": row["id"], "dataset_id": str(row["id"]), "name": row["name"], "dataset_name": row["name"], "original_filename": row["original_filename"] or row["name"], "file_path": row["file_path"] or str(dataset_path(row)), "processed_file_path": row["processed_file_path"] or (str(DATA_DIR / row["processed_name"]) if row["processed_name"] else None), "file_type": row["file_type"] or Path(row["name"]).suffix.lower(), "status": "ready" if (row["status"] or "processed") in READY_STATUSES else row["status"], "created_at": row["created_at"], "updated_at": row["updated_at"] or row["created_at"], "processed_available": bool(row["processed_name"]), **profile}

@app.get("/")
def root() -> dict[str, str]:
    return {"name": "DataNova AI API", "status": "running", "version": "1.0.0"}

@app.get("/health")
def root_health() -> dict[str, str]:
    return {"status": "healthy"}

@app.post("/api/auth/signup")
def signup(request: SignupRequest) -> dict:
    if len(request.password) < 8:
        raise HTTPException(status_code=400, detail="Use a password with at least 8 characters.")
    try:
        with get_connection() as connection:
            cursor = connection.execute("INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)", (request.name.strip(), request.email.lower().strip(), hash_password(request.password)))
            row = connection.execute("SELECT * FROM users WHERE id = ?", (cursor.lastrowid,)).fetchone()
            connection.commit()
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="An account with this email already exists.") from exc
    token = secrets.token_urlsafe(32)
    SESSIONS[token] = int(row["id"])
    with get_connection() as connection:
        connection.execute("INSERT INTO sessions (token, user_id) VALUES (?, ?)", (token, row["id"]))
        connection.commit()
    return {"token": token, "user": public_user(row)}

@app.post("/api/auth/login")
def login(request: LoginRequest) -> dict:
    with get_connection() as connection:
        row = connection.execute("SELECT * FROM users WHERE email = ?", (request.email.lower().strip(),)).fetchone()
    if row is None or not verify_password(request.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Email or password is incorrect.")
    token = secrets.token_urlsafe(32)
    SESSIONS[token] = int(row["id"])
    with get_connection() as connection:
        connection.execute("INSERT INTO sessions (token, user_id) VALUES (?, ?)", (token, row["id"]))
        connection.commit()
    return {"token": token, "user": public_user(row)}

@app.get("/api/auth/me")
def current_user(token: str) -> dict:
    user_id = SESSIONS.get(token)
    if user_id is None:
        with get_connection() as connection:
            row = connection.execute("SELECT user_id FROM sessions WHERE token = ?", (token,)).fetchone()
        user_id = int(row["user_id"]) if row else None
    if user_id is None:
        raise HTTPException(status_code=401, detail="Your session has expired.")
    with get_connection() as connection:
        row = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=401, detail="Your account could not be found.")
    return {"user": public_user(row)}

@app.post("/api/auth/logout")
def logout(authorization: str | None = Header(default=None)) -> dict[str, str]:
    token = authorization.removeprefix("Bearer ").strip() if authorization else ""
    SESSIONS.pop(token, None)
    with get_connection() as connection:
        connection.execute("DELETE FROM sessions WHERE token = ?", (token,))
        connection.commit()
    return {"status": "logged_out"}

@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "datanova-api"}

@app.get("/api/datasets")
def list_datasets(authorization: str | None = Header(default=None)) -> dict:
    user_id = authenticated_user(authorization)
    with get_connection() as connection:
        rows = connection.execute(f"SELECT * FROM datasets WHERE user_id = ? AND status IN ({','.join('?' * len(READY_STATUSES))}) ORDER BY created_at DESC, id DESC", (user_id, *READY_STATUSES)).fetchall()
    datasets = []
    for row in rows:
        try:
            datasets.append(dataset_response(row))
        except Exception:
            logger.exception("[DATASETS] Skipping unreadable dataset %s", row["id"])
    return {"datasets": datasets}

@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...), authorization: str | None = Header(default=None)) -> dict:
    user_id = authenticated_user(authorization)
    original_name = Path(file.filename or "dataset").name
    logger.info("[UPLOAD] Request received: user=%s file=%s", user_id, original_name)
    suffix = Path(original_name).suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise HTTPException(status_code=400, detail="Upload a CSV (.csv) or Excel (.xlsx, .xls) file.")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The selected file is empty.")
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Files must be smaller than 25 MB.")
    stored_name = f"{uuid.uuid4().hex}{suffix}"
    destination = DATA_DIR / stored_name
    destination.write_bytes(content)
    logger.info("[UPLOAD] Saved original: %s (%s bytes)", destination, len(content))
    with get_connection() as connection:
        cursor = connection.execute("INSERT INTO datasets (user_id, name, stored_name, file_size, original_filename, file_path, file_type, status, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'processing', CURRENT_TIMESTAMP)", (user_id, original_name, stored_name, len(content), original_name, str(destination), suffix))
        dataset_id = int(cursor.lastrowid)
        connection.commit()
    logger.info("[UPLOAD] Dataset record %s created (status=processing)", dataset_id)
    processed_path: Path | None = None
    try:
        frame = load_frame(destination)
        metadata = profile_frame(frame, len(content))
        processed = preprocess_frame(frame)
        processed_name = f"processed_{uuid.uuid4().hex}.csv"
        processed_path = DATA_DIR / processed_name
        processed.to_csv(processed_path, index=False)
    except Exception as exc:
        logger.exception("[UPLOAD] Processing failed for dataset %s", dataset_id)
        destination.unlink(missing_ok=True)
        if processed_path:
            processed_path.unlink(missing_ok=True)
        with get_connection() as connection:
            connection.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
            connection.commit()
        raise HTTPException(status_code=422, detail=f"The dataset could not be processed: {exc}") from exc
    with get_connection() as connection:
        connection.execute("UPDATE datasets SET processed_name = ?, processed_file_path = ?, row_count = ?, column_count = ?, quality_score = ?, status = 'ready', metadata_json = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (processed_name, str(processed_path), metadata["rows"], metadata["columns"], metadata["quality_score"], json.dumps(metadata, default=str), dataset_id))
        row = connection.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
        connection.commit()
    logger.info("[UPLOAD] Dataset %s ready: rows=%s columns=%s quality=%s processed=%s", dataset_id, metadata["rows"], metadata["columns"], metadata["quality_score"], processed_path)
    return {**dataset_response(row), "file_name": original_name, "row_count": metadata["rows"], "column_count": metadata["columns"], "columns_list": metadata["column_names"], "upload_status": "ready", "metadata": metadata}

@app.get("/api/datasets/{dataset_id}")
def get_dataset(dataset_id: int, authorization: str | None = Header(default=None)) -> dict:
    user_id = authenticated_user(authorization)
    row = owned_dataset(dataset_id, user_id)
    return dataset_response(row)

@app.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id: int, authorization: str | None = Header(default=None)) -> dict[str, str]:
    user_id = authenticated_user(authorization)
    with get_connection() as connection:
        row = connection.execute("SELECT * FROM datasets WHERE id = ? AND user_id = ?", (dataset_id, user_id)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Dataset not found.")
        dataset_path(row).unlink(missing_ok=True)
        if row["processed_name"]:
            (DATA_DIR / row["processed_name"]).unlink(missing_ok=True)
        connection.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
        connection.commit()
    return {"status": "deleted"}

@app.get("/api/datasets/{dataset_id}/dashboard")
def get_dashboard(dataset_id: int, authorization: str | None = Header(default=None)) -> dict:
    user_id = authenticated_user(authorization)
    row = owned_dataset(dataset_id, user_id)
    return {"dataset": dataset_response(row), **dashboard_summary(load_analysis_frame(row), int(row["file_size"]), dataset_profile(row))}

@app.get("/api/datasets/{dataset_id}/insights")
def get_insights(dataset_id: int, authorization: str | None = Header(default=None)) -> dict:
    user_id = authenticated_user(authorization)
    row = owned_dataset(dataset_id, user_id)
    insights = generate_insights(load_analysis_frame(row), dataset_profile(row))
    return {"dataset_id": str(row["id"]), "dataset_name": row["name"], "insights": insights}

@app.get("/api/datasets/{dataset_id}/profile")
def get_profile(dataset_id: int, authorization: str | None = Header(default=None)) -> dict:
    return get_dataset(dataset_id, authorization)

@app.get("/api/datasets/{dataset_id}/quality")
def get_quality(dataset_id: int, authorization: str | None = Header(default=None)) -> dict:
    data = get_dataset(dataset_id, authorization)
    return {"dataset_id": data["dataset_id"], "quality_score": data["quality_score"], "missing_values": data["missing_values"], "duplicates": data["duplicates"], "columns": data["columns_detail"]}

@app.get("/api/datasets/{dataset_id}/download")
def download_dataset(dataset_id: int, authorization: str | None = Header(default=None), processed: bool = False) -> FileResponse:
    user_id = authenticated_user(authorization)
    row = owned_dataset(dataset_id, user_id)
    path = DATA_DIR / row["processed_name"] if processed and row["processed_name"] else dataset_path(row)
    filename = f"processed_{row['name']}.csv" if processed and row["processed_name"] else row["name"]
    return FileResponse(path, filename=filename, media_type="text/csv")

@app.get("/api/datasets/{dataset_id}/forecast")
def get_forecast(dataset_id: int, authorization: str | None = Header(default=None)) -> dict:
    user_id = authenticated_user(authorization)
    row = owned_dataset(dataset_id, user_id)
    return forecast_summary(load_analysis_frame(row))

@app.post("/api/query")
def query_dataset(request: QueryRequest, authorization: str | None = Header(default=None)) -> dict:
    user_id = authenticated_user(authorization)
    with get_connection() as connection:
        if request.dataset_id is not None:
            row = connection.execute("SELECT * FROM datasets WHERE user_id = ? AND id = ? AND status IN ('ready', 'processed')", (user_id, request.dataset_id)).fetchone()
        elif request.dataset_name:
            row = connection.execute("SELECT * FROM datasets WHERE user_id = ? AND name = ? AND status IN ('ready', 'processed') ORDER BY created_at DESC LIMIT 1", (user_id, Path(request.dataset_name).name)).fetchone()
        else:
            row = None
    if row is None:
        raise HTTPException(status_code=404, detail="Upload or select a dataset before asking a question.")
    try:
        return analyze_question(load_analysis_frame(row), request.question)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"This question could not be answered: {exc}") from exc
