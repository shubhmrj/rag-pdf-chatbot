import gc
import shutil
import tempfile
import time
import uuid
from pathlib import Path

from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from .config import DB_DIR, DB_LOCK
from .embeddings import get_embeddings


def is_ready() -> bool:
    return DB_DIR.exists() and any(DB_DIR.iterdir())


def _client_settings(path: Path) -> Settings:
    return Settings(
        is_persistent=True,
        persist_directory=str(path),
        allow_reset=True,
        anonymized_telemetry=False,
    )


def _safe_rmtree(path: Path, retries: int = 10, delay: float = 0.3) -> None:
    if not path.exists():
        return

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            shutil.rmtree(path)
            return
        except (PermissionError, OSError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(delay * attempt)

    if last_error:
        raise last_error


def _release_store(store) -> None:
    if store is None:
        return

    client = getattr(store, "_client", None)
    persist = getattr(store, "persist", None)

    try:
        if callable(persist):
            persist()
    except Exception:
        pass

    del store
    gc.collect()

    for method_name in ("close", "_close"):
        method = getattr(client, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:
                pass

    gc.collect()
    time.sleep(0.2)


def replace_index(documents: list[Document]) -> None:
    with DB_LOCK:
        DB_DIR.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix=f"{DB_DIR.name}_tmp_", dir=DB_DIR.parent))
        backup_dir = DB_DIR.with_name(f"{DB_DIR.name}_backup_{uuid.uuid4().hex[:8]}")
        store = None

        try:
            store = Chroma.from_documents(
                documents=documents,
                embedding=get_embeddings(),
                persist_directory=str(temp_dir),
                client_settings=_client_settings(temp_dir),
            )
        finally:
            _release_store(store)

        try:
            if DB_DIR.exists():
                shutil.move(str(DB_DIR), str(backup_dir))

            shutil.move(str(temp_dir), str(DB_DIR))
            _safe_rmtree(backup_dir)
        except Exception:
            if temp_dir.exists():
                _safe_rmtree(temp_dir)

            if backup_dir.exists() and not DB_DIR.exists():
                shutil.move(str(backup_dir), str(DB_DIR))

            raise


def search_documents(question: str, k: int) -> list[Document]:
    with DB_LOCK:
        store = Chroma(
            persist_directory=str(DB_DIR),
            embedding_function=get_embeddings(),
            client_settings=_client_settings(DB_DIR),
        )
        try:
            return store.similarity_search(question, k=k)
        finally:
            _release_store(store)
