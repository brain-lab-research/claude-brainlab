#!/usr/bin/env python3
"""Publish a parsed reading note using the configured caller and verify stored content.

Preserve unrelated sections, including original passages added by other workflows.
The --verify flag remains accepted; read-back is always required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import os
import tempfile
import urllib.request
from pathlib import Path

# The parser belongs to the separately installed Lab Knowledge repository.
VAULT = Path(os.environ['OBSIDIAN_VAULT']).expanduser() if os.environ.get('OBSIDIAN_VAULT') else None
ZOTERO = Path(os.environ.get("ZOTERO_SQLITE", "~/Zotero/zotero.sqlite")).expanduser()
REPO = Path(os.environ.get("LAB_KNOWLEDGE_REPO", os.environ.get("BRAINLAB_REPO", "~/lab-knowledge"))).expanduser()
LIBRARY_SYNC = REPO / "services" / "lab-knowledge" / "scripts" / "library_sync"


def parse(arxiv: str | None) -> list[dict]:
    """Разобрать хранилище и оставить нужную статью."""
    if VAULT is None:
        raise RuntimeError('Set OBSIDIAN_VAULT before parsing the local library')
    if not (LIBRARY_SYNC / 'parse_library.py').is_file():
        raise RuntimeError('Set LAB_KNOWLEDGE_REPO to the installed Lab Knowledge checkout')
    with tempfile.NamedTemporaryFile("r+", suffix=".json", delete=False) as handle:
        out = Path(handle.name)
    command = [sys.executable, str(LIBRARY_SYNC / "parse_library.py"),
               "--vault", str(VAULT), "--out", str(out)]
    if ZOTERO.is_file():
        command += ["--zotero", str(ZOTERO)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(out.read_text(encoding="utf-8"))
    papers = payload.get("papers") if isinstance(payload, dict) else payload
    out.unlink(missing_ok=True)
    if arxiv is None:
        return papers
    wanted = arxiv.strip()
    return [paper for paper in papers if (paper.get("arxiv_id") or "") == wanted]


def existing(arxiv: str) -> dict | None:
    """Что о статье уже знает общая база: её могли добавить до нас."""
    payload = _ask("get_paper", {"arxiv_id": arxiv})
    if payload is None:
        return None
    return {
        "paper": payload.get("paper") or {},
        "sections": payload.get("chunks") or [],
        "claims": payload.get("claims") or [],
    }


def merge_paper(paper: dict, known: dict | None, *, force: bool = False) -> dict:
    """Preserve other contributors' sections and metadata during an approved note update."""
    if known is None:
        return dict(paper)
    previous = known["paper"]
    same_note = bool(paper.get("obsidian_path")) and (
        previous.get("obsidian_path") == paper["obsidian_path"])
    sections = {chunk["section"]: dict(chunk) for chunk in known["sections"]}
    replaced = []
    for chunk in paper.get("sections") or []:
        old = sections.get(chunk["section"])
        if old and old.get("content") != chunk.get("content") and not (same_note or force):
            # Раньше здесь стоял RuntimeError, и добавление останавливалось целиком: агент
            # упирался в стоп без пути вперёд, а разбор не доезжал вовсе. Владелец: «нужно
            # сделать так, чтобы агенту Гермесу… всё хорошо добавлялось».
            #
            # Чужой разбор при этом всё равно не затирается: он остаётся на месте, а новый
            # текст приезжает соседним разделом с пометкой, откуда он. Так обе версии
            # доезжают целиком, и человек видит расхождение вместо того, чтобы гадать,
            # почему статья не обновилась.
            source_path = paper.get("obsidian_path") or "неизвестный разбор"
            source = Path(source_path).name
            suffix = hashlib.sha256(source_path.encode()).hexdigest()[:12]
            name = f"{chunk['section']} — второй разбор ({source}, {suffix})"
            sections[name] = {**chunk, "section": name}
            replaced.append(chunk["section"])
            continue
        sections[chunk["section"]] = dict(chunk)
    if replaced:
        print("разделы разошлись с уже сохранёнными и добавлены рядом, а не поверх: "
              + ", ".join(replaced), file=sys.stderr)
    # Only accepted input fields, not server-generated IDs or counters.
    fields = ("title", "arxiv_id", "doi", "zotero_key", "citation_key", "authors", "year",
              "venue", "abstract", "summary_ru", "obsidian_path", "library_folder",
              "library_folders", "tags", "url", "code_url")
    merged = {key: previous[key] for key in fields if previous.get(key) not in (None, "", [])}
    merged.update({key: value for key, value in paper.items()
                   if key in fields and value not in (None, "", [])})
    if previous.get('obsidian_path') and not same_note:
        merged['obsidian_path'] = previous['obsidian_path']
    if previous.get("tags"):
        merged["tags"] = list(dict.fromkeys([*previous["tags"], *paper.get("tags", [])]))
    if previous.get("theme_slugs"):
        merged["themes"] = list(dict.fromkeys([*previous["theme_slugs"], *paper.get("themes", [])]))
    merged["sections"] = [{"section": x["section"], "content": x["content"], "ordinal": i}
                          for i, x in enumerate(sections.values())]
    return merged


def _connection() -> tuple[str, dict]:
    """Use this installation's caller, never a server-wide lead token fallback."""
    if os.environ.get("LAB_MCP_URL") and os.environ.get("LAB_MCP_TOKEN"):
        return os.environ["LAB_MCP_URL"], {"Authorization": "Bearer " + os.environ["LAB_MCP_TOKEN"]}
    hermes = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
    if (hermes / "config.yaml").is_file():
        import re
        import yaml
        config = yaml.safe_load((hermes / "config.yaml").read_text())["mcp_servers"]["lab-knowledge"]
        if config.get("enabled", True) is False:
            raise RuntimeError("Lab MCP disabled in this Hermes profile")
        values = {}
        for path in [Path.home() / ".hermes/.env", hermes / ".env"]:
            if path.exists():
                for line in path.read_text().splitlines():
                    if "=" in line and not line.lstrip().startswith("#"):
                        key, value = line.split("=", 1)
                        values[key.strip()] = value.strip().strip(chr(34)).strip(chr(39))
        values.update(os.environ)
        headers = {key: re.sub(r"\$\{([^}]+)\}", lambda m: values[m[1]], value)
                   for key, value in config.get("headers", {}).items()}
        return config["url"], headers
    settings = Path.home() / ".claude/settings.json"
    if settings.exists():
        config = json.loads(settings.read_text()).get("mcpServers", {}).get("lab-knowledge")
        if config:
            return config["url"], config.get("headers", {})
    raise RuntimeError("No Lab MCP configured for this caller; publication remains pending")


def _ask(tool: str, arguments: dict) -> dict | None:
    url, auth = _connection()
    headers = {**auth, "Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    def rpc(method: str, params: dict) -> dict:
        body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        request = urllib.request.Request(url, json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            session = response.headers.get("Mcp-Session-Id")
            if session:
                headers["Mcp-Session-Id"] = session
            raw = response.read().decode()
        if not raw.lstrip().startswith("{"):
            raw = next(line[5:].strip() for line in raw.splitlines() if line.startswith("data:"))
        payload = json.loads(raw)
        if "error" in payload:
            raise RuntimeError("MCP protocol error: " + str(payload["error"].get("message", "unknown")))
        return payload["result"]
    rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                       "clientInfo": {"name": "paper-ingest", "version": "2"}})
    result = rpc("tools/call", {"name": tool, "arguments": arguments})
    text = next((block.get("text", "") for block in result.get("content", [])
                 if block.get("type") == "text"), "{}")
    if result.get("isError"):
        if tool == "get_paper" and "paper not found" in text.lower():
            return None
        raise RuntimeError("MCP rejected " + tool + ": " + text)
    return result.get("structuredContent") or json.loads(text)


def verify(arxiv: str, expected: dict | None = None) -> bool:
    known = existing(arxiv)
    if known is None:
        return False
    if expected is None:
        raise ValueError("Verification requires the expected content, not just existence")
    actual = known["paper"]
    for key in ("title", "arxiv_id", "doi", "zotero_key", "citation_key", "authors", "year",
                "venue", "abstract", "summary_ru", "obsidian_path", "library_folder",
                "tags", "url", "code_url"):
        if key in expected and actual.get(key) != expected[key]:
            return False
    if expected.get("themes") and not set(expected["themes"]).issubset(actual.get("theme_slugs", [])):
        return False
    sections = {chunk["section"]: chunk.get("content", "") for chunk in known["sections"]}
    if not all(sections.get(chunk["section"]) == chunk.get("content", "")
               for chunk in expected.get("sections", [])):
        return False
    return all(any(all(saved.get(key, "") == value for key, value in claim.items())
                   for saved in known["claims"])
               for claim in expected.get("claims", []))


def sync_paper(paper: dict, *, force: bool = False) -> None:
    arxiv = paper.get("arxiv_id")
    if not arxiv:
        # У книги и постера конференции arXiv нет вовсе, и с 10-09-2026 служба принимает их
        # по пути заметки. Этот путь синхронизации всё же требует arXiv: он сопоставляет
        # статью вызовом get_paper по нему. Для остального работает library_sync, где
        # сопоставление идёт по любому естественному ключу, включая путь.
        raise RuntimeError(
            "This sync matches papers by arXiv ID. A book or a poster without one goes "
            "through library_sync: parse_library.py -> push_library.py, which matches by "
            "any natural key including the note path."
        )
    known = existing(arxiv)
    if known is None:
        # Create only the natural identity, then read again: another writer may
        # have created this paper since the lookup. Never replace its sections.
        _ask("upsert_paper", {"title": paper["title"], "arxiv_id": arxiv})
        known = existing(arxiv)
        if known is None:
            raise RuntimeError("Paper creation was not confirmed; publication remains pending")
    outgoing = merge_paper(paper, known, force=force)
    metadata = {key: value for key, value in outgoing.items() if key != "sections"}
    if any(known["paper"].get("theme_slugs" if key == "themes" else key) != value
           for key, value in metadata.items()):
        _ask("upsert_paper", metadata)
    previous = {chunk["section"]: chunk.get("content", "") for chunk in known["sections"]}
    for chunk in outgoing.get("sections") or []:
        if previous.get(chunk["section"]) != chunk.get("content", ""):
            _ask("add_paper_section", {"paper_id": known["paper"]["id"],
                 "section": chunk["section"], "content": chunk["content"]})
    claims = [{"statement": claim["statement"], "kind": claim.get("kind") or "empirical",
               "locator": claim.get("locator") or "", "quote": claim.get("quote") or ""}
              for claim in paper.get("claims") or []]
    for claim in claims:
        # MCP deduplicates by quotation or (for paraphrases) statement and kind.
        # The live tool has no caller-supplied idempotency-key parameter.
        _ask("record_paper_claim", {"paper_id": known["paper"]["id"], **claim})
    outgoing["claims"] = claims
    if not verify(arxiv, outgoing):
        raise RuntimeError("Read-back differs from the published paper; publication remains pending")
    print(f"{arxiv}: metadata, {len(outgoing.get('sections', []))} sections "
          f"and {len(claims)} claims verified")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arxiv", help="номер статьи, например 2312.00752")
    parser.add_argument("--all", action="store_true",
                        help="отправить всю библиотеку: полная сверка")
    parser.add_argument("--verify", action="store_true",
                        help="после отправки спросить у базы, легла ли статья")
    parser.add_argument("--force", action="store_true",
                        help="заменить чужой разбор своим, даже если чужой полнее")
    args = parser.parse_args()

    if not (args.arxiv or args.all):
        parser.error("нужен --arxiv или --all")

    papers = parse(None if args.all else args.arxiv)
    if not papers:
        print(f"в хранилище нет заметки с arXiv {args.arxiv}: "
              "сперва должна пройти запись заметки")
        return 1
    for paper in papers:
        sync_paper(paper, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
