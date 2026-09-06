"""koi-структура как ОСНОВНОЙ store движка autoresearch (без мостов).

Читает/пишет проект как:
  koi-structure/project.md     — дерево koi (problem→cause→cause_evidence→method)
  koi-structure/engine.json    — НАШИ поля тех же узлов (elo/attempts/scoop/...
                                 status-нюанс/idea/open/plan/short) + xref + root_idea
  koi-structure/research.json  — выводы (генерится из выживших узлов; читает ResearchOS)

load_graph_like()/save_graph_like() отдают/принимают ТОТ ЖЕ dict-формат, что и
graph_io (nodes[*]: id/short/parent/col/status/idea/verdict/open/plan/elo/attempts/
scoop/failure_class + xref), поэтому selector/dedup/tournament/taxonomy/autodraft
работают над koi-store без изменений.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from . import koi_compat

# наш статус -> koi verdict (enum open|supported|refuted)
_STATUS2VERDICT = {"proven": "supported", "confirmed_emp": "supported",
                   "rejected": "refuted", "refuted_breadth": "refuted"}
# обратное (когда узла нет в engine.json)
_VERDICT2STATUS = {"supported": "proven", "refuted": "rejected", "open": "open"}
_SURVIVED = ("proven", "confirmed_emp", "conditional")


def _paths(koi_dir: str):
    return (os.path.join(koi_dir, "project.md"),
            os.path.join(koi_dir, "engine.json"),
            os.path.join(koi_dir, "research.json"))


def _depth(nid: str, parent: Dict[str, str]) -> int:
    d, seen, p = 0, {nid}, parent.get(nid)
    while p and p not in seen:
        seen.add(p); d += 1; p = parent.get(p)
    return d


def _node_type(depth: int) -> str:
    return ["problem", "cause", "cause_evidence", "method"][min(depth, 3)]


# ---------------- READ ----------------

def load_graph_like(koi_dir: str, cfg: Any = None) -> Dict[str, Any]:
    md_io, _ = koi_compat.koi_modules(cfg)
    proj_md, eng_path, _ = _paths(koi_dir)
    project = md_io.parse_project_md(open(proj_md, encoding="utf-8").read())
    eng = {}
    if os.path.exists(eng_path):
        eng = json.load(open(eng_path, encoding="utf-8"))
    en = eng.get("nodes", {})
    parent = {n.id: (n.parent_id or "") for n in project.nodes}
    nodes: List[Dict[str, Any]] = []
    for n in project.nodes:
        e = en.get(n.id, {})
        nodes.append({
            "id": n.id,
            "short": e.get("short") or n.title,
            "parent": n.parent_id,
            "col": e.get("col", _depth(n.id, parent)),
            "status": e.get("status") or _VERDICT2STATUS.get(n.verdict.value, "open"),
            "idea": e.get("idea", n.description or ""),
            "verdict": e.get("verdict", ""),
            "open": e.get("open", ""),
            "plan": e.get("plan", ""),
            "elo": e.get("elo"),
            "attempts": e.get("attempts", 0),
            "scoop": e.get("scoop"),
            "failure_class": e.get("failure_class"),
        })
    return {"nodes": nodes, "xref": eng.get("xref", []),
            "root_idea": eng.get("root_idea", ""),
            "overall_verdict": eng.get("overall_verdict", ""),
            "updated": eng.get("updated", "")}


# ---------------- WRITE ----------------

def save_graph_like(data: Dict[str, Any], koi_dir: str, cfg: Any = None) -> None:
    md_io, models = koi_compat.koi_modules(cfg)
    proj_md, eng_path, rj_path = _paths(koi_dir)
    os.makedirs(koi_dir, exist_ok=True)
    nodes = data["nodes"]
    parent = {n["id"]: (n.get("parent") or "") for n in nodes}
    ids = {n["id"] for n in nodes}

    # корень = id 'root' | status 'root' | col 0; прочие безродные -> под корень
    root = next((n for n in nodes if n["id"] == "root"), None) \
        or next((n for n in nodes if n.get("status") == "root" or n.get("col") == 0), None)
    root_id = root["id"] if root else None

    knodes = []
    for n in nodes:
        nid = n["id"]
        pid = n.get("parent")
        if nid == root_id:
            pid, depth = None, 0
        else:
            if pid not in ids:
                pid = root_id
            depth = _depth(nid, {**parent, nid: pid}) if pid else 1
        v = _STATUS2VERDICT.get(n.get("status"), "open")
        knodes.append(models.Node(
            id=nid, project_id=data.get("project_id", "wsd-muon"), parent_id=pid,
            node_type=models.NodeType(_node_type(depth)),
            title=n.get("short") or nid, description=(n.get("idea") or "").strip(),
            verdict=models.Verdict(v)))
    project = models.Project(id=data.get("project_id", "wsd-muon"),
                             title=data.get("project_title", "wsd-muon"),
                             description=data.get("root_idea", ""), nodes=knodes)
    open(proj_md, "w", encoding="utf-8").write(md_io.serialize_project_md(project))

    # engine.json — наши поля
    en = {n["id"]: {k: n.get(k) for k in
                    ("short", "idea", "open", "plan", "verdict", "status", "col",
                     "elo", "attempts", "scoop", "failure_class")} for n in nodes}
    json.dump({"version": 1, "root_idea": data.get("root_idea", ""),
               "overall_verdict": data.get("overall_verdict", ""),
               "updated": data.get("updated", ""), "nodes": en,
               "xref": data.get("xref", [])},
              open(eng_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # research.json — выводы для ResearchOS из выживших узлов
    questions = []
    for n in nodes:
        if n.get("status") in _SURVIVED and (n.get("verdict") or n.get("idea")):
            questions.append({
                "id": f"rq-{n['id']}", "method_id": n["id"],
                "question": n.get("short", n["id"]),
                "certainty": "definite" if n.get("status") in ("proven", "confirmed_emp") else "tentative",
                "importance": 5 if n.get("status") == "confirmed_emp" else 4,
                "answer": (n.get("verdict") or "")[:400],
                "narrative": (n.get("verdict") or n.get("idea") or "")[:600],
                "card_id": None})
    json.dump({"version": 1, "questions": questions},
              open(rj_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
