#!/usr/bin/env python3
"""Merge curated public publications into the showcase without reading private data."""
import argparse
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit


DATA = Path(__file__).resolve().parents[1] / "docs" / "lab-showcase" / "data"
ORIGIN = "author-publication"
SECTION = "Публикации Андрея"


def read_snapshot(path, variable):
    return json.loads(path.read_text(encoding="utf-8").split(
        f"window.{variable}=", 1)[1].strip().removesuffix(";"))


def write_snapshot(path, variable, data):
    header = "/* Демонстрационные данные: открытые публикации и вымышленный учебный проект. */\n"
    path.write_text(header + f"window.{variable}=" + json.dumps(
        data, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")


def validate(manifest):
    topics = {t["slug"] for t in manifest["topics"]}
    ids, sources = set(), set()
    for paper in manifest["papers"]:
        assert paper["id"].startswith("pub") and len(paper["id"]) == 8
        assert paper["id"] not in ids
        assert paper["source_url"] not in sources
        ids.add(paper["id"])
        sources.add(paper["source_url"])
        assert any("Veprikov" in author for author in paper["authors"])
        assert paper["topic"] in topics
        assert paper["title"] and paper["summary"] and paper["claims"]
        urls = [paper["source_url"], paper["venue_url"]]
        if paper.get("code_url"):
            urls.append(paper["code_url"])
        for claim in paper["claims"]:
            assert claim["kind"] in {"method", "empirical", "theoretical", "definition"}
            assert claim["text"] and claim["locator"]
            urls.append(claim["source_url"])
        assert all(urlsplit(url).scheme == "https" and urlsplit(url).hostname for url in urls)
    assert all(q["paper"] in ids and q["text"] for q in manifest["questions"])


def build(data_dir=DATA):
    manifest = json.loads((data_dir / "publications.json").read_text(encoding="utf-8"))
    validate(manifest)
    library = read_snapshot(data_dir / "library-snapshot.js", "LAB_LIBRARY")
    tree = read_snapshot(data_dir / "atlas-tree.js", "LAB_TREE")
    demo = read_snapshot(data_dir / "demo-flag.js", "LAB_DEMO")
    existing = [p for p in library["papers"] if p.get("origin") != ORIGIN]
    assert not {p["id"] for p in existing} & {p["id"] for p in manifest["papers"]}
    papers = []
    for paper in manifest["papers"]:
        claims = [dict(k=c["kind"], s=c["text"], q="", v=False,
                       source_url=c["source_url"], locator=c["locator"])
                  for c in paper["claims"]]
        papers.append(dict(
            id=paper["id"], ax=paper["arxiv"], key=None, t=paper["title"],
            au=", ".join(paper["authors"]), y=paper["year"], v=paper["venue"],
            s=paper["summary"], sr=paper["summary"], ab=paper["summary"],
            f=manifest["folder"], th=None, nc=len(claims), ns=1, nm=0, c=claims,
            origin=ORIGIN, source_url=paper["source_url"], venue_url=paper["venue_url"],
            code_url=paper.get("code_url"),
            reading_scope="; ".join(dict.fromkeys(c["locator"] for c in claims))))
    library["papers"] = papers + existing
    counts = Counter(p["f"] for p in library["papers"])
    library["folders"] = [dict(f=f, n=n) for f, n in counts.items()]
    abstract = "Открытые статьи и препринты Андрея Веприкова и соавторов: методы, результаты и ссылки на оригиналы."
    tree["sections"] = [dict(name=SECTION, abstract=abstract, folders=[manifest["folder"]])] + [
        s for s in tree["sections"] if s["name"] != SECTION]
    tree["folders"] = [dict(f=manifest["folder"], a=abstract, rest=[], sub=[
        dict(slug=t["slug"], t=t["title"], a=t["summary"],
             p=[p["id"] for p in manifest["papers"] if p["topic"] == t["slug"]])
        for t in manifest["topics"]])] + [
        f for f in tree["folders"] if f["f"] != manifest["folder"]]
    demo.update(
        label="Публикации · BRAIn Lab",
        note="Статьи Андрея Веприкова и соавторов — настоящие открытые публикации и препринты. "
             "Проект Демо-лаборатории, его прогоны, числа и учебные статьи вымышлены. "
             "Поиск работает в браузере; приватная база лаборатории не подключена.",
        publication_folder=manifest["folder"], publication_count=len(papers),
        hints=[q["text"] for q in manifest["questions"]])
    write_snapshot(data_dir / "library-snapshot.js", "LAB_LIBRARY", library)
    write_snapshot(data_dir / "atlas-tree.js", "LAB_TREE", tree)
    write_snapshot(data_dir / "demo-flag.js", "LAB_DEMO", demo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    build(parser.parse_args().data_dir)
