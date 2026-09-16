"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const root = path.resolve(__dirname, "..");
const data = path.join(root, "docs/lab-showcase/data");
const search = require(path.join(root, "docs/lab-showcase/search-core.js"));
const ctx = { window: {} };
for (const name of ["library-snapshot", "base-snapshot", "atlas-tree"])
  vm.runInNewContext(fs.readFileSync(path.join(data, `${name}.js`), "utf8"), ctx);
const lib = ctx.window.LAB_LIBRARY;
const index = search.build({ lib, base: ctx.window.LAB_BASE, tree: ctx.window.LAB_TREE });
const manifest = JSON.parse(fs.readFileSync(path.join(data, "publications.json"), "utf8"));
for (const question of manifest.questions) {
  const hits = index.search(question.text).filter(hit => hit.kind === "paper");
  assert.equal(hits[0]?.id, question.paper, question.text);
}
for (const paper of manifest.papers) {
  const hits = index.search(paper.title).filter(hit => hit.kind === "paper");
  assert.equal(hits[0]?.id, paper.id, paper.title);
}

// Exercise the actual excerpt selector: a question about the method must not
// receive the longest (unrelated) benchmark statement from the same paper.
const app = fs.readFileSync(path.join(root, "docs/lab-showcase/app.js"), "utf8");
const start = app.indexOf("  function paperClaim(");
const end = app.indexOf("\n  }", start) + 4;
assert(start >= 0 && end > start);
const select = vm.runInNewContext(`(${app.slice(start, end).trim()})`, {
  window: { LabSearch: search },
});
const rankedStart = app.indexOf("  function snapshotRanked(");
const rankedEnd = app.indexOf("\n  }", rankedStart) + 4;
const ranked = vm.runInNewContext(`(${app.slice(rankedStart, rankedEnd).trim()})`, {
  Set,
  searchIndex: index,
  NODE_KINDS: search.NODE_KINDS,
  byCode: Object.fromEntries(ctx.window.LAB_BASE.records.map(r => [r.code, r])),
  papersById: Object.fromEntries(lib.papers.map(p => [p.id, p])),
  paperClaim: select,
});
for (const question of manifest.questions)
  assert.equal(ranked(question.text)[0]?.entity_id, question.paper, `Rendered ranking: ${question.text}`);
const weight = lib.papers.find(p => p.id === "pub00014");
assert.match(select(weight, manifest.questions[0].text).s, /выбират|выбор|выбира|отбор/);
const fisher = lib.papers.find(p => p.id === "pub00009");
assert.match(select(fisher, "как DyKAF приближает матрицу Фишера").s, /Фишера/);
console.log(`PASS: ${manifest.questions.length} examples, ${manifest.papers.length} titles, relevant excerpts`);
