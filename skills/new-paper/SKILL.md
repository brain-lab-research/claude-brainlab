---
name: new-paper
description: Track a new research idea or paper project with an Obsidian hub, without creating its code repository.
---

# Track a research idea

## Shared publication contract

Before writing shared records, read `lab-knowledge/references/record-contract.md` from the
installed skills directory and the live MCP schema. Use the already authorized scope,
resolve the existing destination, and verify stored content. Keep failed publications pending.


Create a hub for a new paper or research idea. Repository creation and shared workspace
provisioning belong to `create-project` and `lab-project-onboarding` when requested.

## Resolve the existing context

Read the current vault conventions and the client's `obsidian-projects.json`. Reuse an
existing project by its subject and aliases, not only its exact title. Resolve its complete
mapped path. Paper projects use `Papers/<theme>/<slug>` when the vault is organized by theme;
never flatten a known mapping or create a `Research` root.

Use details already supplied by the user or available in the project's sources. Ask one
focused question for any missing choice that changes the hub. Do not demand a nine-field
form when the title, collaborators, or paper link are already known. Inspect existing tags
and theme folders before proposing new categories.

## Create the hub

If a paper link is provided, read its primary source or local TeX to establish the idea.
Do not infer missing collaborators, affiliations, results, or acceptance status.

Use a lowercase hyphenated slug for a new project, preserving an existing slug and links.
Create or link people cards only for identified people; preserve their existing roles and
project links. Record intended filesystem paths as planned if no repository exists yet.

Adapt the existing project-card format. A minimal hub is:

```markdown
---
обновлено: <DD-MM-YYYY>
участники:
  - "[[people/<existing-person>]]"
tags:
  - <existing-topic-tag>
  - тип/<type>
  - статус/<status>
---
# <Project name>

## Суть
<Source-grounded idea and current question>

## Участники
- [[people/<existing-person>]]

## Пути
- проект: <verified path, or clearly marked planned path>
- источник: <paper or other source link>
```

Add `org/` and `conf/` tags only when known. Keep tags in `tags`, omit unknown values, and
follow existing links rather than duplicating people or projects. Use the vault's current
icon and color procedure when it is part of the requested setup; do not hard-code plugin
JSON or restart the user's app from a stale example.

## Register and check

For a new mapping, preserve unrelated entries and record the full relative vault path,
including the theme where applicable. Use the project's actual filesystem root and the
appropriate mapped Papers, Projects, or Staff root. A hub-only request does not authorize
creating a repository, a server workspace, or shared MCP/Yonote records.

Read back the hub and verify its links and mapping. Report the created path and what still
exists only as a plan.
