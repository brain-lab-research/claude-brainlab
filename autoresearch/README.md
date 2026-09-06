# autoresearch — переиспользуемый апгрейд автономного research-лупа

Проект-агностичный тулкит: 8 механизмов из лучших autonomous-research систем (AIDE,
AI Scientist-v2, Google AI co-scientist, ml-intern, claude-autoresearch, ResearchOS),
адаптированных под **theory-трек с тройными воротами** (derive + proof + code + critic).
Работает поверх `IdeaGraph/graph.json` любого проекта. **Stdlib-only** (sklearn
опционален — есть фолбэк), поэтому запускается и на bare-Python серверах.

## Установка / расположение
- Пакет: `~/.claude/autoresearch/` (этот каталог). Импорт: `PYTHONPATH=~/.claude`.
- Лаунчер: `~/.claude/autoresearch/ar` → `ar <cmd> [--project PATH] ...`.
- Деплой на сервер: `rsync -az --exclude=__pycache__ ~/.claude/autoresearch/ <host>:~/.claude/autoresearch/`.

## Подключить новый проект
```bash
ar init   --project /path/to/project --name myproj --domain "о чём проект (1-2 предлож.)"
ar migrate --project /path/to/project       # +поля elo/attempts/scoop/failure_class, бэкап, канвас
ar status  --project /path/to/project
```
Требования к проекту: `IdeaGraph/graph.json` (+ `idea_graph_canvas.py` рядом). Корень
проекта = папка с `IdeaGraph/graph.json`; задаётся `--project`, env `AUTORESEARCH_PROJECT`
или автодетектом от cwd. Конфиг проекта: `IdeaGraph/autoresearch.json` (домен, пути,
опц. `research_cycle_path`).

## Команды
| cmd | # | что делает |
|-----|---|-----------|
| `status` | — | фронтир (selector) + таксономия провалов + бэкенд дедупа |
| `doctor` | — | жёсткая live-диагностика dashboard/log/worker-pool; `--smoke-prompts` ловит prompt `.format()` падения до рестарта |
| `dedup "<идея>"` | 3 | блок дубль-угла перед раундом |
| `select -k N` | 2 | авто-выбор узлов фронтира |
| `scoop <id>` | 7 | prior-art по локальным заметкам |
| `evolve -n N` | 4 | кросс-веточные гибриды топ-Elo |
| `meta` | 5 | hard-rules из критики → framework_overlay |
| `draft` | 8 | выжившие узлы → LaTeX-стабы |
| `round [-k N] [--dry]` | 1,2,7,4,5,8 | полный раунд |

## Поток `round`
```
selector(#2) → scoop(#7) → [research_cycle: derive+proof+code+critic, если задан путь]
            → meta(#5) → evolution(#4, через dedup #3) → autodraft(#8) → save+канвас
```
Без `research_cycle_path` авто-цикл пропускается, печатаются выбранные узлы — их
прогоняет оркестрирующий агент сам (серверный режим).

## Архитектура файлов
`config.py` (резолв проекта + frozen-dataclass) · `graph_io.py` (IO+бэкап+канвас) ·
`engines.py` (codex/claude, timeout+killpg, fallback codex→claude) · `dedup_guard.py`
(sklearn|stdlib) · `tournament.py` (Elo) · `selector.py` · `evolution.py` ·
`meta_review.py` · `failure_taxonomy.py` · `scoop_gate.py` · `autodraft.py` ·
`orchestrator.py` (CLI). Каждый <140 строк, type hints, без глобалов.

## Заметки
- Эмбеддинги не используются (нет ключей/sentence-transformers на серверах) →
  TF-IDF (sklearn) или TF-косинус (stdlib) по символьным n-граммам.
- Узловая схема расширяется аддитивно; старый граф читается без миграции.
- Глобальная директива: `~/.claude/rules/autoresearch.md`.
