"""Typed experiment specifications for the autoresearch GPU loop.

The gate may request a real LLM experiment, but the supervisor must not map
every such request to the same harness.  This module keeps a small allowlist of
known experiment kinds and normalizes specs before they reach shell launch code.
"""
from __future__ import annotations

import hashlib
import json
import posixpath
import re
from typing import Any, Dict, Optional


DEFAULT_MODEL = "llama"
PYTHIA_MODEL_RE = re.compile(r"(?:EleutherAI/)?pythia-(\d+(?:\.\d+)?[a-z]?)", re.IGNORECASE)
LEGACY_CHECKPOINT_TEXT_RE = re.compile(
    r"(?:EleutherAI/)?pythia-\d+(?:\.\d+)?[a-z]?|\bPythia\b|\b(?:160m|410m)\b",
    re.IGNORECASE,
)
BARE_LLAMA_ALIASES = {"llama", "llama-style", "llama_style"}
GENERIC_HF_KINDS = {
    "real_forgetting",
    "gate_sensitivity",
    "ce_mechanism",
    "muon_decomp",
    "gnmuon_wallclock",
    "gnmuon_matched_target",
    "rho_predictor",
}


KIND_ALIASES = {
    "forgetting": "real_forgetting",
    "retention": "real_forgetting",
    "optimizer_forgetting": "legacy_llama_forgetting",
    "llama_forgetting": "legacy_llama_forgetting",
    "legacy_forgetting": "legacy_llama_forgetting",
    "llama_gnmuon": "llama_gnmuon_audit",
    "llama_gnmuon_audit": "llama_gnmuon_audit",
    "llama_retention_audit": "llama_gnmuon_audit",
    "llama_style_retention": "llama_gnmuon_audit",
    "realnet_gnmuon": "llama_gnmuon_audit",
    "gnmuon_llama": "llama_gnmuon_audit",
    "llama_checkpoint_pool_selector": "llama_checkpoint_pool_selector",
    "checkpoint_pool_selector": "llama_checkpoint_pool_selector",
    "checkpoint_selector": "llama_checkpoint_pool_selector",
    "c1_selector": "llama_checkpoint_pool_selector",
    "selector_firewall": "llama_checkpoint_pool_selector",
    "llama_selector_firewall": "llama_checkpoint_pool_selector",
    "llama_firewall_calibration_audit": "llama_firewall_calibration_audit",
    "firewall_calibration_audit": "llama_firewall_calibration_audit",
    "firewall_calibration": "llama_firewall_calibration_audit",
    "prospective_firewall": "llama_firewall_calibration_audit",
    "llama_alignment_bridge_audit": "llama_alignment_bridge_audit",
    "alignment_bridge_audit": "llama_alignment_bridge_audit",
    "source_target_alignment": "llama_alignment_bridge_audit",
    "source_target_alignment_bridge": "llama_alignment_bridge_audit",
    # решающий theorem-matched аудит A*-критика: source-бит против target-only правил на общем transcript
    "llama_source_bit_frontier_audit": "llama_source_bit_frontier_audit",
    "source_bit_frontier_audit": "llama_source_bit_frontier_audit",
    "source_bit_frontier": "llama_source_bit_frontier_audit",
    "source_bit_audit": "llama_source_bit_frontier_audit",
    "llama_proxy_falsification_grid": "llama_proxy_falsification_grid",
    "proxy_falsification_grid": "llama_proxy_falsification_grid",
    "lmo_proxy_grid": "llama_proxy_falsification_grid",
    "gn_proxy_grid": "llama_proxy_falsification_grid",
    "llama_fixed_baseline_audit": "llama_fixed_baseline_audit",
    "fixed_baseline_audit": "llama_fixed_baseline_audit",
    "fixed_baseline": "llama_fixed_baseline_audit",
    "fixed-baseline": "llama_fixed_baseline_audit",
    "baseline_purity": "llama_fixed_baseline_audit",
    "gate": "gate_sensitivity",
    "gate_covariance": "gate_sensitivity",
    "confidence_gate": "gate_sensitivity",
    "mechanism": "ce_mechanism",
    "ce_gate_mechanism": "ce_mechanism",
    "decomposition": "muon_decomp",
    "muon_checkpoint_decomp": "muon_decomp",
    "c1": "c1_retention",
    "gnmuon_retention": "c1_retention",
    "gnmuon_vs_muon": "c1_retention",
    "wallclock": "gnmuon_wallclock",
    "wall_clock": "gnmuon_wallclock",
    "gnmuon_baseline": "gnmuon_wallclock",
    "gnmuon_baselines": "gnmuon_wallclock",
    "gnbaselines": "gnmuon_wallclock",
    "baseline_completeness": "gnmuon_wallclock",
    "matched_target": "gnmuon_matched_target",
    "target_parity": "gnmuon_matched_target",
    "target_ce_parity": "gnmuon_matched_target",
    "matched_target_audit": "gnmuon_matched_target",
    "rho": "rho_predictor",
    "rho_predictor": "rho_predictor",
    "fisher_profile": "rho_predictor",
    "sigma_delta_profile": "rho_predictor",
    "depth_profile": "rho_predictor",
    "cfisher": "rho_predictor",
    "c_fisher": "rho_predictor",
}


KIND_SCRIPTS = {
    # Current legacy harness in ~/llama_exp_kit.py on GPU hosts.
    "legacy_llama_forgetting": "",
    # Project scripts copied from the project checkout to the selected GPU host.
    "real_forgetting": ".research_loop/exp_forget.py",
    "gate_sensitivity": ".research_loop/exp_gate.py",
    "ce_mechanism": ".research_loop/exp_mechanism.py",
    "muon_decomp": ".research_loop/exp_muon_decomp.py",
    "c1_retention": "Results/scripts/exp_c1_retention_conv1d.py",
    "llama_gnmuon_audit": "Results/scripts/exp_llama_gnmuon_audit.py",
    "llama_checkpoint_pool_selector": "Results/scripts/exp_llama_checkpoint_pool_selector.py",
    "llama_firewall_calibration_audit": "Results/scripts/exp_llama_checkpoint_pool_selector.py",
    "llama_alignment_bridge_audit": "Results/scripts/exp_llama_checkpoint_pool_selector.py",
    # скрипта нет на диске → экспериментатор автогенерит его под claim (+код-ревью), см. _ensure_exp_script
    "llama_source_bit_frontier_audit": "Results/scripts/exp_llama_source_bit_frontier_audit.py",
    "llama_proxy_falsification_grid": "Results/scripts/exp_llama_gnmuon_audit.py",
    "llama_fixed_baseline_audit": "Results/scripts/exp_llama_gnmuon_audit.py",
    "gnmuon_wallclock": "Results/scripts/exp_gnmuon_wallclock.py",
    "gnmuon_matched_target": "Results/scripts/exp_gnmuon_wallclock.py",
    "rho_predictor": "Results/scripts/exp_rho_predictor.py",
    # ЛЮБОЙ нужный эксперимент, которого нет в списке выше: агент сам пишет llama-скрипт под claim
    # (см. empirical._ensure_exp_script). Путь генерится по хэшу claim в canonicalize().
    "custom_llama": "",
}


_MODEL_RE = re.compile(r"^[A-Za-z0-9_.:/@+-]{1,120}$")
_ENV_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,39}$")
_ENV_VAL_RE = re.compile(r"^[A-Za-z0-9_.,:/@+=-]{0,160}$")


# Первоклассные llama-audit kinds (typed EXP_SPEC). ЕДИНСТВЕННЫЙ источник: supervisor и
# orchestrator раньше держали по СВОЕЙ копии этого набора (3 дубля) — новый kind, добавленный
# в одном месте, молча отсутствовал в других (класс багов «hardcoded-список рулит наукой»).
LLAMA_AUDIT_KINDS = frozenset({
    "llama_gnmuon_audit",
    "llama_checkpoint_pool_selector",
    "llama_firewall_calibration_audit",
    "llama_alignment_bridge_audit",
    "llama_proxy_falsification_grid",
    "llama_fixed_baseline_audit",
    "llama_source_bit_frontier_audit",
})


def known_kinds() -> str:
    # Заметка про НОВЫЕ kinds обязательна: без неё код-судья ворот отклонял EXP_SPEC_REQUEST с
    # незнакомым kind («kind отсутствует в списке»), хотя незнакомый kind легален — он идёт в
    # custom_llama-полосу и экспериментатор пишет скрипт сам (динамическая регистрация).
    return (", ".join(sorted(KIND_SCRIPTS))
            + ". НОВЫЙ kind, которого нет в списке, ТОЖЕ ВАЛИДЕН: назови его осмысленно "
              "(llama_<что_проверяем>_audit) — он автоматически пойдёт как custom_llama, "
              "и экспериментатор сам напишет скрипт под claim (не отклоняй за «незнакомый kind»)")


def normalize_model(model: Any) -> str:
    m = str(model or DEFAULT_MODEL).strip()
    if not m:
        return DEFAULT_MODEL
    if not _MODEL_RE.match(m):
        return DEFAULT_MODEL
    if is_pythia_model(m):
        return DEFAULT_MODEL  # pythia в проекте запрещена — только llama-like → сводим к llama-харнесу
    return m


def is_pythia_model(model: Any) -> bool:
    return bool(PYTHIA_MODEL_RE.search(str(model or "")))


def is_bare_llama_alias(model: Any) -> bool:
    return str(model or "").strip().lower() in BARE_LLAMA_ALIASES


def is_bare_llama_generic_hf_spec(spec: Dict[str, Any]) -> bool:
    kind = str((spec or {}).get("kind") or "").strip().lower()
    return kind in GENERIC_HF_KINDS and is_bare_llama_alias((spec or {}).get("model"))


def _mentioned_pythia_models(text: str) -> set[str]:
    out = set()
    for m in PYTHIA_MODEL_RE.finditer(text or ""):
        size = m.group(1).lower()
        out.add(f"EleutherAI/pythia-{size}")
    return out


def _enforce_model_contract(spec: Dict[str, Any], context: str, exp_kind: str = "") -> None:
    if str(spec.get("kind") or "") == "custom_llama":
        return  # кастомный эксп пишется свежим на llama_exp_kit (model=llama, без legacy/pythia harness) → пропускаем
    model = str(spec.get("model") or "")
    policy = str(exp_kind or "").strip().lower()
    mentioned = _mentioned_pythia_models(context)
    if mentioned and is_pythia_model(model) and model.lower() not in {x.lower() for x in mentioned}:
        raise ValueError(
            "EXP_SPEC.model mismatch: context mentions "
            f"{sorted(mentioned)}, but spec model is {model!r}"
        )
    if policy == "llama":
        contract_text = " ".join(str(spec.get(k) or "") for k in (
            "model", "claim", "success_metric", "confirm_rule",
        ))
        if is_pythia_model(model) or LEGACY_CHECKPOINT_TEXT_RE.search(context or "") or LEGACY_CHECKPOINT_TEXT_RE.search(contract_text):
            raise ValueError(
                "project exp_kind=llama forbids legacy checkpoint-specific typed runs or claims; "
                "use llama-style evidence only"
            )
        if is_bare_llama_generic_hf_spec(spec):
            raise ValueError(
                f"EXP_SPEC.kind={spec.get('kind')!r} uses a generic HF/llm_exp_kit harness; "
                "model='llama' is a policy alias, not a loadable model id. Use a llama_exp_kit "
                "kind such as legacy_llama_forgetting, llama_gnmuon_audit, or "
                "llama_checkpoint_pool_selector, llama_proxy_falsification_grid, or provide "
                "an explicit non-legacy model id that the harness supports."
            )


def normalize_kind(kind: Any) -> str:
    k = str(kind or "").strip().lower().replace("-", "_")
    k = KIND_ALIASES.get(k, k)
    if k not in KIND_SCRIPTS:
        # НЕ отвергаем незнакомый тип: заранее не знаем, какие экспы понадобятся. Любой такой →
        # кастомный llama-эксп, скрипт которого экспериментатор пишет сам под claim (на модели llama).
        return "custom_llama"
    return k


def _clean_text(x: Any, limit: int) -> str:
    s = str(x or "").replace("\n", " ").strip()
    return re.sub(r"\s+", " ", s)[:limit]


def _has_stale_rho_ci_text(text: str) -> bool:
    return bool(re.search(
        r"bootstrap|confidence[- ]?interval|CI∌0|endpoints?",
        text or "",
        re.IGNORECASE,
    ))


def _clean_claim(kind: str, raw_claim: Any, context: str, limit: int = 700) -> str:
    claim = _clean_text(raw_claim or context, 5000)
    if kind != "rho_predictor":
        return claim[:limit]

    # rho_predictor confirms by its typed rule, not by old reviewer snippets.
    # If claim was built from node context, keep the scientific prefix and drop
    # embedded gate feedback such as "... gate=remote: [gate ... bootstrap CI ...".
    claim = re.split(
        r"\s+(?:gate|last_gate)\s*=\s*(?:remote:\s*)*(?:\[gate\b|remote:)",
        claim,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    if _has_stale_rho_ci_text(claim):
        claim = re.sub(
            r"[^.;|]*?(?:bootstrap|confidence[- ]?interval|CI∌0|endpoints?)[^.;|]*[.;|]?",
            " ",
            claim,
            flags=re.IGNORECASE,
        ).strip()
    if len(claim) < 40:
        claim = (
            "Measure the source Sigma_delta low-spectrum rho profile of the initial "
            "target gradient on the configured llama-style model and check dimension "
            "validity, finite rho_grad_weighted, and a non-degenerate rho profile."
        )
    return _clean_text(claim, limit)


def clean_env(env: Any) -> Dict[str, str]:
    if not isinstance(env, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in env.items():
        kk = str(k).strip().upper()
        vv = str(v).strip()
        if not _ENV_KEY_RE.match(kk):
            continue
        if not _ENV_VAL_RE.match(vv):
            continue
        out[kk] = vv
    return out


def _safe_script(path: Any) -> str:
    p = str(path or "").strip()
    if not p:
        return ""
    p = posixpath.normpath(p)
    if p.startswith("../") or p.startswith("/") or p == "..":
        raise ValueError(f"unsafe EXP_SPEC.script={path!r}")
    if p not in set(KIND_SCRIPTS.values()):
        raise ValueError(f"script is not allowlisted: {p!r}")
    return p


def canonicalize(raw: Any, context: str = "", exp_kind: str = "") -> Dict[str, Any]:
    """Return a validated EXP_SPEC dict.

    Missing specs are inferred conservatively from context.  Unknown kinds are
    rejected instead of falling back to an unrelated experiment.
    """
    if isinstance(raw, str):
        raw = raw.strip()
        if raw:
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raw = {"kind": infer_kind(context + " " + raw), "claim": raw}
        else:
            raw = {}
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    kind = normalize_kind(raw.get("kind") or infer_kind(context))
    if kind == "custom_llama":
        # уникальный путь скрипта по хэшу claim → разные кастомные экспы не перетирают друг друга
        _h = hashlib.sha1(_clean_text(raw.get("claim") or context, 5000).encode("utf-8")).hexdigest()[:10]
        script = f"Results/scripts/exp_custom_{_h}.py"
    else:
        script = _safe_script(raw.get("script") or KIND_SCRIPTS[kind])
    markers = (["FINAL", "C1_DONE"] if kind == "c1_retention"
               else ["FINAL", "RHO_DONE"] if kind == "rho_predictor"
               else ["RESULT_JSON", "EMP_VERDICT", "EXP_DONE"])
    default_rule = (
        "router confirms iff FINAL.C1.excl0 is true and FINAL.C1.mean > 0"
        if kind == "c1_retention"
        else ("router confirms iff FINAL.primary_confirmed is true: positive Delta_C1 "
              "bootstrap CI above zero plus target non-inferiority on the llama-style audit")
        if kind == "llama_gnmuon_audit"
        else ("router confirms iff FINAL.proxy_primary_confirmed is true: the llama-style "
              "method grid ran and the GN/LMO proxy score has a Spearman bootstrap CI "
              "excluding zero against held-out Delta_C1")
        if kind == "llama_proxy_falsification_grid"
        else ("router confirms iff FINAL.fixed_baseline_primary_confirmed is true: "
              "each fixed baseline requested by the critic has a seed-level Delta_C1 "
              "bootstrap CI and target/wall-clock reporting")
        if kind == "llama_fixed_baseline_audit"
        else ("router confirms iff FINAL.frontier_primary_confirmed is true: sign-paired "
              "source tasks share one target transcript, target-only rules fail to certify "
              "the source sign, the source-bit/probe rule meets coverage with bounded error, "
              "and paired CRN bootstrap CIs with target non-inferiority are reported")
        if kind == "llama_source_bit_frontier_audit"
        else ("router confirms iff FINAL.calibration_primary_confirmed is true: "
              "prospective C1 firewall has low false-accept bootstrap CI and "
              "non-inferior target loss")
        if kind == "llama_firewall_calibration_audit"
        else ("router confirms iff FINAL.alignment_primary_confirmed is true: "
              "high pre-finetune source-target gradient alignment has lower "
              "held-out C1 forgetting than target-loss-matched low-alignment "
              "checkpoints with bootstrap CI lower bound above zero")
        if kind == "llama_alignment_bridge_audit"
        else ("router confirms iff FINAL.primary_confirmed is true: C1-selector beats "
              "loss/time/GN-proxy checkpoint selectors on held-out Delta_C1 with every "
              "pairwise bootstrap CI lower bound above zero and target non-inferiority")
        if kind == "llama_checkpoint_pool_selector"
        else ("router confirms iff FINAL is dimension-valid, non-degenerate, and "
              "rho_grad_weighted is finite")
        if kind == "rho_predictor"
        else "trust only EMP_VERDICT: confirmed with RESULT_JSON matching this EXP_SPEC"
    )
    model = normalize_model(raw.get("model"))
    if kind == "legacy_llama_forgetting" and not raw.get("model"):
        model = "llama"
    spec = {
        "kind": kind,
        "script": script,
        "model": model,
        "claim": _clean_claim(kind, raw.get("claim"), context, 700),
        "success_metric": (
            "" if kind == "rho_predictor" and _has_stale_rho_ci_text(str(raw.get("success_metric") or ""))
            else _clean_text(raw.get("success_metric"), 300)
        ),
        "confirm_rule": _clean_text(
            default_rule
            if kind == "rho_predictor" and _has_stale_rho_ci_text(str(raw.get("confirm_rule") or ""))
            else (raw.get("confirm_rule") or default_rule),
            300,
        ),
        "env": clean_env(raw.get("env")),
        "required_markers": markers,
    }
    _enforce_model_contract(spec, context, exp_kind)
    return spec


def dump(spec: Dict[str, Any]) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def infer_kind(text: str) -> str:
    t = (text or "").lower()
    # ЯВНО НАЗВАННЫЙ kind ВАЖНЕЕ эвристик: критик пишет имя вида `llama_..._audit` прямо в тексте,
    # а keyword-матчинг ниже мог свести его к ДРУГОМУ (в т.ч. опровергнутому) kind → cornerstone
    # игнорировался как duplicate-refuted и решающий эксп не сеялся. Незнакомое имя легально:
    # normalize_kind сведёт его в custom_llama → экспериментатор пишет скрипт сам (динамическая
    # регистрация kinds, никакой ручной правки реестра). Приоритет — имени в бэктиках.
    for pat in (r"`(llama_[a-z0-9_]{3,60})`", r"\b(llama_[a-z0-9_]{3,60})\b"):
        mk = re.search(pat, t)
        if mk and mk.group(1) not in ("llama_style", "llama_like", "llama_compatible"):
            return mk.group(1)
    llamaish = any(x in t for x in (
        "llama-style", "llama style", "llama_style", "real-net", "real net",
        "llm-baselines", "llama audit",
    ))
    proxy_gridish = any(x in t for x in (
        "proxy-falsification", "proxy falsification", "spearman", "lmo score",
        "lmo-score", "proxy/lmo", "gn proxy/lmo", "gn-proxy/lmo",
        "projector rotation", "epsilon_out", "ε_out", "gamma_t",
    ))
    method_gridish = any(x in t for x in (
        "full-gn", "full gn", "full_gn", "adamw", "muon",
        "gn-muon", "gnmuon",
    ))
    if llamaish and proxy_gridish and method_gridish:
        return "llama_proxy_falsification_grid"
    fixed_baselineish = any(x in t for x in (
        "fixed-baseline", "fixed baseline", "fixed_baseline", "baseline purity",
        "fixed baseline `b`", "заранее фиксирован", "фиксированный baseline",
        "fixed optimizer", "per-fixed-baseline", "per fixed baseline",
        "gn-muon vs adamw", "vs adamw/full", "adamw/full-gn/muon",
    ))
    if llamaish and fixed_baselineish and method_gridish:
        return "llama_fixed_baseline_audit"
    # source-bit frontier audit — ДО calibration-ветки: его текст («false-accept ceiling» и т.п.)
    # раньше зацеплял calibrationish и маппился в ОПРОВЕРГНУТЫЙ llama_firewall_calibration_audit →
    # cornerstone критика игнорировался как duplicate-refuted и решающий эксп не сеялся вообще.
    source_bitish = any(x in t for x in (
        "source_bit", "source-bit", "source bit", "frontier_audit", "frontier audit",
        "sign-paired source", "sign paired source",
    ))
    if source_bitish:
        return "llama_source_bit_frontier_audit"
    calibrationish = any(x in t for x in (
        "firewall-calibration", "firewall calibration", "false-accept",
        "false accept", "lower-ci", "lower ci", "lower-ci rule",
        "prospective firewall", "preselect checkpoint", "checkpoint×optimizer",
        "checkpoint x optimizer", "checkpoint-optimizer",
    ))
    if llamaish and calibrationish:
        return "llama_firewall_calibration_audit"
    alignment_bridgeish = any(x in t for x in (
        "source-target alignment", "source target alignment", "source/target alignment",
        "source-target bridge", "alignment bridge", "pre-finetune alignment",
        "pre finetune alignment", "target-loss-matched", "target loss matched",
        "alignment variable", "gradient alignment", "held-out delta c1 bucket",
        "bucket gap", "alignment audit",
    ))
    if llamaish and alignment_bridgeish:
        return "llama_alignment_bridge_audit"
    selectorish = any(x in t for x in (
        "checkpoint pool", "checkpoint-pool", "checkpoint selector",
        "checkpoint-selector", "c1-selector", "c1 selector", "loss-selector",
        "loss selector", "time-selector", "time selector", "gn-proxy",
        "gn proxy", "selector firewall", "selection firewall",
        "pool selector", "release selector", "release-selector",
        "селектор", "чекпойнт", "чекпоинт",
    ))
    if llamaish and selectorish:
        return "llama_checkpoint_pool_selector"
    gnmuonish = any(x in t for x in (
        "gn-muon", "gnmuon", "three-factor", "three factor", "delta_c1",
        "Δc1", "positive `δc1", "positive delta", "full gn variant",
    ))
    if llamaish and gnmuonish:
        return "llama_gnmuon_audit"
    if any(x in t for x in (
        "matched-target", "matched target", "target parity", "target ce parity",
        "same target", "same `l_tar", "l_tar <=", "stopping at same",
        "matched-target audit",
    )):
        return "gnmuon_matched_target"
    if any(x in t for x in (
        "wall-clock", "wallclock", "wall clock", "matched-compute", "matched compute",
        "shampoo", "kfac", "baseline-completeness", "baseline completeness",
        "best baseline", "pareto", "gnbaselines",
    )):
        return "gnmuon_wallclock"
    if any(x in t for x in (
        "rho_l", "rho predictor", "rho_predictor", "sigma_delta", "sigma delta",
        "cfisher", "c_fisher", "fisher sensitivity", "фишер-чувствительность",
        "c_ℓ", "c_l", "c ell", "depth profile",
        "профиль глубины", "gn ядра", "gn kernels",
    )):
        return "rho_predictor"
    if any(x in t for x in ("gate", "confidence", "1-p", "1 - p", "softmax gate", "Σ_x^gate", "sigma_x^gate")):
        return "gate_sensitivity"
    if any(x in t for x in ("mechanism", "spectral control", "randomized", "alignment control")):
        return "ce_mechanism"
    if any(x in t for x in ("decomp", "checkpoint", "period-2", "period 2", "box")):
        return "muon_decomp"
    if any(x in t for x in ("forget", "retention", "source loss", "muon", "sgd", "adam", "optimizer")):
        return "real_forgetting"
    return "legacy_llama_forgetting"


def no_exp_needed_invalid_kind(text: str, exp_kind: str = "") -> str:
    """Return the typed kind that must be requested instead of NO_EXP_NEEDED.

    This is intentionally conservative and only covers project-policy real-net
    claims where a deterministic gate reject is better than relying on the LLM
    code judge to notice the missing experiment.
    """
    if str(exp_kind or "").strip().lower() != "llama":
        return ""
    t = (text or "").lower()
    no_new_real_net = any(x in t for x in (
        "без нового real-net claim",
        "без нового real net claim",
        "без нового real-net",
        "без нового real net",
        "no new real-net claim",
        "no new real net claim",
        "no new real-net",
        "no new real net",
        "not a new real-net claim",
        "not a new real net claim",
        "not a new real-net",
        "not a new real net",
        "without new real-net claim",
        "without new real net claim",
        "without a new real-net claim",
        "without a new real net claim",
        "without new real-net",
        "without new real net",
        "without a new real-net",
        "without a new real net",
        "without new real-net/llm claim",
        "without new real net/llm claim",
        "without new real-net/llm",
        "without new real net/llm",
        "без нового real-net/llm claim",
        "без нового real net/llm claim",
        "без нового real-net/llm",
        "без нового real net/llm",
        "новых real-net утверждений не добавлено",
        "новых real net утверждений не добавлено",
        "новых real-net/llm утверждений не добавлено",
        "новых real net/llm утверждений не добавлено",
        "новое real-net утверждение не добавлено",
        "новое real net утверждение не добавлено",
        "не добавляет новых real-net",
        "не добавляет новых real net",
        "does not add new real-net",
        "does not add new real net",
        "no new real-net claims are added",
        "no new real net claims are added",
    ))
    local_negative_pivot = any(x in t for x in (
        "чистая локальная",
        "локальная алгебра",
        "локальная алгебраическая",
        "локальной алгебраической",
        "локальная аналитическая",
        "локальной аналитической",
        "аналитическая теорема",
        "аналитической теоремой",
        "локальная/протокольная",
        "локальной/протокольной",
        "протокольная теорема",
        "протокольной теоремой",
        "протокольной теорем",
        "purely local",
        "local analytic",
        "local analytical",
        "local algebra",
        "local algebraic",
        "analytic theorem",
        "analytical theorem",
        "protocol theorem",
        "settled refutation",
        "formalize/report the settled",
        "negative-pivot",
        "story/scope pivot",
        "story scope pivot",
        "scope pivot",
        "source-free ce identifiability",
        "source free ce identifiability",
        "source-free identifiability",
        "source free identifiability",
        "identifiability result",
        "identifiability theorem",
        "no-certificate result",
        "no certificate result",
        "certificate limitation",
    ))
    if no_new_real_net and local_negative_pivot:
        return ""
    kind = infer_kind(text)
    if kind in {
        "llama_gnmuon_audit",
        "llama_checkpoint_pool_selector",
        "llama_firewall_calibration_audit",
        "llama_alignment_bridge_audit",
        "llama_proxy_falsification_grid",
        "llama_fixed_baseline_audit",
    }:
        return kind
    return ""


def _balanced_object_at(text: str, start: int) -> Optional[str]:
    depth = 0
    in_str = False
    esc = False
    begin = -1
    for i in range(start, len(text)):
        ch = text[i]
        if begin < 0:
            if ch == "{":
                begin = i
                depth = 1
            continue
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[begin:i + 1]
    return None


def extract(text: str) -> Optional[Dict[str, Any]]:
    """Extract EXP_SPEC JSON from model/gate output."""
    if not text:
        return None
    raw = None
    for marker in ("EXP_SPEC_REQUEST:", "EXP_SPEC:"):
        idx = text.find(marker)
        if idx >= 0:
            raw = _balanced_object_at(text, idx + len(marker))
            if raw is not None:
                break
    if raw is None:
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        raw = m.group(1) if m else None
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None
