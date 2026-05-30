"""
backend/services/insight_service.py
─────────────────────────────────────
LLM-generated human-like insights. Builds a structured factual brief from
the user's real data (profile + recent activity + ML predictions + anomalies
+ cohort comparison) and asks the chatbot to generate a JSON list of short,
specific insights.

If no LLM key is configured the service emits one rule-based insight per
detected anomaly and per major data fact — never fabricated, always grounded
in numbers passed in.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Optional

from loguru import logger

from backend.services.chatbot_service import get_chatbot_service


_INSIGHT_PROMPT_HEADER = (
    "You are a concise, science-based fitness coach. Read the user dossier "
    "below and produce 3 to 5 brief, *highly specific* insights. Each insight "
    "must reference a real number from the dossier — no generic advice.\n\n"
    "Output ONLY a JSON array of objects with this schema:\n"
    "  {\"category\": one of [\"PROGRESS\", \"RISK\", \"SUGGESTION\", \"TREND\", \"COMPARISON\"],\n"
    "   \"icon\": one short emoji,\n"
    "   \"text\": insight under 25 words,\n"
    "   \"severity\": one of [\"info\", \"warn\", \"alert\"]}\n\n"
    "Do not wrap in markdown code fences. Output the raw JSON array.\n\n"
)


class InsightService:

    @staticmethod
    def _build_dossier(brief: dict) -> str:
        """Formats the structured input into a clear text block for the LLM."""
        lines = []
        p = brief.get("profile") or {}
        if p:
            lines.append(
                f"Profile: {p.get('age')}y {p.get('gender','?')}, "
                f"{p.get('weight_kg')}kg → goal {p.get('target_weight','?')}kg "
                f"({p.get('goal','maintain')}), height {p.get('height_cm')}cm, "
                f"activity level {p.get('activity_level')}"
            )
        kpi = brief.get("kpi") or {}
        if kpi:
            lines.append(
                f"Computed: BMI={kpi.get('bmi')}, TDEE={kpi.get('tdee')} kcal, "
                f"daily target={kpi.get('calories_goal')} kcal, "
                f"streak={kpi.get('streak_days')} days, "
                f"workouts in last 30d={kpi.get('workouts_30d')}, "
                f"avg form={kpi.get('avg_form_score')}, "
                f"consistency={kpi.get('consistency')}, "
                f"weight trend (30d)={kpi.get('weight_trend_30d')}kg"
            )
        ml = brief.get("ml_predictions") or {}
        if ml:
            lines.append(
                f"ML predictions (NOTE: the raw 30-day weight-change figure below is "
                f"goal-blind — it ignores the user's goal/calorie target, so do NOT "
                f"quote it as their expected result; use the goal-aware blended "
                f"forecast trend instead): raw 30-day weight change="
                f"{ml.get('weight_change_30d_kg')}kg, "
                f"fitness level={ml.get('fitness_level')} "
                f"(model={ml.get('fitness_source','model')})"
            )
        f = brief.get("forecast") or {}
        if f.get("available"):
            tr_day = f.get("trend_kg_per_day")
            tr_30 = f.get("trend_kg_per_30d")
            if tr_30 is None and tr_day is not None:
                tr_30 = round(tr_day * 30, 2)
            lines.append(
                f"AUTHORITATIVE goal-aware forecast (use THIS for any weight-trend "
                f"statement): trend={tr_day} kg/day (~{tr_30} kg over 30 days), "
                f"per-model breakdown={f.get('models_kg_change_30d')}, "
                f"stability={f.get('stability')}"
            )
        tt = brief.get("timeline_to_goal") or {}
        if tt.get("available"):
            lines.append(
                f"Timeline to target: ~{tt.get('days_to_target')} days "
                f"(target date {tt.get('target_date')}, feasibility={tt.get('feasibility')})"
            )
        ah = brief.get("anomalies") or []
        if ah:
            lines.append("Anomalies:")
            for a in ah[:4]:
                lines.append(f"  • [{a.get('severity')}] {a.get('title')} — {a.get('detail')}")
        co = brief.get("cohort") or {}
        if co.get("available"):
            v = co.get("you_vs_cohort", {})
            lines.append(
                f"Cohort ({co.get('cohort_size')} peers, same goal & age band): "
                f"workouts/wk you={v.get('workouts_per_week',{}).get('you')} vs "
                f"avg={v.get('workouts_per_week',{}).get('cohort_avg')} "
                f"(percentile {v.get('workouts_per_week',{}).get('your_percentile')}), "
                f"weight move 30d you={v.get('weight_change_30d_kg',{}).get('you')} vs "
                f"avg={v.get('weight_change_30d_kg',{}).get('cohort_avg')}"
            )
        return "\n".join(lines)

    @staticmethod
    def _extract_json_array(text: str) -> list[dict]:
        """Best-effort: find a JSON array in the LLM's reply."""
        if not text:
            return []
        # Strip code fences
        text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.MULTILINE)
        text = re.sub(r"```$", "", text.strip(), flags=re.MULTILINE)
        # Find first '[' ... last ']'
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return []
        try:
            arr = json.loads(text[start : end + 1])
            if isinstance(arr, list):
                return [x for x in arr if isinstance(x, dict)]
        except Exception:
            return []
        return []

    @staticmethod
    def _fallback(brief: dict) -> list[dict]:
        """Rule-based insights when the LLM is unavailable. Numbers are real."""
        out: list[dict] = []
        kpi = brief.get("kpi") or {}
        ml  = brief.get("ml_predictions") or {}
        ah  = brief.get("anomalies") or []
        co  = brief.get("cohort") or {}
        fc  = brief.get("forecast") or {}
        prof = brief.get("profile") or {}

        if kpi.get("streak_days", 0) >= 5:
            out.append({"category": "PROGRESS", "icon": "🔥", "severity": "info",
                        "text": f"You're on a {kpi['streak_days']}-day workout streak. Keep it alive."})
        # TREND insight — use the GOAL-AWARE blended forecast trend, not the
        # raw ML number. The ML weight-change model is goal-blind (it never sees
        # the user's goal), so on its own it can claim a cutter will "gain", which
        # contradicts both their plan and the forecast graph. forecast.trend_*
        # already blends ML + energy-balance + observed data and is direction-
        # clamped to the user's goal (see forecast_service.weight_curve), so it's
        # the single source of truth the rest of the dashboard already shows.
        trend_30d = fc.get("trend_kg_per_30d")
        if trend_30d is None and fc.get("trend_kg_per_day") is not None:
            trend_30d = round(fc["trend_kg_per_day"] * 30, 2)
        # Fall back to the raw ML number only if the forecast is unavailable.
        if trend_30d is None:
            trend_30d = ml.get("weight_change_30d_kg")

        if trend_30d is not None:
            d = trend_30d
            goal = (prof.get("goal") or "").lower()
            losing  = goal in ("lose", "cut", "fat_loss", "fat loss")
            gaining = goal in ("gain", "bulk", "build", "muscle_gain")
            if abs(d) < 0.1:
                # Essentially flat — phrase it as maintenance, not a tiny gain/loss.
                text = "Your weight is projected to hold steady over the next 30 days at this pace."
                icon = "➡️"
            else:
                arrow = "lose" if d < 0 else "gain"
                text = f"At your current pace, you're on track to {arrow} {abs(d):.1f}kg in the next 30 days."
                icon = "📉" if d < 0 else "📈"
                # Add a one-line note when the projection aligns with the stated goal.
                if (losing and d < 0) or (gaining and d > 0):
                    text += " That's in line with your goal — keep it up."
            out.append({"category": "TREND", "icon": icon, "severity": "info", "text": text})
        for a in ah[:3]:
            out.append({"category": "RISK" if a["severity"] == "alert" else "SUGGESTION",
                        "icon": "⚠️" if a["severity"] != "info" else "💡",
                        "severity": a["severity"], "text": a["title"]})
        c_wpw = co.get("you_vs_cohort", {}).get("workouts_per_week", {}) if co.get("available") else {}
        if c_wpw.get("your_percentile") is not None:
            pct = c_wpw["your_percentile"]
            if pct >= 75:
                out.append({"category": "COMPARISON", "icon": "🏆", "severity": "info",
                            "text": f"You train more than {pct:.0f}% of similar users (same goal & age)."})
            elif pct < 25:
                out.append({"category": "COMPARISON", "icon": "📊", "severity": "warn",
                            "text": f"You train less than {100-pct:.0f}% of similar users — small bumps add up."})
        return out[:5]

    async def generate(self, brief: dict, *, max_insights: int = 5) -> list[dict]:
        dossier = self._build_dossier(brief)
        if not dossier.strip():
            return []

        chat = get_chatbot_service()
        engine_name = chat.engine.name
        if engine_name == "none":
            logger.debug("InsightService: no LLM engine, returning rule-based fallbacks")
            return self._fallback(brief)

        prompt = _INSIGHT_PROMPT_HEADER + "Dossier:\n" + dossier
        try:
            # Use the chatbot service's plain chat with no profile / no user_id
            # Special user_id "_insights" so it doesn't pollute real session memory.
            res = await chat.chat(user_id="_insights", message=prompt, profile={})
            reply = res.get("reply", "")
            arr = self._extract_json_array(reply)
            if not arr:
                logger.warning("InsightService: LLM did not return parseable JSON, falling back")
                return self._fallback(brief)
            # Sanitise output
            clean = []
            for item in arr[:max_insights]:
                clean.append({
                    "category": str(item.get("category", "SUGGESTION"))[:20].upper(),
                    "icon":     str(item.get("icon", "💡"))[:4],
                    "text":     str(item.get("text", "")).strip()[:200],
                    "severity": str(item.get("severity", "info")).lower(),
                })
            return [c for c in clean if c["text"]]
        except Exception as e:
            logger.warning(f"InsightService LLM call failed: {e}")
            return self._fallback(brief)


_singleton: Optional[InsightService] = None


def get_insight_service() -> InsightService:
    global _singleton
    if _singleton is None:
        _singleton = InsightService()
    return _singleton
