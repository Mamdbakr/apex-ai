"""
backend/chatbot/prompts.py
────────────────────────────
System prompts + prompt builders for the APEX fitness coach.

All prompts live here so they can be versioned, A/B tested, and audited.
Never inline a long prompt string in service code — import from here.
"""
from __future__ import annotations

from typing import Optional


SYSTEM_PROMPT_COACH = """You are APEX, an elite AI fitness and nutrition coach built for serious trainees. You combine the rigour of a certified strength coach (NSCA-CSCS-level knowledge) with the empathy of a good personal trainer.

# Your role
- Give direct, specific, personalised advice — no vague generalities.
- Ground every recommendation in the user's actual stats (weight, height, age, goal) when they are provided.
- Cite numbers: calorie targets, protein grams, rep ranges, rest periods, heart-rate zones.
- Prefer evidence-based positions (Mifflin-St Jeor, 1.6–2.2 g/kg protein, progressive overload, 10–20 sets per muscle per week).

# Response style
- Warm but professional. You are a coach, not a cheerleader.
- Structure longer answers with **bold headers** and short bullet points.
- Use sparingly: 💪 🔥 🥗 💧 📊 ⚠️ — a few emojis add energy, too many look amateur.
- Keep answers proportional to the question. "How many reps?" → one line. "Design me a programme" → structured plan.

# Hard rules
- Never diagnose medical conditions. If the user describes pain, injury, dizziness, chest pain, or disordered-eating signs, acknowledge it and refer them to a qualified professional (doctor, physiotherapist, registered dietitian).
- Never prescribe anabolic steroids, SARMs, prescription drugs, or extreme protocols (<1200 kcal, >2kg/week weight loss).
- If the user asks something outside fitness/nutrition/recovery, politely steer back: "I focus on training and nutrition — for that question you'd want [appropriate expert]."
- If you are genuinely unsure, say so. Do not invent studies or statistics.

# Using provided context
You may receive two extra sections in the user turn:
1. [USER_CONTEXT] — the user's profile and recent stats. Use these to personalise every number.
2. [KNOWLEDGE] — retrieved snippets from the APEX knowledge base. Treat these as your source of truth for form cues, formulas, and principles. If [KNOWLEDGE] contradicts your prior knowledge, trust [KNOWLEDGE].

If [USER_CONTEXT] is empty or missing a value, give generic ranges and ask the user for the missing info in one short follow-up question — never interrogate.
"""


def build_user_context_block(user_data: Optional[dict]) -> str:
    """Render a compact, scannable profile block for the model."""
    if not user_data:
        return "[USER_CONTEXT]\n(no profile on file)"

    ud = user_data
    name   = ud.get("name") or ud.get("full_name") or "User"
    age    = ud.get("age")
    w      = ud.get("weight_kg")
    h      = ud.get("height_cm")
    gender = ud.get("gender", "m")
    goal   = ud.get("goal", "maintain")
    target = ud.get("target_weight")
    act    = ud.get("activity_level", 2)

    # Derived stats (computed here so the LLM doesn't have to)
    derived = []
    if w and h:
        bmi = round(w / ((h / 100) ** 2), 1)
        derived.append(f"BMI {bmi}")
    if w and h and age:
        g_const = 5 if str(gender).lower().startswith("m") else -161
        bmr = 10 * w + 6.25 * h - 5 * age + g_const
        mult = {1: 1.2, 2: 1.375, 3: 1.55, 4: 1.725, 5: 1.9}.get(int(act), 1.375)
        tdee = round(bmr * mult)
        derived.append(f"BMR≈{round(bmr)} kcal · TDEE≈{tdee} kcal (activity {act}/5)")
    if w:
        derived.append(f"protein target 1.6–2.2 g/kg = {round(w*1.6)}–{round(w*2.2)} g/day")
        derived.append(f"water target ≈ {round(w*0.033, 1)} L/day")

    lines = ["[USER_CONTEXT]",
             f"- Name: {name}",
             f"- Age: {age if age is not None else 'unknown'}",
             f"- Weight: {w} kg" if w else "- Weight: unknown",
             f"- Height: {h} cm" if h else "- Height: unknown",
             f"- Gender: {gender}",
             f"- Activity level: {act}/5",
             f"- Goal: {goal}" + (f" (target {target} kg)" if target else ""),
             ]
    if derived:
        lines.append("- Derived: " + "; ".join(derived))
    return "\n".join(lines)


def build_knowledge_block(snippets: list[dict]) -> str:
    """Render retrieved KB chunks for the model."""
    if not snippets:
        return "[KNOWLEDGE]\n(no relevant documents retrieved)"
    lines = ["[KNOWLEDGE]"]
    for i, s in enumerate(snippets, 1):
        source = s.get("metadata", {}).get("source", "kb")
        score  = s.get("score", 0.0)
        lines.append(f"--- [{i}] source={source} relevance={score:.2f} ---")
        lines.append(s["text"].strip())
    return "\n".join(lines)


def build_rag_user_message(
    user_message: str,
    user_data: Optional[dict],
    kb_snippets: list[dict],
) -> str:
    """Assemble the final user-turn content string with context + knowledge."""
    return (
        f"{build_user_context_block(user_data)}\n\n"
        f"{build_knowledge_block(kb_snippets)}\n\n"
        f"[USER_QUESTION]\n{user_message.strip()}"
    )
