from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from backend.rag_coach.tools import (
    TOOL_REGISTRY,
    calculate_bmr,
    calculate_macros,
    calculate_tdee,
    hydration_needs,
)
from backend.rag_coach.rag import RAGService


class GraphState(TypedDict, total=False):
    message: str
    profile: Dict[str, Any]
    intent: str
    missing_fields: List[str]
    calculations: Dict[str, Any]
    retrieved: List[Dict[str, Any]]
    plan: Dict[str, List[str]]
    final_payload: Dict[str, Any]


class ChatPayload(BaseModel):
    response: str
    calories: Optional[int] = None
    macros: Optional[Dict[str, int]] = None
    workout_plan: List[str] = Field(default_factory=list)
    meal_plan: List[str] = Field(default_factory=list)


load_dotenv()


def get_llm():
    provider = os.getenv("RAG_LLM_PROVIDER", "groq").lower()
    if provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set (required for RAG_LLM_PROVIDER=gemini)")
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            google_api_key=api_key,
            temperature=0.2,
        )
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set (required for RAG_LLM_PROVIDER=groq)")
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        api_key=api_key,
        temperature=0.2,
    )


class FitnessGraph:
    def __init__(self, rag: RAGService) -> None:
        self.rag = rag
        self._llm = None
        self.graph = self._build()

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    def _intent_node(self, state: GraphState) -> GraphState:
        text = state["message"].lower()
        if any(token in text for token in ["macro", "calorie", "bmr", "tdee", "1rm", "hydration"]):
            intent = "calculation"
        elif any(token in text for token in ["meal", "nutrition", "diet", "food"]):
            intent = "nutrition"
        elif any(token in text for token in ["workout", "program", "training", "split", "exercise"]):
            intent = "workout"
        else:
            intent = "general"
        return {"intent": intent}

    def _calculator_node(self, state: GraphState) -> GraphState:
        profile = state.get("profile", {})
        missing: List[str] = []
        calculations: Dict[str, Any] = {}

        needed = ["age", "weight", "height", "gender", "goal", "activity_level"]
        if state.get("intent") in ("calculation", "nutrition", "workout"):
            for field in needed:
                if profile.get(field) in (None, ""):
                    missing.append(field)

        if not missing:
            gender = "male" if str(profile["gender"]).lower().startswith("m") else "female"
            bmr = calculate_bmr.invoke(
                {
                    "weight": float(profile["weight"]),
                    "height": float(profile["height"]),
                    "age": int(profile["age"]),
                    "gender": gender,
                }
            )["bmr"]
            _activity_map = {
                "1": "sedentary", "2": "light", "3": "moderate", "4": "active", "5": "very_active",
                "sedentary": "sedentary", "light": "light", "moderate": "moderate",
                "active": "active", "very_active": "very_active",
            }
            raw_activity = str(profile["activity_level"]).lower().strip()
            activity_level = _activity_map.get(raw_activity, "moderate")
            tdee = calculate_tdee.invoke(
                {
                    "bmr": bmr,
                    "activity_level": activity_level,
                }
            )["tdee"]

            goal = str(profile["goal"]).lower()
            if goal in ("bulk", "muscle gain", "gain"):
                calories = tdee + 300
            elif goal in ("cut", "fat loss", "lose"):
                calories = tdee - 500
            else:
                calories = tdee

            macros = calculate_macros.invoke(
                {
                    "weight": float(profile["weight"]),
                    "calories_goal": float(calories),
                }
            )
            hydration = hydration_needs.invoke({"weight": float(profile["weight"])})
            calculations = {
                "bmr": round(float(bmr), 2),
                "tdee": round(float(tdee), 2),
                "calories": int(calories),
                "macros": {
                    "protein_g": int(macros["protein_g"]),
                    "carbs_g": int(macros["carbs_g"]),
                    "fats_g": int(macros["fats_g"]),
                },
                "hydration": hydration,
            }

        return {"missing_fields": missing, "calculations": calculations}

    def _retriever_node(self, state: GraphState) -> GraphState:
        return {"retrieved": self.rag.search(state["message"], top_k=4)}

    def _planner_node(self, state: GraphState) -> GraphState:
        plan = {"workout_plan": [], "meal_plan": []}
        intent = state.get("intent")
        goal = str(state.get("profile", {}).get("goal", "maintain")).lower()

        if intent == "workout":
            if "bulk" in goal or "gain" in goal:
                plan["workout_plan"] = [
                    "Day 1 Push: bench, incline press, shoulder press, triceps",
                    "Day 2 Pull: deadlift, row, pull-up, biceps",
                    "Day 3 Legs: squat, RDL, lunges, calves",
                    "Day 4 Rest and mobility",
                    "Day 5 Upper hypertrophy",
                    "Day 6 Lower hypertrophy and core",
                    "Day 7 Rest",
                ]
            else:
                plan["workout_plan"] = [
                    "3 full-body resistance sessions per week",
                    "2 cardio sessions in zone 2 or intervals",
                    "1 mobility day and 1 full rest day",
                ]

        if intent == "nutrition":
            plan["meal_plan"] = [
                "Meal 1: high-protein breakfast plus complex carbs",
                "Meal 2: lean protein, rice or potatoes, vegetables",
                "Meal 3: protein snack and fruit",
                "Meal 4: protein-focused dinner with healthy fats",
            ]

        return {"plan": plan}

    def _final_node(self, state: GraphState) -> GraphState:
        missing = state.get("missing_fields", [])
        if missing:
            return {
                "final_payload": {
                    "response": "I need these fields to personalize your plan: " + ", ".join(missing),
                    "calories": None,
                    "macros": None,
                    "workout_plan": [],
                    "meal_plan": [],
                }
            }

        profile = state.get("profile", {})
        calculations = state.get("calculations", {})
        plan = state.get("plan", {"workout_plan": [], "meal_plan": []})
        retrieved = state.get("retrieved", [])

        evidence = "\n\n".join(
            [f"Source: {item['source']}\nContent: {str(item['text'])[:1100]}" for item in retrieved]
        )

        system = (
            "You are a science-based fitness and nutrition coach. "
            "Use ACSM, ISSN, and NSCA style recommendations. "
            "Do not invent numeric targets when calculations are available. "
            "Always personalize based on profile and provide practical steps."
        )

        bound_llm = self.llm.bind_tools(TOOL_REGISTRY)
        msgs = [
            SystemMessage(content=system),
            HumanMessage(
                content=(
                    f"User profile: {profile}\n"
                    f"Intent: {state.get('intent')}\n"
                    f"Calculated values: {calculations}\n"
                    f"Plan scaffold: {plan}\n"
                    f"RAG evidence: {evidence}\n\n"
                    f"User question: {state['message']}"
                )
            ),
        ]

        first = bound_llm.invoke(msgs)
        tool_messages: List[ToolMessage] = []
        tool_map = {tool.name: tool for tool in TOOL_REGISTRY}

        if getattr(first, "tool_calls", None):
            for call in first.tool_calls:
                tool_name = call.get("name")
                args = call.get("args", {})
                if tool_name in tool_map:
                    result = tool_map[tool_name].invoke(args)
                    tool_messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

        response = self.llm.invoke(msgs + [first] + tool_messages)
        response_text = getattr(response, "content", str(response))

        final_payload = ChatPayload(
            response=response_text,
            calories=calculations.get("calories"),
            macros=calculations.get("macros"),
            workout_plan=plan.get("workout_plan", []),
            meal_plan=plan.get("meal_plan", []),
        ).model_dump()
        return {"final_payload": final_payload}

    def _build(self):
        graph = StateGraph(GraphState)
        graph.add_node("intent", self._intent_node)
        graph.add_node("calculator", self._calculator_node)
        graph.add_node("retriever", self._retriever_node)
        graph.add_node("planner", self._planner_node)
        graph.add_node("final", self._final_node)

        graph.set_entry_point("intent")
        graph.add_edge("intent", "calculator")
        graph.add_edge("calculator", "retriever")
        graph.add_edge("retriever", "planner")
        graph.add_edge("planner", "final")
        graph.add_edge("final", END)
        return graph.compile()

    def invoke(self, message: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        state: GraphState = {
            "message": message,
            "profile": profile,
        }
        result = self.graph.invoke(state)
        return result.get("final_payload", {})
