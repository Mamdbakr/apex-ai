from __future__ import annotations

from typing import Any, Dict, Literal

from langchain_core.tools import tool

ActivityLevel = Literal["sedentary", "light", "moderate", "active", "very_active"]
GoalType = Literal["cut", "bulk", "maintain"]


@tool
def calculate_bmr(weight: float, height: float, age: int, gender: Literal["male", "female"]) -> Dict[str, Any]:
    """Calculate BMR using Mifflin-St Jeor. weight in kg, height in cm."""
    base = 10 * weight + 6.25 * height - 5 * age
    bmr = base + 5 if gender == "male" else base - 161
    return {"bmr": round(bmr, 2), "formula": "Mifflin-St Jeor"}


@tool
def calculate_tdee(bmr: float, activity_level: ActivityLevel) -> Dict[str, Any]:
    """Calculate TDEE from BMR and activity level multiplier."""
    multipliers = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very_active": 1.9,
    }
    mult = multipliers.get(activity_level, 1.55)
    return {"tdee": round(bmr * mult, 2), "multiplier": mult}


@tool
def calculate_macros(weight: float, calories_goal: float) -> Dict[str, Any]:
    """Calculate macros. Protein 2.0g/kg, fats 0.8g/kg, carbs = remaining calories."""
    protein_g = round(weight * 2.0)
    fats_g = round(weight * 0.8)
    protein_cal = protein_g * 4
    fat_cal = fats_g * 9
    carbs_g = max(round((calories_goal - protein_cal - fat_cal) / 4), 0)
    return {
        "calories": int(calories_goal),
        "protein_g": protein_g,
        "carbs_g": carbs_g,
        "fats_g": fats_g,
    }


@tool
def calculate_1rm(weight: float, reps: int) -> Dict[str, Any]:
    """Estimate one-rep max with Epley formula."""
    safe_reps = max(reps, 1)
    one_rm = weight * (1 + safe_reps / 30.0)
    return {"one_rm": round(one_rm, 2), "formula": "Epley"}


@tool
def hydration_needs(weight: float) -> Dict[str, Any]:
    """Estimate hydration needs in liters/day (35 ml/kg)."""
    return {"water_liters_per_day": round(weight * 0.035, 2)}


TOOL_REGISTRY = [
    calculate_bmr,
    calculate_tdee,
    calculate_macros,
    calculate_1rm,
    hydration_needs,
]
