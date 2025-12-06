"""
Function calling tools and implementations for PochampsPlayer.
Contains tool definitions and calculation functions for battle analysis.

Tool Philosophy:
================
TEAM PREVIEW PHASE (analysis_tools in pochamps_player.py):
- Heavy data collection: base stats, types, abilities, usage stats
- Results are CACHED in BattleState for battle phase use

BATTLE PHASE (BATTLE_TOOLS here):
- Lightweight, situation-specific queries
- Focus on: damage calc, tera matchup changes, conditional probability updates
- NO basic Pokemon info (already cached from team preview)
"""

import json
from typing import Dict, List, Optional, Any


# =============================================================================
# BATTLE PHASE Tool Definitions
# =============================================================================
# These tools are OPTIMIZED for battle - no redundant basic info requests
# All basic Pokemon data should already be in context from team preview cache

# Responses API format: {"type": "function", "name": "...", "description": "...", "parameters": {...}}
# NOT Chat Completions format: {"type": "function", "function": {"name": "...", ...}}

BATTLE_TOOLS = [
    # =========================================================================
    # 1. Damage Calculation - Most critical battle tool
    # =========================================================================
    {
        "type": "function",
        "name": "calculate_damage",
        "description": "Calculate damage range for a specific attack. Use when you need precise damage numbers to determine if you can KO or survive. Returns min/max damage percentages.",
        "parameters": {
            "type": "object",
            "properties": {
                "attacker": {
                    "type": "string",
                    "description": "Attacking Pokemon name"
                },
                "defender": {
                    "type": "string",
                    "description": "Defending Pokemon name"
                },
                "move": {
                    "type": "string",
                    "description": "Move being used"
                },
                "attacker_tera_type": {
                    "type": "string",
                    "description": "Attacker's Tera type if terastallized (optional)"
                },
                "defender_tera_type": {
                    "type": "string",
                    "description": "Defender's Tera type if terastallized (optional)"
                },
                "field_conditions": {
                    "type": "object",
                    "description": "Current field conditions (optional)",
                    "properties": {
                        "weather": {"type": "string"},
                        "terrain": {"type": "string"},
                        "screens": {"type": "array", "items": {"type": "string"}}
                    }
                }
            },
            "required": ["attacker", "defender", "move"]
        }
    },
    
    # =========================================================================
    # 2. Tera Matchup - Only needed when tera changes situation
    # =========================================================================
    {
        "type": "function",
        "name": "get_tera_matchup",
        "description": "Calculate how terastallization changes type matchup. Use when opponent has terastallized or you're considering tera. Returns new weaknesses/resistances.",
        "parameters": {
            "type": "object",
            "properties": {
                "pokemon": {
                    "type": "string",
                    "description": "Pokemon name"
                },
                "original_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Pokemon's original types before tera"
                },
                "tera_type": {
                    "type": "string",
                    "description": "The Tera type"
                },
                "attacking_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Types you want to check effectiveness against (optional)"
                }
            },
            "required": ["pokemon", "tera_type"]
        }
    },
    
    # =========================================================================
    # 3. Speed Check with Current Modifiers
    # =========================================================================
    {
        "type": "function",
        "name": "check_speed_order",
        "description": "Determine turn order considering CURRENT field/stat modifiers. Use when Tailwind, Trick Room, or stat changes affect speed.",
        "parameters": {
            "type": "object",
            "properties": {
                "pokemon_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of active Pokemon to compare"
                },
                "trick_room": {
                    "type": "boolean",
                    "description": "Is Trick Room active?"
                },
                "tailwind_sides": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Which sides have Tailwind? ['my_side', 'opponent_side']"
                }
            },
            "required": ["pokemon_list"]
        }
    },
    
    # =========================================================================
    # 4. Conditional Stats - After observing item/ability/move
    # =========================================================================
    {
        "type": "function",
        "name": "update_pokemon_prediction",
        "description": "Update predictions after observing new information. Use when you've confirmed item, ability, or specific move usage. Returns revised probability distributions.",
        "parameters": {
            "type": "object",
            "properties": {
                "pokemon": {
                    "type": "string",
                    "description": "Pokemon name"
                },
                "observed": {
                    "type": "object",
                    "description": "What was observed",
                    "properties": {
                        "item": {"type": "string"},
                        "ability": {"type": "string"},
                        "move": {"type": "string"},
                        "tera_type": {"type": "string"},
                        "speed_relation": {
                            "type": "string",
                            "description": "e.g., 'faster_than:rillaboom' or 'slower_than:tornadus'"
                        }
                    }
                }
            },
            "required": ["pokemon", "observed"]
        }
    },
    
    # =========================================================================
    # 5. Query Cached Info - Fallback for context retrieval
    # =========================================================================
    {
        "type": "function",
        "name": "query_cached_info",
        "description": "Retrieve cached Pokemon info from team preview analysis. Use ONLY if specific data isn't in your context. Avoid calling for info already provided.",
        "parameters": {
            "type": "object",
            "properties": {
                "pokemon": {
                    "type": "string",
                    "description": "Pokemon name to query"
                },
                "info_type": {
                    "type": "string",
                    "enum": ["usage_stats", "base_stats", "common_moves", "common_items", "speed_tier"],
                    "description": "Type of information needed"
                }
            },
            "required": ["pokemon", "info_type"]
        }
    }
]


# =============================================================================
# Cached Data Store - Populated by Team Preview, Used by Battle Phase
# =============================================================================

class TeamPreviewCache:
    """
    Stores data collected during team preview for use in battle phase.
    Prevents redundant API calls for basic Pokemon information.
    """
    
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        
    def store(self, pokemon: str, data_type: str, data: Any) -> None:
        """Store data for a Pokemon."""
        pokemon_key = pokemon.lower().replace(" ", "").replace("-", "")
        if pokemon_key not in self._cache:
            self._cache[pokemon_key] = {}
        self._cache[pokemon_key][data_type] = data
        
    def get(self, pokemon: str, data_type: str) -> Optional[Any]:
        """Retrieve cached data for a Pokemon."""
        pokemon_key = pokemon.lower().replace(" ", "").replace("-", "")
        return self._cache.get(pokemon_key, {}).get(data_type)
    
    def get_all(self, pokemon: str) -> Dict[str, Any]:
        """Get all cached data for a Pokemon."""
        pokemon_key = pokemon.lower().replace(" ", "").replace("-", "")
        return self._cache.get(pokemon_key, {})
    
    def has_data(self, pokemon: str) -> bool:
        """Check if we have any cached data for this Pokemon."""
        pokemon_key = pokemon.lower().replace(" ", "").replace("-", "")
        return pokemon_key in self._cache and len(self._cache[pokemon_key]) > 0
    
    def clear(self) -> None:
        """Clear all cached data (for new battle)."""
        self._cache.clear()
        
    def summary(self) -> Dict[str, List[str]]:
        """Get summary of what's cached for each Pokemon."""
        return {
            pokemon: list(data.keys()) 
            for pokemon, data in self._cache.items()
        }


# =============================================================================
# Tool Executor - Implements actual calculations
# =============================================================================

class ToolExecutor:
    """
    Executes function calling tools for battle calculations.
    Optimized for battle phase - uses cached data when available.
    """
    
    def __init__(
        self, 
        game_data: Optional[Dict] = None,
        cache: Optional[TeamPreviewCache] = None
    ):
        """
        Initialize tool executor with game data and optional cache.
        
        :param game_data: Dict containing moves, pokedex, items, abilities data
        :param cache: TeamPreviewCache instance for cached team preview data
        """
        self.game_data = game_data or {}
        self.moves = self.game_data.get("moves", {})
        self.pokedex = self.game_data.get("pokedex", {})
        self.items = self.game_data.get("items", {})
        self.abilities = self.game_data.get("abilities", {})
        self.typechart = self.game_data.get("typechart", {})
        self.cache = cache or TeamPreviewCache()
        
    def set_cache(self, cache: TeamPreviewCache) -> None:
        """Set cache reference for team preview data."""
        self.cache = cache
    
    def execute(self, function_name: str, function_args: Dict) -> Dict:
        """
        Execute a tool function by name.
        
        :param function_name: Name of the function to execute
        :param function_args: Arguments for the function
        :return: Result dictionary
        """
        executors = {
            "calculate_damage": self.calculate_damage,
            "get_tera_matchup": self.get_tera_matchup,
            "check_speed_order": self.check_speed_order,
            "update_pokemon_prediction": self.update_pokemon_prediction,
            "query_cached_info": self.query_cached_info,
        }
        
        if function_name in executors:
            return executors[function_name](**function_args)
        else:
            return {"error": f"Unknown tool: {function_name}"}
    
    # =========================================================================
    # Tool 1: Damage Calculation
    # =========================================================================
    def calculate_damage(
        self, 
        attacker: str, 
        defender: str, 
        move: str,
        attacker_tera_type: Optional[str] = None,
        defender_tera_type: Optional[str] = None,
        field_conditions: Optional[Dict] = None,
        **kwargs
    ) -> Dict:
        """
        Calculate damage for a move with full context.
        
        Damage Formula (Gen 9):
        ((2*L/5 + 2) * Power * Atk/Def / 50 + 2) * Modifiers
        
        Modifiers include: STAB, type effectiveness, weather, terrain, items, abilities
        """
        # Get cached base stats if available
        attacker_stats = self.cache.get(attacker, "base_stats") or {}
        defender_stats = self.cache.get(defender, "base_stats") or {}
        
        # Get move data
        move_key = move.lower().replace(" ", "")
        move_data = self.moves.get(move_key, {})
        
        # Basic damage calculation (simplified - can be expanded)
        base_power = move_data.get("basePower", 0)
        move_type = move_data.get("type", "Normal")
        category = move_data.get("category", "Physical")
        
        # Get attacker's types for STAB check
        attacker_types = []
        if attacker_stats:
            attacker_types = attacker_stats.get("types", [])
        
        # Determine attack/defense stats to use
        atk_stat = "atk" if category == "Physical" else "spa"
        def_stat = "def" if category == "Physical" else "spd"
        
        # Default to estimated stats (can be refined with cache data)
        attack_value = attacker_stats.get(atk_stat, 100)
        defense_value = defender_stats.get(def_stat, 100)
        
        # Calculate type effectiveness
        effectiveness = self._calculate_type_effectiveness(
            move_type, defender, defender_tera_type
        )
        
        # STAB check (including Tera STAB)
        stab = 1.0
        if attacker_tera_type:
            if move_type.lower() == attacker_tera_type.lower():
                stab = 2.0 if move_type.lower() in [t.lower() for t in attacker_types] else 1.5
        elif move_type in attacker_types:
            stab = 1.5
        
        # Simplified damage calc (Level 50)
        level = 50
        if base_power > 0:
            base_damage = ((2 * level / 5 + 2) * base_power * attack_value / defense_value / 50 + 2)
            min_roll = 0.85
            max_roll = 1.0
            
            min_damage = int(base_damage * min_roll * stab * effectiveness)
            max_damage = int(base_damage * max_roll * stab * effectiveness)
        else:
            min_damage = 0
            max_damage = 0
        
        # Estimate HP for percentage calc
        defender_hp = defender_stats.get("hp", 100)
        # Convert to Lv50 HP: ((2*base + IV + EV/4) * L/100) + L + 10
        estimated_hp = int((2 * defender_hp + 31 + 0) * 50 / 100) + 50 + 10
        
        min_percent = round(min_damage / estimated_hp * 100, 1) if estimated_hp > 0 else 0
        max_percent = round(max_damage / estimated_hp * 100, 1) if estimated_hp > 0 else 0
        
        return {
            "attacker": attacker,
            "defender": defender,
            "move": move,
            "move_type": move_type,
            "base_power": base_power,
            "effectiveness": effectiveness,
            "stab": stab,
            "min_damage": min_damage,
            "max_damage": max_damage,
            "min_damage_percent": min_percent,
            "max_damage_percent": max_percent,
            "is_ohko": min_percent >= 100,
            "is_2hko": min_percent >= 50,
            "notes": self._get_damage_notes(effectiveness, stab, field_conditions)
        }
    
    def _calculate_type_effectiveness(
        self, 
        move_type: str, 
        defender: str,
        defender_tera_type: Optional[str] = None
    ) -> float:
        """Calculate type effectiveness multiplier."""
        # Get defender's types
        if defender_tera_type:
            defend_types = [defender_tera_type]
        else:
            defender_data = self.cache.get(defender, "base_stats") or {}
            defend_types = defender_data.get("types", ["Normal"])
        
        effectiveness = 1.0
        move_type_lower = move_type.lower()
        
        for def_type in defend_types:
            def_type_lower = def_type.lower()
            type_chart = self.typechart.get(def_type_lower, {})
            damage_taken = type_chart.get("damageTaken", {})
            
            # 0 = normal, 1 = super effective (2x), 2 = not very effective (0.5x), 3 = immune (0x)
            eff_value = damage_taken.get(move_type_lower, 0)
            if eff_value == 1:
                effectiveness *= 2.0
            elif eff_value == 2:
                effectiveness *= 0.5
            elif eff_value == 3:
                effectiveness *= 0.0
        
        return effectiveness
    
    def _get_damage_notes(
        self, 
        effectiveness: float, 
        stab: float,
        field_conditions: Optional[Dict]
    ) -> List[str]:
        """Generate human-readable notes about damage calculation."""
        notes = []
        
        if effectiveness >= 2.0:
            notes.append(f"Super effective ({effectiveness}x)")
        elif effectiveness == 0:
            notes.append("Immune (0x)")
        elif effectiveness < 1.0:
            notes.append(f"Not very effective ({effectiveness}x)")
        
        if stab >= 2.0:
            notes.append("Tera STAB (2x)")
        elif stab > 1.0:
            notes.append("STAB (1.5x)")
        
        if field_conditions:
            if field_conditions.get("weather"):
                notes.append(f"Weather: {field_conditions['weather']}")
            if field_conditions.get("terrain"):
                notes.append(f"Terrain: {field_conditions['terrain']}")
        
        return notes
    
    # =========================================================================
    # Tool 2: Tera Matchup
    # =========================================================================
    def get_tera_matchup(
        self,
        pokemon: str,
        tera_type: str,
        original_types: Optional[List[str]] = None,
        attacking_types: Optional[List[str]] = None,
        **kwargs
    ) -> Dict:
        """
        Calculate how Terastallization changes type matchup.
        """
        # Get original types from cache if not provided
        if not original_types:
            pokemon_data = self.cache.get(pokemon, "base_stats") or {}
            original_types = pokemon_data.get("types", ["Normal"])
        
        # Calculate weaknesses/resistances for original types
        original_matchup = self._calculate_defensive_matchup(original_types)
        
        # Calculate for tera type
        tera_matchup = self._calculate_defensive_matchup([tera_type])
        
        # Compare changes
        changes = {
            "new_weaknesses": [],
            "new_resistances": [],
            "lost_weaknesses": [],
            "lost_resistances": [],
            "new_immunities": [],
            "lost_immunities": []
        }
        
        # Find differences
        for type_name, orig_eff in original_matchup.items():
            tera_eff = tera_matchup.get(type_name, 1.0)
            
            if orig_eff > 1.0 and tera_eff <= 1.0:
                changes["lost_weaknesses"].append(type_name)
            elif orig_eff <= 1.0 and tera_eff > 1.0:
                changes["new_weaknesses"].append(type_name)
            
            if orig_eff < 1.0 and orig_eff > 0 and tera_eff >= 1.0:
                changes["lost_resistances"].append(type_name)
            elif orig_eff >= 1.0 and tera_eff < 1.0 and tera_eff > 0:
                changes["new_resistances"].append(type_name)
            
            if orig_eff == 0 and tera_eff > 0:
                changes["lost_immunities"].append(type_name)
            elif orig_eff > 0 and tera_eff == 0:
                changes["new_immunities"].append(type_name)
        
        # Check specific attacking types if requested
        matchup_vs_attacks = {}
        if attacking_types:
            for atk_type in attacking_types:
                orig_eff = original_matchup.get(atk_type.lower(), 1.0)
                tera_eff = tera_matchup.get(atk_type.lower(), 1.0)
                matchup_vs_attacks[atk_type] = {
                    "before": orig_eff,
                    "after": tera_eff,
                    "change": "better" if tera_eff < orig_eff else "worse" if tera_eff > orig_eff else "same"
                }
        
        return {
            "pokemon": pokemon,
            "original_types": original_types,
            "tera_type": tera_type,
            "changes": changes,
            "matchup_vs_attacks": matchup_vs_attacks,
            "summary": self._summarize_tera_change(changes)
        }
    
    def _calculate_defensive_matchup(self, types: List[str]) -> Dict[str, float]:
        """Calculate defensive matchup for given types."""
        matchup = {}
        all_types = ["normal", "fire", "water", "electric", "grass", "ice", 
                     "fighting", "poison", "ground", "flying", "psychic", 
                     "bug", "rock", "ghost", "dragon", "dark", "steel", "fairy"]
        
        for atk_type in all_types:
            effectiveness = 1.0
            for def_type in types:
                type_chart = self.typechart.get(def_type.lower(), {})
                damage_taken = type_chart.get("damageTaken", {})
                eff_value = damage_taken.get(atk_type, 0)
                
                if eff_value == 1:
                    effectiveness *= 2.0
                elif eff_value == 2:
                    effectiveness *= 0.5
                elif eff_value == 3:
                    effectiveness *= 0.0
            
            matchup[atk_type] = effectiveness
        
        return matchup
    
    def _summarize_tera_change(self, changes: Dict) -> str:
        """Generate human-readable summary of tera change."""
        parts = []
        
        if changes["lost_weaknesses"]:
            parts.append(f"No longer weak to: {', '.join(changes['lost_weaknesses'])}")
        if changes["new_resistances"]:
            parts.append(f"Now resists: {', '.join(changes['new_resistances'])}")
        if changes["new_immunities"]:
            parts.append(f"Now immune to: {', '.join(changes['new_immunities'])}")
        if changes["new_weaknesses"]:
            parts.append(f"Now weak to: {', '.join(changes['new_weaknesses'])}")
        
        return "; ".join(parts) if parts else "No significant defensive changes"
    
    # =========================================================================
    # Tool 3: Speed Order Check
    # =========================================================================
    def check_speed_order(
        self,
        pokemon_list: List[str],
        trick_room: bool = False,
        tailwind_sides: Optional[List[str]] = None,
        **kwargs
    ) -> Dict:
        """
        Determine turn order with current field modifiers.
        """
        tailwind_sides = tailwind_sides or []
        
        speed_data = []
        for pokemon in pokemon_list:
            # Check if it's our pokemon or opponent's
            is_mine = "my_side" in pokemon.lower() or not any(
                x in pokemon.lower() for x in ["opponent", "enemy", "foe"]
            )
            
            # Get base speed from cache
            pokemon_name = pokemon.split(":")[-1].strip() if ":" in pokemon else pokemon
            cached = self.cache.get(pokemon_name, "base_stats") or {}
            base_speed = cached.get("spe", 80)  # Default 80 if unknown
            
            # Get speed tier info if available
            speed_tier = self.cache.get(pokemon_name, "speed_tier") or {}
            min_speed = speed_tier.get("min", int(base_speed * 0.9 * 0.9))  # Min nature, 0 EV
            max_speed = speed_tier.get("max", int(base_speed * 1.1 * 1.1 + 31.5))  # Max nature, 252 EV
            
            # Apply Tailwind (2x speed)
            tailwind_mult = 2.0 if (
                (is_mine and "my_side" in tailwind_sides) or
                (not is_mine and "opponent_side" in tailwind_sides)
            ) else 1.0
            
            speed_data.append({
                "pokemon": pokemon,
                "is_mine": is_mine,
                "base_speed": base_speed,
                "min_speed": int(min_speed * tailwind_mult),
                "max_speed": int(max_speed * tailwind_mult),
                "tailwind": tailwind_mult > 1.0
            })
        
        # Sort by speed (reverse for Trick Room)
        speed_data.sort(
            key=lambda x: x["max_speed"], 
            reverse=not trick_room
        )
        
        return {
            "turn_order": [p["pokemon"] for p in speed_data],
            "details": speed_data,
            "trick_room": trick_room,
            "tailwind_active": len(tailwind_sides) > 0,
            "notes": self._speed_order_notes(speed_data, trick_room)
        }
    
    def _speed_order_notes(self, speed_data: List[Dict], trick_room: bool) -> List[str]:
        """Generate notes about speed comparisons."""
        notes = []
        
        if trick_room:
            notes.append("Trick Room active - slower Pokemon move first")
        
        # Check for speed ties or close ranges
        for i in range(len(speed_data) - 1):
            p1, p2 = speed_data[i], speed_data[i + 1]
            if p1["min_speed"] <= p2["max_speed"] and p1["max_speed"] >= p2["min_speed"]:
                notes.append(f"Speed tie possible: {p1['pokemon']} vs {p2['pokemon']}")
        
        return notes
    
    # =========================================================================
    # Tool 4: Update Pokemon Prediction
    # =========================================================================
    def update_pokemon_prediction(
        self,
        pokemon: str,
        observed: Dict,
        **kwargs
    ) -> Dict:
        """
        Update predictions after observing new information.
        Uses Bayesian-style reasoning to narrow down possibilities.
        """
        # Get current predictions from cache
        usage_stats = self.cache.get(pokemon, "usage_stats") or {}
        
        result = {
            "pokemon": pokemon,
            "observed": observed,
            "inferences": [],
            "updated_probabilities": {}
        }
        
        # If item observed, update move/ability probabilities
        if observed.get("item"):
            item = observed["item"]
            result["inferences"].append(f"Item confirmed: {item}")
            
            # Choice item implications
            if "choice" in item.lower():
                result["inferences"].append("Locked into one move per switch-in")
                result["inferences"].append("Likely max Speed or Attack EVs")
            elif "assault vest" in item.lower():
                result["inferences"].append("Cannot use status moves")
                result["inferences"].append("Likely specially defensive")
            elif "focus sash" in item.lower():
                result["inferences"].append("Will survive one hit from full HP")
                result["inferences"].append("Possibly glass cannon set")
        
        # If ability observed
        if observed.get("ability"):
            ability = observed["ability"]
            result["inferences"].append(f"Ability confirmed: {ability}")
            
            # Ability implications
            ability_lower = ability.lower()
            if "intimidate" in ability_lower:
                result["inferences"].append("Physical attacks weakened by 1 stage")
            elif "prankster" in ability_lower:
                result["inferences"].append("Status moves have +1 priority")
            elif "defiant" in ability_lower:
                result["inferences"].append("Attack +2 when stats lowered")
        
        # If move observed
        if observed.get("move"):
            move = observed["move"]
            result["inferences"].append(f"Move known: {move}")
            
            # Move set implications
            common_moves = usage_stats.get("moves", [])
            if common_moves:
                remaining = [m for m in common_moves if m["move"].lower() != move.lower()]
                result["updated_probabilities"]["likely_moves"] = remaining[:5]
        
        # If speed relation observed
        if observed.get("speed_relation"):
            relation = observed["speed_relation"]
            result["inferences"].append(f"Speed relation: {relation}")
            
            # Update speed estimate
            if "faster_than:" in relation:
                benchmark = relation.split(":")[1]
                result["inferences"].append(f"Faster than {benchmark} - likely Speed investment")
            elif "slower_than:" in relation:
                benchmark = relation.split(":")[1]
                result["inferences"].append(f"Slower than {benchmark} - possibly bulky/Trick Room set")
        
        # If tera type observed
        if observed.get("tera_type"):
            tera = observed["tera_type"]
            result["inferences"].append(f"Tera type confirmed: {tera}")
            result["inferences"].append("Consider new type matchups for remaining turns")
        
        return result
    
    # =========================================================================
    # Tool 5: Query Cached Info
    # =========================================================================
    def query_cached_info(
        self,
        pokemon: str,
        info_type: str,
        **kwargs
    ) -> Dict:
        """
        Retrieve cached info from team preview.
        Use sparingly - most data should already be in context.
        """
        valid_types = ["usage_stats", "base_stats", "common_moves", "common_items", "speed_tier"]
        
        if info_type not in valid_types:
            return {"error": f"Invalid info_type. Must be one of: {valid_types}"}
        
        cached_data = self.cache.get(pokemon, info_type)
        
        if cached_data:
            return {
                "pokemon": pokemon,
                "info_type": info_type,
                "data": cached_data,
                "source": "team_preview_cache"
            }
        else:
            return {
                "pokemon": pokemon,
                "info_type": info_type,
                "data": None,
                "error": "Not found in cache. Data may not have been collected during team preview."
            }


# =============================================================================
# Helper Functions
# =============================================================================

def get_tool_names() -> List[str]:
    """Get list of available tool names."""
    return [tool["name"] for tool in BATTLE_TOOLS]


def get_tool_by_name(name: str) -> Optional[Dict]:
    """Get tool definition by name."""
    for tool in BATTLE_TOOLS:
        if tool["name"] == name:
            return tool
    return None


def get_battle_tools_summary() -> str:
    """Get a summary of available battle tools for debugging."""
    summary = []
    for tool in BATTLE_TOOLS:
        summary.append(f"- {tool['name']}: {tool['description'][:50]}...")
    return "\n".join(summary)


# =============================================================================
# Tool Optimization Helpers
# =============================================================================

def should_call_tool(tool_name: str, context: Dict) -> bool:
    """
    Determine if a tool call is necessary given current context.
    Helps prevent redundant calls.
    
    :param tool_name: Name of tool being considered
    :param context: Current battle context with cached info
    :return: True if tool call would provide new information
    """
    cached_data = context.get("cached_data", {})
    
    if tool_name == "query_cached_info":
        # Only call if data isn't already in context
        pokemon = context.get("query_pokemon", "")
        info_type = context.get("query_type", "")
        return not cached_data.get(pokemon, {}).get(info_type)
    
    if tool_name == "get_tera_matchup":
        # Only needed if tera actually changed
        return context.get("tera_changed", False)
    
    if tool_name == "check_speed_order":
        # Only needed if field conditions changed since last check
        return context.get("field_changed", False)
    
    # calculate_damage and update_pokemon_prediction are always useful
    return True
