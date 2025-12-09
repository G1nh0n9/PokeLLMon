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

import requests
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
        "name": "calculate_showdown_damage",
        "description": "Calculate damage range for a specific attack. Use when you need precise damage numbers to determine if you can KO or survive. Returns min/max damage percentages.",
        "parameters": {
            "type": "object",
            "properties": {
                "attacker": {
                    "type": "object",
                    "description": "Attacking Pokemon data",
                    "properties": {
                        "species": {
                            "type": "string",
                            "description": "Pokemon species name"
                        },
                        "level": {
                            "type": "integer",
                            "description": "Pokemon level (1-100)"
                        },
                        "ability": {
                            "type": "string",
                            "description": "Pokemon's active ability"
                        },
                        "teraType": {
                            "type": "string",
                            "description": "Tera type when terastallized, omit if not terastallized"
                        },
                        "item": {
                            "type": "string",
                            "description": "Held item name"
                        },
                        "nature": {
                            "type": "string",
                            "description": "Pokemon's nature (e.g., 'Jolly', 'Adamant')"
                        },
                        "isSaltCure": {
                            "type": "boolean",
                            "description": "Whether Pokemon is affected by Salt Cure"
                        },
                        "alliesFainted": {
                            "type": "integer",
                            "description": "Number of fainted allies on user's side"
                        },
                        "originalCurHP": {
                            "type": "integer",
                            "description": "Current HP before attack"
                        },
                        "boosts": {
                            "type": "object",
                            "description": "Stat stage changes (-6 to +6)",
                            "properties": {
                                "atk": {
                                    "type": "integer",
                                    "description": "Attack stage boost"
                                },
                                "def": {
                                    "type": "integer",
                                    "description": "Defense stage boost"
                                },
                                "spa": {
                                    "type": "integer",
                                    "description": "Special Attack stage boost"
                                },
                                "spd": {
                                    "type": "integer",
                                    "description": "Special Defense stage boost"
                                },
                                "spe": {
                                    "type": "integer",
                                    "description": "Speed stage boost"
                                }
                            }
                        },
                        "ivs": {
                            "type": "object",
                            "description": "Individual Values (0-31 for each stat)",
                            "properties": {
                                "hp": {
                                    "type": "integer",
                                    "description": "HP IV"
                                },
                                "atk": {
                                    "type": "integer",
                                    "description": "Attack IV"
                                },
                                "def": {
                                    "type": "integer",
                                    "description": "Defense IV"
                                },
                                "spa": {
                                    "type": "integer",
                                    "description": "Special Attack IV"
                                },
                                "spd": {
                                    "type": "integer",
                                    "description": "Special Defense IV"
                                },
                                "spe": {
                                    "type": "integer",
                                    "description": "Speed IV"
                                }
                            }
                        },
                        "evs": {
                            "type": "object",
                            "description": "Effort Values (0-252 per stat, max 510 total)",
                            "properties": {
                                "hp": {
                                    "type": "integer",
                                    "description": "HP EV"
                                },
                                "atk": {
                                    "type": "integer",
                                    "description": "Attack EV"
                                },
                                "def": {
                                    "type": "integer",
                                    "description": "Defense EV"
                                },
                                "spa": {
                                    "type": "integer",
                                    "description": "Special Attack EV"
                                },
                                "spd": {
                                    "type": "integer",
                                    "description": "Special Defense EV"
                                },
                                "spe": {
                                    "type": "integer",
                                    "description": "Speed EV"
                                }
                            }
                        },
                        "status": {
                            "type": "string",
                            "description": "Status condition: slp (sleep), psn (poison), brn (burn), frz (freeze), par (paralysis), tox (badly poisoned)",
                            "enum": ["slp", "psn", "brn", "frz", "par", "tox"]
                        },
                        "toxicCounter": {
                            "type": "integer",
                            "description": "Badly poisoned counter (0 if not badly poisoned)"
                        }
                    },
                    "required": ["species", "level"]
                },
                "defender": {
                    "type": "object",
                    "description": "Defending Pokemon data",
                    "properties": {
                        "species": {
                            "type": "string",
                            "description": "Pokemon species name"
                        },
                        "level": {
                            "type": "integer",
                            "description": "Pokemon level (1-100)"
                        },
                        "ability": {
                            "type": "string",
                            "description": "Pokemon's active ability"
                        },
                        "teraType": {
                            "type": "string",
                            "description": "Tera type when terastallized, omit if not terastallized"
                        },
                        "item": {
                            "type": "string",
                            "description": "Held item name"
                        },
                        "nature": {
                            "type": "string",
                            "description": "Pokemon's nature (e.g., 'Jolly', 'Adamant')"
                        },
                        "isSaltCure": {
                            "type": "boolean",
                            "description": "Whether Pokemon is affected by Salt Cure"
                        },
                        "alliesFainted": {
                            "type": "integer",
                            "description": "Number of fainted allies on user's side"
                        },
                        "originalCurHP": {
                            "type": "integer",
                            "description": "Current HP before attack"
                        },
                        "boosts": {
                            "type": "object",
                            "description": "Stat stage changes (-6 to +6)",
                            "properties": {
                                "atk": {
                                    "type": "integer",
                                    "description": "Attack stage boost"
                                },
                                "def": {
                                    "type": "integer",
                                    "description": "Defense stage boost"
                                },
                                "spa": {
                                    "type": "integer",
                                    "description": "Special Attack stage boost"
                                },
                                "spd": {
                                    "type": "integer",
                                    "description": "Special Defense stage boost"
                                },
                                "spe": {
                                    "type": "integer",
                                    "description": "Speed stage boost"
                                }
                            }
                        },
                        "ivs": {
                            "type": "object",
                            "description": "Individual Values (0-31 for each stat)",
                            "properties": {
                                "hp": {
                                    "type": "integer",
                                    "description": "HP IV"
                                },
                                "atk": {
                                    "type": "integer",
                                    "description": "Attack IV"
                                },
                                "def": {
                                    "type": "integer",
                                    "description": "Defense IV"
                                },
                                "spa": {
                                    "type": "integer",
                                    "description": "Special Attack IV"
                                },
                                "spd": {
                                    "type": "integer",
                                    "description": "Special Defense IV"
                                },
                                "spe": {
                                    "type": "integer",
                                    "description": "Speed IV"
                                }
                            }
                        },
                        "evs": {
                            "type": "object",
                            "description": "Effort Values (0-252 per stat, max 510 total)",
                            "properties": {
                                "hp": {
                                    "type": "integer",
                                    "description": "HP EV"
                                },
                                "atk": {
                                    "type": "integer",
                                    "description": "Attack EV"
                                },
                                "def": {
                                    "type": "integer",
                                    "description": "Defense EV"
                                },
                                "spa": {
                                    "type": "integer",
                                    "description": "Special Attack EV"
                                },
                                "spd": {
                                    "type": "integer",
                                    "description": "Special Defense EV"
                                },
                                "spe": {
                                    "type": "integer",
                                    "description": "Speed EV"
                                }
                            }
                        },
                        "status": {
                            "type": "string",
                            "description": "Status condition: slp (sleep), psn (poison), brn (burn), frz (freeze), par (paralysis), tox (badly poisoned)",
                            "enum": ["slp", "psn", "brn", "frz", "par", "tox"]
                        },
                        "toxicCounter": {
                            "type": "integer",
                            "description": "Badly poisoned counter (0 if not badly poisoned)"
                        }
                    },
                    "required": ["species", "level"]
                },
                "move": {
                    "type": "string",
                    "description": "Move being used"
                },
                "field_conditions": {
                    "type": "object",
                    "description": "Current field conditions (optional)",
                    "properties": {
                        "gameType": {
                            "type": "string",
                            "description": "Game type ('Singles' or 'Doubles')"
                        },
                        "terrain": {
                            "type": "string",
                            "description": "Active terrain (e.g., 'Electric', 'Grassy', 'Misty', 'Psychic')"
                        },
                        "weather": {
                            "type": "string",
                            "description": "Active weather (e.g., 'Rain', 'Sun', 'Hail', 'Sandstorm')"
                        },
                        "isMagicRoom": {
                            "type": "boolean",
                            "description": "Magic Room active"
                        },
                        "isWonderRoom": {
                            "type": "boolean",
                            "description": "Wonder Room active"
                        },
                        "isGravity": {
                            "type": "boolean",
                            "description": "Gravity active"
                        },
                        "isAuraBreak": {
                            "type": "boolean",
                            "description": "Aura Break active"
                        },
                        "isFairyAura": {
                            "type": "boolean",
                            "description": "Fairy Aura active"
                        },
                        "isDarkAura": {
                            "type": "boolean",
                            "description": "Dark Aura active"
                        },
                        "isBeadsOfRuin": {
                            "type": "boolean",
                            "description": "Beads of Ruin active"
                        },
                        "isSwordOfRuin": {
                            "type": "boolean",
                            "description": "Sword of Ruin active"
                        },
                        "isTabletsOfRuin": {
                            "type": "boolean",
                            "description": "Tablets of Ruin active"
                        },
                        "isVesselOfRuin": {
                            "type": "boolean",
                            "description": "Vessel of Ruin active"
                        },
                        "attackerSide": {
                            "type": "object",
                            "description": "Attacker's side conditions",
                            "properties": {
                                "spikes": {
                                    "type": "integer",
                                    "description": "Spikes layers (0-3)"
                                },
                                "steelsurge": {
                                    "type": "boolean",
                                    "description": "Steel Surge active"
                                },
                                "vinelash": {
                                    "type": "boolean",
                                    "description": "Vine Lash active"
                                },
                                "wildfire": {
                                    "type": "boolean",
                                    "description": "Wildfire active"
                                },
                                "cannonade": {
                                    "type": "boolean",
                                    "description": "Cannonade active"
                                },
                                "volcalith": {
                                    "type": "boolean",
                                    "description": "Volcalith active"
                                },
                                "isSR": {
                                    "type": "boolean",
                                    "description": "Stealth Rock active"
                                },
                                "isReflect": {
                                    "type": "boolean",
                                    "description": "Reflect active"
                                },
                                "isLightScreen": {
                                    "type": "boolean",
                                    "description": "Light Screen active"
                                },
                                "isProtected": {
                                    "type": "boolean",
                                    "description": "Protected active"
                                },
                                "isSeeded": {
                                    "type": "boolean",
                                    "description": "Leech Seed active"
                                },
                                "isForesight": {
                                    "type": "boolean",
                                    "description": "Foresight active"
                                },
                                "isTailwind": {
                                    "type": "boolean",
                                    "description": "Tailwind active"
                                },
                                "isHelpingHand": {
                                    "type": "boolean",
                                    "description": "Helping Hand active"
                                },
                                "isFlowerGift": {
                                    "type": "boolean",
                                    "description": "Flower Gift active"
                                },
                                "isFriendGuard": {
                                    "type": "boolean",
                                    "description": "Friend Guard active"
                                },
                                "isAuroraVeil": {
                                    "type": "boolean",
                                    "description": "Aurora Veil active"
                                },
                                "isBattery": {
                                    "type": "boolean",
                                    "description": "Battery ability active"
                                },
                                "isPowerSpot": {
                                    "type": "boolean",
                                    "description": "Power Spot active"
                                },
                                "isSwitching": {
                                    "type": ["boolean", "null"],
                                    "description": "Pokemon switching state"
                                }
                            }
                        },
                        "defenderSide": {
                            "type": "object",
                            "description": "Defender's side conditions",
                            "properties": {
                                "spikes": {
                                    "type": "integer",
                                    "description": "Spikes layers (0-3)"
                                },
                                "steelsurge": {
                                    "type": "boolean",
                                    "description": "Steel Surge active"
                                },
                                "vinelash": {
                                    "type": "boolean",
                                    "description": "Vine Lash active"
                                },
                                "wildfire": {
                                    "type": "boolean",
                                    "description": "Wildfire active"
                                },
                                "cannonade": {
                                    "type": "boolean",
                                    "description": "Cannonade active"
                                },
                                "volcalith": {
                                    "type": "boolean",
                                    "description": "Volcalith active"
                                },
                                "isSR": {
                                    "type": "boolean",
                                    "description": "Stealth Rock active"
                                },
                                "isReflect": {
                                    "type": "boolean",
                                    "description": "Reflect active"
                                },
                                "isLightScreen": {
                                    "type": "boolean",
                                    "description": "Light Screen active"
                                },
                                "isProtected": {
                                    "type": "boolean",
                                    "description": "Protected active"
                                },
                                "isSeeded": {
                                    "type": "boolean",
                                    "description": "Leech Seed active"
                                },
                                "isForesight": {
                                    "type": "boolean",
                                    "description": "Foresight active"
                                },
                                "isTailwind": {
                                    "type": "boolean",
                                    "description": "Tailwind active"
                                },
                                "isHelpingHand": {
                                    "type": "boolean",
                                    "description": "Helping Hand active"
                                },
                                "isFlowerGift": {
                                    "type": "boolean",
                                    "description": "Flower Gift active"
                                },
                                "isFriendGuard": {
                                    "type": "boolean",
                                    "description": "Friend Guard active"
                                },
                                "isAuroraVeil": {
                                    "type": "boolean",
                                    "description": "Aurora Veil active"
                                },
                                "isBattery": {
                                    "type": "boolean",
                                    "description": "Battery ability active"
                                },
                                "isPowerSpot": {
                                    "type": "boolean",
                                    "description": "Power Spot active"
                                },
                                "isSwitching": {
                                    "type": ["boolean", "null"],
                                    "description": "Pokemon switching state"
                                }
                            }
                        }
                    }
                }
            },
            "required": ["attacker", "defender", "move"]
        }
    },
    
    # =========================================================================
    # 2. Move Info - Get move metadata
    # =========================================================================
    {
        "type": "function",
        "name": "get_move_info",
        "description": "Get metadata about a move: category, base power, priority, spread flag, and targeting. Use when you need to understand how a move behaves (single-target vs spread, priority, etc.).",
        "parameters": {
            "type": "object",
            "properties": {
                "move": {
                    "type": "string",
                    "description": "Move name or ID"
                }
            },
            "required": ["move"]
        }
    },
    
    # =========================================================================
    # 3. Residual Damage Estimation - Hazards, weather, status
    # =========================================================================
    {
        "type": "function",
        "name": "estimate_residual_damage",
        "description": "Estimate HP after entry hazards, weather, and status damage at the end of the current turn. Use when comparing switch vs stay-in lines.",
        "parameters": {
            "type": "object",
            "properties": {
                "pokemon": {
                    "type": "object",
                    "description": "Pokemon state (hp, maxHP, types, item, ability, status, etc.)",
                    "properties": {
                        "species": {"type": "string"},
                        "current_hp": {"type": "integer"},
                        "max_hp": {"type": "integer"},
                        "types": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "item": {"type": "string"},
                        "ability": {"type": "string"},
                        "status": {"type": "string"}
                    },
                    "required": ["species", "current_hp", "max_hp", "types"]
                },
                "field_conditions": {
                    "type": "object",
                    "description": "Hazards and weather/status relevant to residual damage",
                    "properties": {
                        "isSR": {"type": "boolean"},
                        "spikes": {"type": "integer"},
                        "toxic_spikes": {"type": "integer"},
                        "weather": {"type": "string"},
                        "terrain": {"type": "string"}
                    }
                },
                "is_switching_in": {
                    "type": "boolean",
                    "description": "Is this Pokemon switching in (hazards apply)?"
                }
            },
            "required": ["pokemon"]
        }
    },
    
    # =========================================================================
    # 4. Type Matchup - Basic type effectiveness (non-tera)
    # =========================================================================
    {
        "type": "function",
        "name": "get_type_matchup",
        "description": "Get type effectiveness multiplier (attacking type vs defending types). Use for non-tera type checks and quick matchup evaluation. Returns multiplier (0.0=immune, 0.25=4x resist, 0.5=resist, 1.0=neutral, 2.0=super effective, 4.0=4x super effective).",
        "parameters": {
            "type": "object",
            "properties": {
                "attack_type": {
                    "type": "string",
                    "description": "Attacking move type (e.g., 'fire', 'water', 'dragon')"
                },
                "defend_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Defender's types before tera (e.g., ['grass', 'poison'])"
                }
            },
            "required": ["attack_type", "defend_types"]
        }
    },
    
    # =========================================================================
    # 5. Tera Matchup - Only needed when tera changes situation
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
    # 6. Speed Check with Current Modifiers
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
            "calculate_showdown_damage": self.calculate_showdown_damage,
            "get_move_info": self.get_move_info,
            "estimate_residual_damage": self.estimate_residual_damage,
            "get_type_matchup": self.get_type_matchup,
            "get_tera_matchup": self.get_tera_matchup,
            "check_speed_order": self.check_speed_order,
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
    
    # =========================================================================
    # Tool 1: Damage Calculation From Showdown Format
    # =========================================================================
    def calculate_showdown_damage(
        self, 
        attacker: Dict, 
        defender: Dict, 
        move: str,
        field_conditions: Optional[Dict] = {},
        **kwargs
    ) -> Dict:
        """
        Sends data required for Showdown damage calculation to the endpoint and returns the result.
        
        Modifiers include: STAB, type effectiveness, weather, terrain, items, abilities
        """

        url = 'http://localhost:3000/calculate'
        pokemon_gen = 9  # Assuming Gen 9 for this example

        try:
            response = requests.post(
                url,
                json={
                    "generation": pokemon_gen,
                    "attacker": attacker,
                    "defender": defender,
                    "move": move,
                    "fieldConditions": field_conditions
                },
                timeout=10
            )
            
            response.raise_for_status()
            
            # Parse JSON response
            result = response.json()
            
            # Extract relevant data
            attacker_name = result.get("attackerName", attacker['species'])
            attacker_info = result.get("attacker", {})
            attacker_types = attacker_info.get("types", [])
            attacker_tera = attacker_info.get("teraType")  # 기본값 None
            defender_name = result.get("defenderName", defender['species'])
            defender_original_cur_hp = result.get("defender", {}).get("originalCurHP", 0)
            move_name = result.get("move", {}).get("originalName", move)
            move_type = result.get("move", {}).get("type", "???")
            move_bp = result.get("move", {}).get("bp", 0)
            field = result.get("field", field_conditions)
            
            # Calculate type effectiveness
            effectiveness = self._calculate_type_effectiveness(
                result.get("move", {}).get("type", "???"),
                defender['species'],
                defender['teraType']
            )
            
            # STAB check (including Tera STAB)
            stab = 1.0
            if attacker_tera:
                if move_type.lower() == attacker_tera.lower():
                    stab = 2.0 if move_type.lower() in [t.lower() for t in attacker_types] else 1.5
            elif move_type in attacker_types:
                stab = 1.5
            
            # damage range (min, max)
            min_damage = min(result.get("damage", [0]))
            max_damage = max(result.get("damage", [0]))
            min_percent = round(min_damage / defender_original_cur_hp * 100, 1) \
                if defender_original_cur_hp > 0 else 0
            max_percent = round(max_damage / defender_original_cur_hp * 100, 1) \
                if defender_original_cur_hp > 0 else 0
            
            return {
                "attacker": attacker_name,
                "defender": defender_name,
                "move": move_name,
                "move_type": move_type,
                "base_power": move_bp,
                "effectiveness": effectiveness,
                "stab": stab,
                "min_damage": min_damage,
                "max_damage": max_damage,
                "min_damage_percent": min_percent,
                "max_damage_percent": max_percent,
                "is_ohko": min_percent >= 100,
                "is_2hko": min_percent >= 50,
                "notes": self._get_damage_notes(effectiveness, stab, field)
            }
            
        except requests.exceptions.RequestException as e:
            print(f"Showdown Damage Calculation Request Failed: {e}")
            return self.calculate_damage(
                attacker=attacker.get('species', 'Unknown'),
                defender=defender.get('species', 'Unknown'),
                move=move,
                attacker_tera_type=attacker.get('teraType') or None,
                defender_tera_type=defender.get('teraType') or None,
                field_conditions=field_conditions if field_conditions else None
            )
        except json.JSONDecodeError as e:
            print(f"Server response JSON decoding failed: {e}")
            return self.calculate_damage(
                attacker=attacker.get('species', 'Unknown'),
                defender=defender.get('species', 'Unknown'),
                move=move,
                attacker_tera_type=attacker.get('teraType') or None,
                defender_tera_type=defender.get('teraType') or None,
                field_conditions=field_conditions if field_conditions else None
            )
    
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
    # Tool 2: Move Info
    # =========================================================================
    def get_move_info(
        self,
        move: str,
        **kwargs
    ) -> Dict:
        """
        Get metadata about a move for battle decision-making.
        
        Returns move properties:
        - category: Physical, Special, or Status
        - basePower: Base power of the move (0 for status moves)
        - priority: Priority bracket (-7 to +5)
        - target: Targeting mode (normal, allAdjacent, allAdjacentFoes, etc.)
        - isSpread: Whether the move hits multiple targets (affects damage)
        - type: Move type
        - accuracy: Move accuracy (True for never-miss moves)
        - flags: Important flags (protect, contact, sound, etc.)
        """
        # Try to get move data from PKHeX.Core first (primary source)
        move_data = None

        try:
            from src.data.gen_data import GenData
            gen_data = GenData.from_gen(9)
            move_id = move.lower().replace(" ", "").replace("-", "").replace("'", "")
            gen_move_data = gen_data.moves.get(move_id, {})
            if gen_move_data:
                move_data = gen_move_data
        except Exception as e:
            print(f"Failed to load move data from GenData: {e}")
        
        # Final fallback to self.moves
        if not move_data:
            move_id = move.lower().replace(" ", "").replace("-", "").replace("'", "")
            move_data = self.moves.get(move_id, {})
        
        if not move_data:
            return {
                "error": f"Move '{move}' not found in database",
                "move": move,
                "suggestion": "Check move name spelling or use exact Showdown ID"
            }
        
        # Extract key metadata
        category = move_data.get("category", "Status")
        base_power = move_data.get("basePower", 0)
        priority = move_data.get("priority", 0)
        target = move_data.get("target", "normal")
        move_type = move_data.get("type", "Normal")
        accuracy = move_data.get("accuracy", True)  # True means never-miss
        
        # Determine if move is spread (hits multiple targets)
        spread_targets = ["allAdjacent", "allAdjacentFoes", "allySide", "foeSide", "all"]
        is_spread = target in spread_targets
        
        # Extract important flags
        flags = move_data.get("flags", {})
        important_flags = {
            "protect": flags.get("protect", False),  # Blocked by Protect
            "bypasssub": flags.get("bypasssub", False),  # Bypasses Substitute
            "contact": flags.get("contact", False),  # Makes contact (triggers Rocky Helmet, etc.)
            "sound": flags.get("sound", False),  # Sound-based move
            "punch": flags.get("punch", False),  # Punch move (Iron Fist boost)
            "bite": flags.get("bite", False),  # Bite move (Strong Jaw boost)
            "bullet": flags.get("bullet", False),  # Bullet move (blocked by Bulletproof)
            "pulse": flags.get("pulse", False),  # Pulse move (Mega Launcher boost)
            "wind": flags.get("wind", False),  # Wind move (Wind Power/Wind Rider)
        }
        
        # Get secondary effects info
        secondary = move_data.get("secondary", {})
        has_secondary = bool(secondary)
        
        # Target description for clarity
        target_descriptions = {
            "normal": "Single adjacent target",
            "allAdjacent": "All adjacent Pokemon (enemies and allies)",
            "allAdjacentFoes": "Both adjacent opponents (spread move)",
            "adjacentAlly": "Adjacent ally only",
            "adjacentAllyOrSelf": "Self or adjacent ally",
            "adjacentFoe": "Adjacent opponent",
            "allySide": "Your entire side (ally + self)",
            "foeSide": "Opponent's entire side",
            "all": "All Pokemon on field",
            "self": "User only",
            "randomNormal": "Random adjacent opponent",
            "scripted": "Special targeting (varies by move)",
        }
        
        target_desc = target_descriptions.get(target, target)
        
        # Compile damage reduction info for spread moves
        spread_note = ""
        if is_spread and category in ["Physical", "Special"]:
            spread_note = "Damage reduced to 75% when hitting multiple targets"
        
        return {
            "move": move_data.get("name", move),
            "move_id": move_id,
            "category": category,
            "type": move_type,
            "base_power": base_power,
            "priority": priority,
            "target": target,
            "target_description": target_desc,
            "is_spread": is_spread,
            "spread_note": spread_note,
            "accuracy": accuracy,
            "flags": important_flags,
            "has_secondary_effect": has_secondary,
            "secondary_effect": secondary if has_secondary else None,
            "notes": self._get_move_notes(move_data, category, priority, is_spread)
        }
    
    def _get_move_notes(self, move_data: Dict, category: str, priority: int, is_spread: bool) -> List[str]:
        """Generate human-readable notes about move behavior."""
        notes = []
        
        # Priority notes
        if priority > 0:
            notes.append(f"Priority +{priority} (goes before most moves)")
        elif priority < 0:
            notes.append(f"Priority {priority} (goes after most moves)")
        
        # Category notes
        if category == "Status":
            notes.append("Status move (no direct damage)")
        
        # Spread notes
        if is_spread and category in ["Physical", "Special"]:
            notes.append("Spread move: 75% damage when hitting multiple targets")
        
        # Special move properties
        if move_data.get("multihit"):
            hits = move_data["multihit"]
            if isinstance(hits, list):
                notes.append(f"Multi-hit: {hits[0]}-{hits[1]} times")
            else:
                notes.append(f"Multi-hit: {hits} times")
        
        if move_data.get("recoil"):
            recoil = move_data["recoil"]
            notes.append(f"Recoil: {abs(recoil[0])}/{recoil[1]} of damage dealt")
        
        if move_data.get("drain"):
            drain = move_data["drain"]
            notes.append(f"Draining: Recovers {drain[0]}/{drain[1]} of damage dealt")
        
        if move_data.get("hasCrashDamage"):
            notes.append("Crash damage if it misses")
        
        if move_data.get("selfdestruct"):
            notes.append("User faints after using")
        
        # Protection/bypass notes
        flags = move_data.get("flags", {})
        if not flags.get("protect", False):
            notes.append("Bypasses Protect/Detect")
        
        if flags.get("bypasssub", False):
            notes.append("Bypasses Substitute")
        
        return notes
    
    # =========================================================================
    # Tool 3: Residual Damage Estimation
    # =========================================================================
    def estimate_residual_damage(
        self,
        pokemon: Dict,
        field_conditions: Optional[Dict] = None,
        is_switching_in: bool = False,
        **kwargs
    ) -> Dict:
        """
        Estimate HP after entry hazards, weather, and status damage.
        
        Calculates damage from:
        - Entry hazards (Stealth Rock, Spikes, Toxic Spikes) when switching in
        - Weather damage (Sandstorm, Hail)
        - Status damage (Burn, Poison, Toxic)
        - Item recovery (Leftovers, Black Sludge)
        - Ability effects (Dry Skin, Solar Power, etc.)
        """
        field_conditions = field_conditions or {}
        
        species = pokemon.get("species", "Unknown")
        current_hp = pokemon.get("current_hp", 100)
        max_hp = pokemon.get("max_hp", 100)
        types = [t.lower() for t in pokemon.get("types", [])]
        item = pokemon.get("item", "").lower()
        ability = pokemon.get("ability", "").lower()
        status = pokemon.get("status", "").lower()
        
        damage_sources = []
        total_damage = 0
        total_healing = 0
        
        # =====================================================================
        # Entry Hazards (only when switching in)
        # =====================================================================
        if is_switching_in:
            # Stealth Rock
            if field_conditions.get("isSR", False):
                # Get rock type effectiveness
                rock_effectiveness = 1.0
                for poke_type in types:
                    type_chart = self.typechart.get(poke_type, {})
                    damage_taken = type_chart.get("damageTaken", {})
                    rock_eff = damage_taken.get("rock", 0)
                    
                    if rock_eff == 1:  # Super effective
                        rock_effectiveness *= 2.0
                    elif rock_eff == 2:  # Not very effective
                        rock_effectiveness *= 0.5
                    elif rock_eff == 3:  # Immune
                        rock_effectiveness *= 0.0
                
                sr_damage = int(max_hp * (rock_effectiveness / 8))
                if sr_damage > 0:
                    total_damage += sr_damage
                    damage_sources.append(f"Stealth Rock: -{sr_damage} HP ({rock_effectiveness}x effective)")
            
            # Spikes (1/8, 1/6, 1/4 for 1/2/3 layers)
            spikes_layers = field_conditions.get("spikes", 0)
            if spikes_layers > 0 and "flying" not in types and ability != "levitate":
                spikes_fractions = {1: 8, 2: 6, 3: 4}
                spikes_damage = max_hp // spikes_fractions.get(spikes_layers, 8)
                total_damage += spikes_damage
                damage_sources.append(f"Spikes (L{spikes_layers}): -{spikes_damage} HP")
            
            # Toxic Spikes (poison or badly poison)
            toxic_spikes = field_conditions.get("toxic_spikes", 0)
            if toxic_spikes > 0 and "poison" not in types and "steel" not in types:
                if toxic_spikes == 1:
                    damage_sources.append("Toxic Spikes: Poisoned")
                elif toxic_spikes >= 2:
                    damage_sources.append("Toxic Spikes: Badly Poisoned")
        
        # =====================================================================
        # Weather Damage (every turn)
        # =====================================================================
        weather = field_conditions.get("weather", "").lower()
        
        if weather in ["sandstorm", "sand"]:
            # Sandstorm: 1/16 damage unless Rock/Steel/Ground or specific abilities
            if not any(t in types for t in ["rock", "steel", "ground"]):
                if ability not in ["sandveil", "sandrush", "sandforce", "overcoat", "magicguard"]:
                    sand_damage = max_hp // 16
                    total_damage += sand_damage
                    damage_sources.append(f"Sandstorm: -{sand_damage} HP")
        
        elif weather in ["hail", "snow"]:
            # Hail/Snow: 1/16 damage unless Ice type or specific abilities
            if "ice" not in types:
                if ability not in ["icebody", "snowcloak", "overcoat", "magicguard"]:
                    hail_damage = max_hp // 16
                    total_damage += hail_damage
                    damage_sources.append(f"Hail/Snow: -{hail_damage} HP")
        
        # =====================================================================
        # Status Damage
        # =====================================================================
        if status == "brn":
            # Burn: 1/16 damage
            burn_damage = max_hp // 16
            total_damage += burn_damage
            damage_sources.append(f"Burn: -{burn_damage} HP")
        
        elif status in ["psn", "tox"]:
            # Poison: 1/8 damage
            poison_damage = max_hp // 8
            total_damage += poison_damage
            damage_sources.append(f"Poison: -{poison_damage} HP")
        
        # =====================================================================
        # Item Recovery
        # =====================================================================
        if item == "leftovers":
            leftovers_heal = max_hp // 16
            total_healing += leftovers_heal
            damage_sources.append(f"Leftovers: +{leftovers_heal} HP")
        
        elif item == "blacksludge":
            if "poison" in types:
                sludge_heal = max_hp // 16
                total_healing += sludge_heal
                damage_sources.append(f"Black Sludge: +{sludge_heal} HP")
            else:
                sludge_damage = max_hp // 8
                total_damage += sludge_damage
                damage_sources.append(f"Black Sludge: -{sludge_damage} HP (not Poison type)")
        
        # =====================================================================
        # Ability Effects
        # =====================================================================
        if ability == "icebody" and weather in ["hail", "snow"]:
            ice_body_heal = max_hp // 16
            total_healing += ice_body_heal
            damage_sources.append(f"Ice Body: +{ice_body_heal} HP")
        
        elif ability == "raindish" and weather in ["rain", "raindance"]:
            rain_dish_heal = max_hp // 16
            total_healing += rain_dish_heal
            damage_sources.append(f"Rain Dish: +{rain_dish_heal} HP")
        
        elif ability == "dryskin":
            if weather in ["rain", "raindance"]:
                dry_skin_heal = max_hp // 8
                total_healing += dry_skin_heal
                damage_sources.append(f"Dry Skin (Rain): +{dry_skin_heal} HP")
            elif weather in ["sun", "sunnyday", "desolateland"]:
                dry_skin_damage = max_hp // 8
                total_damage += dry_skin_damage
                damage_sources.append(f"Dry Skin (Sun): -{dry_skin_damage} HP")
        
        elif ability == "solarpower" and weather in ["sun", "sunnyday", "desolateland"]:
            solar_power_damage = max_hp // 8
            total_damage += solar_power_damage
            damage_sources.append(f"Solar Power: -{solar_power_damage} HP")
        
        # =====================================================================
        # Calculate Final HP
        # =====================================================================
        net_damage = total_damage - total_healing
        predicted_hp = max(0, current_hp - net_damage)
        hp_percent = round((predicted_hp / max_hp) * 100, 1) if max_hp > 0 else 0
        
        return {
            "species": species,
            "current_hp": current_hp,
            "predicted_hp": predicted_hp,
            "max_hp": max_hp,
            "hp_percent": hp_percent,
            "damage_taken": total_damage,
            "healing_received": total_healing,
            "net_damage": net_damage,
            "will_faint": predicted_hp <= 0,
            "damage_sources": damage_sources,
            "is_switching_in": is_switching_in,
            "summary": f"{species}: {current_hp} → {predicted_hp} HP ({hp_percent}%)"
        }
    
    # =========================================================================
    # Tool 4: Basic Type Matchup
    # =========================================================================
    def get_type_matchup(
        self,
        attack_type: str,
        defend_types: List[str],
        **kwargs
    ) -> Dict:
        """
        Calculate type effectiveness multiplier for attacking type vs defending types.
        Simple, fast type chart lookup - no damage calculation needed.
        
        Returns:
            {
                "attack_type": "fire",
                "defend_types": ["grass", "poison"],
                "multiplier": 2.0,
                "effectiveness": "super_effective",
                "explanation": "Fire is super effective against Grass"
            }
        """
        attack_type = attack_type.lower()
        defend_types = [t.lower() for t in defend_types]
        
        # Calculate overall multiplier
        multiplier = 1.0
        contributing_factors = []
        
        for def_type in defend_types:
            type_chart = self.typechart.get(def_type, {})
            damage_taken = type_chart.get("damageTaken", {})
            eff_value = damage_taken.get(attack_type, 0)
            
            if eff_value == 1:  # Super effective
                multiplier *= 2.0
                contributing_factors.append(f"{attack_type.capitalize()} is super effective against {def_type.capitalize()}")
            elif eff_value == 2:  # Not very effective
                multiplier *= 0.5
                contributing_factors.append(f"{attack_type.capitalize()} is not very effective against {def_type.capitalize()}")
            elif eff_value == 3:  # Immune
                multiplier *= 0.0
                contributing_factors.append(f"{def_type.capitalize()} is immune to {attack_type.capitalize()}")
        
        # Determine effectiveness category
        if multiplier == 0.0:
            effectiveness = "immune"
        elif multiplier <= 0.25:
            effectiveness = "4x_resisted"
        elif multiplier == 0.5:
            effectiveness = "resisted"
        elif multiplier == 1.0:
            effectiveness = "neutral"
        elif multiplier == 2.0:
            effectiveness = "super_effective"
        elif multiplier >= 4.0:
            effectiveness = "4x_super_effective"
        else:
            effectiveness = "neutral"
        
        explanation = "; ".join(contributing_factors) if contributing_factors else "Neutral damage"
        
        return {
            "attack_type": attack_type,
            "defend_types": defend_types,
            "multiplier": multiplier,
            "effectiveness": effectiveness,
            "explanation": explanation
        }
    
    # =========================================================================
    # Tool 3: Tera Matchup
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
    


if __name__ == "__main__":
    executor = ToolExecutor()
    # Example usage
    move_info = executor.get_move_info("fakeout")
    print(json.dumps(move_info, indent=2))

    data = {
  "attacker": {
    "species": "torkoal",
    "level": 100,
    "ability": "drought",
    "teraType": "FIRE",
    "item": "choicespecs",
    "nature": "Modest",
    "isSaltCure": False,
    "alliesFainted": 1,
    "originalCurHP": 2,
    "boosts": {
      "atk": 0,
      "def": 0,
      "spa": 1,
      "spd": 0,
      "spe": 0
    },
    "ivs": {
      "hp": 31,
      "atk": 0,
      "def": 0,
      "spa": 31,
      "spd": 0,
      "spe": 0
    },
    "evs": {
      "hp": 0,
      "atk": 0,
      "def": 0,
      "spa": 252,
      "spd": 0,
      "spe": 0
    },
    "status": "brn",
    "toxicCounter": 0
  },
  "defender": {
    "species": "ursaluna",
    "level": 100,
    "ability": "",
    "teraType": "",
    "item": "unknown_item",
    "nature": "",
    "isSaltCure": False,
    "alliesFainted": 0,
    "originalCurHP": 1,
    "boosts": {
      "atk": 0,
      "def": 0,
      "spa": 0,
      "spd": 0,
      "spe": 0
    },
    "ivs": {
      "hp": 31,
      "atk": 31,
      "def": 31,
      "spa": 31,
      "spd": 31,
      "spe": 31
    },
    "evs": {
      "hp": 0,
      "atk": 0,
      "def": 0,
      "spa": 0,
      "spd": 0,
      "spe": 0
    },
    "status": "brn",
    "toxicCounter": 0
  },
  "move": "eruption",
  "field_conditions": {
    "gameType": "Doubles",
    "terrain": "Psychic",
    "weather": "Sunny",
    "isMagicRoom": False,
    "isWonderRoom": False,
    "isGravity": False,
    "isAuraBreak": False,
    "isFairyAura": False,
    "isDarkAura": False,
    "isBeadsOfRuin": False,
    "isSwordOfRuin": False,
    "isTabletsOfRuin": False,
    "isVesselOfRuin": False,
    "attackerSide": {
      "spikes": 0,
      "steelsurge": False,
      "vinelash": False,
      "wildfire": False,
      "cannonade": False,
      "volcalith": False,
      "isSR": False,
      "isReflect": False,
      "isLightScreen": False,
      "isProtected": False,
      "isSeeded": False,
      "isForesight": False,
      "isTailwind": False,
      "isHelpingHand": False,
      "isFlowerGift": False,
      "isFriendGuard": False,
      "isAuroraVeil": False,
      "isBattery": False,
      "isPowerSpot": False,
      "isSwitching": None
    },
    "defenderSide": {
      "spikes": 0,
      "steelsurge": False,
      "vinelash": False,
      "wildfire": False,
      "cannonade": False,
      "volcalith": False,
      "isSR": False,
      "isReflect": False,
      "isLightScreen": False,
      "isProtected": False,
      "isSeeded": False,
      "isForesight": False,
      "isTailwind": False,
      "isHelpingHand": False,
      "isFlowerGift": False,
      "isFriendGuard": False,
      "isAuroraVeil": False,
      "isBattery": False,
      "isPowerSpot": False,
      "isSwitching": None
    }
  }
}
    calculation = executor.calculate_showdown_damage(
        data["attacker"],
        data["defender"],
        data["move"],
        data["field_conditions"],
    )

    print(json.dumps(calculation, indent=2))

    matchup_data = {
  "attack_type": "POISON",
  "defend_types": [
    "GRASS",
    "FAIRY"
  ]
}
    type_matchup = executor.get_type_matchup(attack_type=matchup_data["attack_type"], defend_types=matchup_data["defend_types"])
    print(json.dumps(type_matchup, indent=2))
