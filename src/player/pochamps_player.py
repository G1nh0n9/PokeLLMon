"""
PochampsPlayer - Custom Pokemon Battle AI using OpenAI API.
Implements parallel strategy evaluation with function calling.
"""

import copy
import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Any

from openai import OpenAI

from src.data.gen_data import GenData
from src.environment.abstract_battle import AbstractBattle
from src.environment.double_battle import DoubleBattle
from src.player.player import Player, BattleOrder
from src.player.battle_order import DoubleBattleOrder, ForfeitBattleOrder
from src.player.functions import BATTLE_TOOLS, ToolExecutor, TeamPreviewCache
from src.player.strategies import call_fast_strategy, call_normal_strategy, call_deep_strategy
from src.player import pkhex_core


# =============================================================================
# Battle Logger - Detailed turn-by-turn logging for debugging
# =============================================================================

class BattleLogger:
    """
    Comprehensive battle logger that tracks:
    - GPT context (what info was given to the model)
    - GPT analysis (what the model reasoned)
    - Decision made (what action was chosen)
    - Server response (what actually happened)
    """
    
    def __init__(self, battle_tag: str, log_dir: Optional[str] = None):
        self.battle_tag = battle_tag
        self.log_dir = log_dir or "battle_log"
        self.turns: List[Dict] = []
        self.current_turn: Optional[Dict] = None
        self.start_time = datetime.now()
        
        # Create log directory if needed
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Sanitize battle tag for filename
        safe_tag = battle_tag.replace("/", "_").replace(":", "_")
        self.log_file = os.path.join(self.log_dir, f"{safe_tag}_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json")
        self.html_file = os.path.join(self.log_dir, f"{safe_tag}_{self.start_time.strftime('%Y%m%d_%H%M%S')}_analysis.html")
    
    def start_turn(self, turn_number: int, battle_state_summary: Dict):
        """Start logging a new turn."""
        self.current_turn = {
            "turn": turn_number,
            "timestamp": datetime.now().isoformat(),
            "battle_state": battle_state_summary,
            "context_given_to_gpt": None,
            "gpt_responses": {},
            "selected_strategy": None,
            "final_decision": None,
            "target_validation": [],
            "submitted_order": None,
            "server_response": None,
            "turn_result": None
        }
    
    def log_context(self, context: Dict):
        """Log the context given to GPT."""
        if self.current_turn:
            self.current_turn["context_given_to_gpt"] = context
    
    def log_gpt_response(self, strategy: str, response: str, tokens_used: Dict = None):
        """Log a GPT response from one strategy."""
        if self.current_turn:
            try:
                parsed = json.loads(response) if response else None
            except:
                parsed = {"raw": response}
            
            self.current_turn["gpt_responses"][strategy] = {
                "raw": response,
                "parsed": parsed,
                "tokens": tokens_used
            }
    
    def log_selected_strategy(self, strategy: str, response: str):
        """Log which strategy was selected."""
        if self.current_turn:
            self.current_turn["selected_strategy"] = strategy
            try:
                self.current_turn["final_decision"] = json.loads(response)
            except:
                self.current_turn["final_decision"] = {"raw": response}
    
    def log_target_validation(self, slot: int, move: str, requested: int, validated: int, warning: str = None):
        """Log target validation result."""
        if self.current_turn:
            self.current_turn["target_validation"].append({
                "slot": slot,
                "move": move,
                "requested_target": requested,
                "validated_target": validated,
                "warning": warning
            })
    
    def log_submitted_order(self, order_str: str):
        """Log the final order submitted to server."""
        if self.current_turn:
            self.current_turn["submitted_order"] = order_str
    
    def log_server_response(self, events: List[str]):
        """Log relevant events from server response."""
        if self.current_turn:
            self.current_turn["server_response"] = events
    
    def log_turn_result(self, result_summary: Dict):
        """Log what happened as a result of this turn."""
        if self.current_turn:
            self.current_turn["turn_result"] = result_summary
    
    def end_turn(self):
        """Finish logging current turn and save."""
        if self.current_turn:
            self.turns.append(self.current_turn)
            self._save_json()
            self._print_turn_summary()
            self.current_turn = None
    
    def _save_json(self):
        """Save log to JSON file."""
        try:
            log_data = {
                "battle_tag": self.battle_tag,
                "start_time": self.start_time.isoformat(),
                "turns": self.turns
            }
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[LOG ERROR] Failed to save JSON: {e}")
    
    def _print_turn_summary(self):
        """Print a human-readable summary of the turn to console."""
        if not self.current_turn:
            return
        
        turn = self.current_turn
        
        print("\n" + "=" * 80)
        print(f"📍 TURN {turn['turn']} SUMMARY")
        print("=" * 80)
        
        # Battle State
        state = turn.get("battle_state", {})
        print(f"\n🎮 BATTLE STATE:")
        if "my_active" in state:
            for p in state["my_active"]:
                print(f"   MY: {p.get('species', '?')} ({p.get('hp_percent', '?')}%)")
        if "opp_active" in state:
            for p in state["opp_active"]:
                print(f"   OPP: {p.get('species', '?')} ({p.get('hp_percent', '?')}%)")
        
        # GPT Responses
        print(f"\n🤖 GPT RESPONSES:")
        for strategy, data in turn.get("gpt_responses", {}).items():
            status = "✓" if data.get("parsed") else "✗"
            print(f"   {status} {strategy}: {json.dumps(data.get('parsed', {}), ensure_ascii=False)[:100]}...")
        
        # Selected Decision
        print(f"\n✅ SELECTED: {turn.get('selected_strategy', 'none')}")
        decision = turn.get("final_decision", {})
        if decision:
            print(f"   Decision: {json.dumps(decision, ensure_ascii=False)}")
        
        # Target Validation
        validations = turn.get("target_validation", [])
        if validations:
            print(f"\n🎯 TARGET VALIDATION:")
            for v in validations:
                warning = f" ⚠️ {v['warning']}" if v.get('warning') else ""
                print(f"   Slot{v['slot']}: {v['move']} target {v['requested_target']}→{v['validated_target']}{warning}")
        
        # Submitted Order
        if turn.get("submitted_order"):
            print(f"\n📤 SUBMITTED: {turn['submitted_order']}")
        
        # Server Response
        if turn.get("server_response"):
            print(f"\n📥 SERVER EVENTS:")
            for event in turn["server_response"][:5]:  # First 5 events
                print(f"   {event}")
        
        print("=" * 80 + "\n")
    
    def generate_html_report(self):
        """Generate a detailed HTML report of the battle."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Battle Analysis: {self.battle_tag}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }}
        .turn {{ background: #16213e; border-radius: 8px; margin: 20px 0; padding: 15px; }}
        .turn-header {{ font-size: 1.5em; color: #00d9ff; border-bottom: 2px solid #00d9ff; padding-bottom: 10px; }}
        .section {{ margin: 15px 0; padding: 10px; background: #0f3460; border-radius: 5px; }}
        .section-title {{ color: #e94560; font-weight: bold; margin-bottom: 8px; }}
        .pokemon {{ display: inline-block; padding: 5px 10px; margin: 3px; border-radius: 4px; }}
        .pokemon.mine {{ background: #1e5128; }}
        .pokemon.opp {{ background: #5c1a1a; }}
        .decision {{ background: #2d4059; padding: 10px; border-left: 4px solid #00d9ff; }}
        .warning {{ color: #ff6b6b; }}
        .success {{ color: #6bff6b; }}
        pre {{ background: #0a0a0a; padding: 10px; border-radius: 4px; overflow-x: auto; font-size: 0.9em; }}
        .strategy {{ display: inline-block; padding: 3px 8px; border-radius: 3px; margin: 2px; }}
        .strategy.fast {{ background: #ff9f1c; color: black; }}
        .strategy.normal {{ background: #2ec4b6; color: black; }}
        .strategy.deep {{ background: #9b5de5; }}
        .selected {{ border: 2px solid #00ff00; }}
    </style>
</head>
<body>
    <h1>🎮 Battle Analysis: {self.battle_tag}</h1>
    <p>Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
"""
        
        for turn in self.turns:
            html += f"""
    <div class="turn">
        <div class="turn-header">Turn {turn['turn']}</div>
        
        <div class="section">
            <div class="section-title">🎮 Battle State</div>
"""
            state = turn.get("battle_state", {})
            for p in state.get("my_active", []):
                html += f'<span class="pokemon mine">{p.get("species", "?")} {p.get("hp_percent", "?")}%</span>'
            html += " vs "
            for p in state.get("opp_active", []):
                html += f'<span class="pokemon opp">{p.get("species", "?")} {p.get("hp_percent", "?")}%</span>'
            html += "</div>"
            
            # Context
            html += """
        <div class="section">
            <div class="section-title">📋 Context Given to GPT</div>
            <details>
                <summary>Click to expand</summary>
                <pre>""" + json.dumps(turn.get("context_given_to_gpt", {}), indent=2, ensure_ascii=False)[:3000] + """</pre>
            </details>
        </div>
"""
            
            # GPT Responses
            html += """
        <div class="section">
            <div class="section-title">🤖 GPT Strategy Responses</div>
"""
            selected = turn.get("selected_strategy", "")
            for strategy, data in turn.get("gpt_responses", {}).items():
                css_class = f"strategy {strategy}"
                if strategy == selected:
                    css_class += " selected"
                html += f'<span class="{css_class}">{strategy}</span>'
                html += f'<pre>{json.dumps(data.get("parsed", {}), indent=2, ensure_ascii=False)[:500]}</pre>'
            html += "</div>"
            
            # Decision
            html += f"""
        <div class="section">
            <div class="section-title">✅ Final Decision (Strategy: {selected})</div>
            <div class="decision">
                <pre>{json.dumps(turn.get("final_decision", {}), indent=2, ensure_ascii=False)}</pre>
            </div>
        </div>
"""
            
            # Target Validation
            validations = turn.get("target_validation", [])
            if validations:
                html += """
        <div class="section">
            <div class="section-title">🎯 Target Validation</div>
"""
                for v in validations:
                    css = "warning" if v.get("warning") else "success"
                    warning = f' - <span class="warning">{v["warning"]}</span>' if v.get("warning") else ""
                    html += f'<p class="{css}">Slot{v["slot"]}: {v["move"]} target {v["requested_target"]} → {v["validated_target"]}{warning}</p>'
                html += "</div>"
            
            # Submitted
            if turn.get("submitted_order"):
                html += f"""
        <div class="section">
            <div class="section-title">📤 Submitted Order</div>
            <pre>{turn["submitted_order"]}</pre>
        </div>
"""
            
            # Server Response
            if turn.get("server_response"):
                html += """
        <div class="section">
            <div class="section-title">📥 Server Response</div>
            <pre>""" + "\n".join(turn["server_response"][:10]) + """</pre>
        </div>
"""
            
            html += "</div>"
        
        html += """
</body>
</html>
"""
        
        try:
            with open(self.html_file, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"[LOG] HTML report saved: {self.html_file}")
        except Exception as e:
            print(f"[LOG ERROR] Failed to save HTML: {e}")


# =============================================================================
# BattleTurnLog - Dataclass for comprehensive turn-by-turn logging
# =============================================================================

@dataclass
class BattleTurnLog:
    """
    Comprehensive log of a single battle turn for debugging.
    Captures: GPT context, all responses, decision, validation, and outcome.
    """
    turn_number: int
    timestamp: str
    phase: str = "battle"  # "team_preview" or "battle"
    
    # Field state
    field_state: Dict[str, Any] = field(default_factory=dict)
    
    # Active Pokemon
    my_active: List[Dict[str, Any]] = field(default_factory=list)
    opponent_active: List[Dict[str, Any]] = field(default_factory=list)
    
    # GPT interaction
    gpt_context_summary: str = ""  # Summarized context sent to GPT
    gpt_responses: Dict[str, Any] = field(default_factory=dict)  # All strategy responses
    selected_strategy: str = ""  # Which strategy was selected
    
    # Decision
    final_decision: str = ""  # Final parsed decision
    decision_validation: Dict[str, Any] = field(default_factory=dict)  # Target validation log
    
    # Server interaction
    server_response: List[str] = field(default_factory=list)  # Server messages
    turn_outcome: Dict[str, Any] = field(default_factory=dict)  # What happened
    
    @staticmethod
    def format_pokemon_state(pokemon) -> Dict[str, Any]:
        """Format a Pokemon object for logging."""
        if pokemon is None:
            return {"species": "Empty", "hp_percent": 0, "fainted": True}
        
        return {
            "species": pokemon.species,
            "hp_percent": round(pokemon.current_hp_fraction * 100, 1) if hasattr(pokemon, 'current_hp_fraction') else 100,
            "status": str(pokemon.status) if pokemon.status else None,
            "fainted": pokemon.fainted if hasattr(pokemon, 'fainted') else False,
            "terastallized": pokemon.terastallized if hasattr(pokemon, 'terastallized') else False,
            "boosts": dict(pokemon.boosts) if hasattr(pokemon, 'boosts') and pokemon.boosts else {}
        }
    
    @staticmethod
    def format_field(battle) -> Dict[str, Any]:
        """Format field conditions for logging."""
        return {
            "weather": str(battle.weather) if battle.weather else None,
            "terrain": str(battle.terrain) if hasattr(battle, 'terrain') and battle.terrain else None,
            "fields": [str(f) for f in battle.fields] if battle.fields else [],
            "my_side": [str(c) for c in battle.side_conditions] if battle.side_conditions else [],
            "opp_side": [str(c) for c in battle.opponent_side_conditions] if battle.opponent_side_conditions else []
        }
    
    @staticmethod
    def format_move(move, target: int) -> str:
        """Format a move and target for readable logging."""
        target_name = {
            1: "opp_left",
            2: "opp_right",
            -1: "ally",
            -2: "self",
            0: "spread"
        }.get(target, f"unknown({target})")
        return f"{move.id} -> {target_name}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "turn": self.turn_number,
            "timestamp": self.timestamp,
            "phase": self.phase,
            "field": self.field_state,
            "my_active": self.my_active,
            "opponent_active": self.opponent_active,
            "gpt_context": self.gpt_context_summary[:500] + "..." if len(self.gpt_context_summary) > 500 else self.gpt_context_summary,
            "gpt_responses": self.gpt_responses,
            "selected_strategy": self.selected_strategy,
            "final_decision": self.final_decision,
            "validation": self.decision_validation,
            "server_response": self.server_response,
            "outcome": self.turn_outcome
        }
    
    def to_readable_log(self) -> str:
        """Generate human-readable log string."""
        lines = []
        lines.append(f"📍 TURN {self.turn_number} [{self.timestamp}]")
        lines.append("-" * 60)
        
        # Field state
        if self.field_state:
            field_parts = []
            if self.field_state.get("weather"):
                field_parts.append(f"Weather: {self.field_state['weather']}")
            if self.field_state.get("trick_room"):
                field_parts.append("🔄 TRICK ROOM")
            if self.field_state.get("my_side"):
                field_parts.append(f"My Side: {', '.join(self.field_state['my_side'])}")
            if self.field_state.get("opponent_side"):
                field_parts.append(f"Opp Side: {', '.join(self.field_state['opponent_side'])}")
            if field_parts:
                lines.append(f"🌍 Field: {' | '.join(field_parts)}")
        
        # Active Pokemon
        lines.append("\n🎮 ACTIVE POKEMON:")
        for p in self.my_active:
            tera = " ⭐TERA" if p.get("terastallized") else ""
            status = f" [{p['status']}]" if p.get("status") else ""
            lines.append(f"   MY: {p['species']} ({p['hp_percent']}%{status}{tera})")
        for p in self.opponent_active:
            tera = " ⭐TERA" if p.get("terastallized") else ""
            status = f" [{p['status']}]" if p.get("status") else ""
            lines.append(f"   OPP: {p['species']} ({p['hp_percent']}%{status}{tera})")
        
        # GPT responses summary
        lines.append("\n🤖 GPT STRATEGIES:")
        for strategy, resp in self.gpt_responses.items():
            status_icon = "✓" if resp.get("status") == "success" else "✗"
            decision_preview = str(resp.get("decision", ""))[:80]
            lines.append(f"   {status_icon} {strategy}: {decision_preview}")
        
        # Selected strategy & decision
        lines.append(f"\n✅ SELECTED: {self.selected_strategy}")
        lines.append(f"📝 DECISION: {self.final_decision}")
        
        # Validation warnings
        if self.decision_validation:
            validations = self.decision_validation.get("validation_actions", [])
            changed = [v for v in validations if v.get("changed")]
            if changed:
                lines.append("\n⚠️ TARGET CORRECTIONS:")
                for v in changed:
                    lines.append(f"   Slot{v['slot']} {v['move']}: {v['original_target']} → {v['validated_target']}")
        
        return "\n".join(lines)


# =============================================================================
# Battle State Tracking System
# =============================================================================

@dataclass
class MoveState:
    """Track a Pokemon's move state."""
    move_id: str
    name: str
    pp_remaining: Optional[int] = None  # None if unknown
    max_pp: int = 0
    used_count: int = 0  # Times used this battle
    last_used_turn: Optional[int] = None
    probability: float = 100.0  # Probability this pokemon has this move (for opponent)
    is_confirmed: bool = False  # True if observed in battle
    
    @classmethod
    def confirmed(cls, move_id: str, name: str, max_pp: int = 0, turn: int = 0) -> "MoveState":
        return cls(
            move_id=move_id, name=name, max_pp=max_pp,
            used_count=1, last_used_turn=turn, is_confirmed=True
        )
    
    @classmethod
    def predicted(cls, move_id: str, name: str, probability: float = 50.0) -> "MoveState":
        return cls(move_id=move_id, name=name, probability=probability, is_confirmed=False)
    
    def record_use(self, turn: int):
        """Record this move was used."""
        self.used_count += 1
        self.last_used_turn = turn
        self.is_confirmed = True
        self.probability = 100.0
        if self.pp_remaining is not None and self.pp_remaining > 0:
            self.pp_remaining -= 1
    
    def to_dict(self) -> Dict:
        result = {
            "move": self.name,
            "used_count": self.used_count,
            "confirmed": self.is_confirmed
        }
        if not self.is_confirmed:
            result["probability"] = f"{self.probability:.0f}%"
        if self.pp_remaining is not None:
            result["pp"] = f"{self.pp_remaining}/{self.max_pp}"
        return result


@dataclass
class PokemonState:
    """
    Track comprehensive state for a single Pokemon throughout battle.
    Works for both my team and opponent team.
    """
    # Identity
    species: str
    nickname: Optional[str] = None
    
    # Battle participation status
    selected_for_battle: bool = False  # Chosen during team preview
    is_lead: bool = False  # Started on field
    is_active: bool = False  # Currently on field
    slot: Optional[int] = None  # Current field slot (1 or 2 for doubles)
    fainted: bool = False
    
    # HP tracking
    hp_percent: float = 100.0  # Current HP percentage
    hp_exact: Optional[int] = None  # Exact HP if known (my pokemon)
    max_hp: Optional[int] = None  # Max HP if known
    
    # Status
    status: Optional[str] = None  # brn, par, slp, frz, psn, tox
    volatile_status: List[str] = field(default_factory=list)  # confusion, taunt, etc.
    
    # Stats & Boosts
    stat_boosts: Dict[str, int] = field(default_factory=lambda: {
        "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0, "accuracy": 0, "evasion": 0
    })
    
    # Moves tracking
    moves: Dict[str, MoveState] = field(default_factory=dict)  # move_id -> MoveState
    
    # Tera
    tera_type: Optional[str] = None  # Current tera type if terastallized
    has_terastallized: bool = False
    
    # Items & Ability (for opponent inference)
    item: Optional[str] = None  # Confirmed item
    item_consumed: bool = False
    item_effect_observed: Optional[str] = None  # "life_orb_recoil", "focus_sash", etc.
    ability: Optional[str] = None  # Confirmed ability
    ability_effect_observed: Optional[str] = None  # "intimidate_activated", etc.
    
    # =========================================================================
    # Stat Range Tracking (for opponent inference)
    # =========================================================================
    # Actual stat ranges at Lv50 (min/max based on observations)
    stat_ranges: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        "hp": {"min": 0, "max": 999},
        "atk": {"min": 0, "max": 999},
        "def": {"min": 0, "max": 999},
        "spa": {"min": 0, "max": 999},
        "spd": {"min": 0, "max": 999},
        "spe": {"min": 0, "max": 999}
    })
    
    # EV allocation tracking (508 total limit)
    ev_estimates: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        "hp": {"min": 0, "max": 252},
        "atk": {"min": 0, "max": 252},
        "def": {"min": 0, "max": 252},
        "spa": {"min": 0, "max": 252},
        "spd": {"min": 0, "max": 252},
        "spe": {"min": 0, "max": 252}
    })
    ev_used_min: int = 0  # Minimum EVs definitely used
    ev_used_max: int = 508  # Maximum EVs possibly used
    
    # Nature inference
    nature_candidates: List[Dict] = field(default_factory=list)  # [{"nature": "Jolly", "probability": 80}]
    
    # =========================================================================
    # Speed Order Tracking
    # =========================================================================
    # Relative speed observations: [{"faster_than": "Incineroar", "turn": 1, "context": "moved first"}]
    speed_relations: List[Dict] = field(default_factory=list)
    # Known speed tier bracket
    speed_tier: Optional[str] = None  # "fast", "medium", "slow", "trick_room"
    
    # Base speed stat (from pokedex, for reference)
    base_speed: Optional[int] = None
    
    # Confirmed/inferred actual speed stat at Lv50 (before modifiers)
    # This is the "real" speed value used for calculations
    # Modifiers like Tailwind (2x), Paralysis (0.5x), Choice Scarf (1.5x) are applied separately
    actual_speed: Optional[int] = None  # Confirmed actual speed if known
    actual_speed_range: Dict[str, int] = field(default_factory=lambda: {"min": 0, "max": 999})
    
    # =========================================================================
    # Damage Observations (for stat inference)
    # =========================================================================
    # [{"turn": 1, "move": "Close Combat", "target": "Incineroar", "damage_percent": 85, "crit": False}]
    damage_dealt: List[Dict] = field(default_factory=list)
    damage_received: List[Dict] = field(default_factory=list)
    
    # Turn tracking
    turns_on_field: int = 0  # Total turns spent on field
    last_action_turn: Optional[int] = None
    last_action: Optional[str] = None  # Last action taken
    
    def set_active(self, slot: int):
        """Pokemon switches in."""
        self.is_active = True
        self.slot = slot
        
    def set_inactive(self):
        """Pokemon switches out."""
        self.is_active = False
        self.slot = None
        # Clear volatile status on switch
        self.volatile_status = []
        # Reset stat boosts
        self.stat_boosts = {k: 0 for k in self.stat_boosts}
    
    def initialize_stat_ranges_from_base(self, base_stats: Dict[str, int], level: int = 50):
        """
        Initialize stat ranges based on base stats.
        
        VGC Lv50 stat formula:
        HP:    floor((2*base + IV + floor(EV/4)) * level/100 + level + 10)
        Other: floor((floor((2*base + IV + floor(EV/4)) * level/100) + 5) * nature)
        
        With EV 0, IV 31:   floor((2*base + 31) * 50/100) + 5 = base + 20
        With EV 252, IV 31: floor((2*base + 31 + 63) * 50/100) + 5 = base + 52
        
        Args:
            base_stats: {"hp": X, "atk": X, ...}
            level: Pokemon level (50 for VGC)
        """
        if not base_stats:
            return
        
        for stat in ["hp", "atk", "def", "spa", "spd", "spe"]:
            base = base_stats.get(stat, 0)
            if base == 0:
                continue
            
            if stat == "hp":
                # HP formula: floor((2*base + IV + floor(EV/4)) * level/100 + level + 10)
                # Min (0 EV, 0 IV): floor((2*base + 0 + 0) * 50/100) + 50 + 10 = base + 60
                # Max (252 EV, 31 IV): floor((2*base + 31 + 63) * 50/100) + 50 + 10 = base + 107
                min_hp = int((2 * base + 0 + 0) * level / 100) + level + 10
                max_hp = int((2 * base + 31 + 63) * level / 100) + level + 10
                self.stat_ranges["hp"]["min"] = min_hp
                self.stat_ranges["hp"]["max"] = max_hp
            else:
                # Other stats: floor((floor((2*base + IV + floor(EV/4)) * level/100) + 5) * nature)
                # Min (0 EV, 0 IV, -nature): floor((base + 5) * 0.9) 
                # Max (252 EV, 31 IV, +nature): floor((base + 52) * 1.1)
                min_stat = int((int((2 * base + 0 + 0) * level / 100) + 5) * 0.9)
                max_stat = int((int((2 * base + 31 + 63) * level / 100) + 5) * 1.1)
                self.stat_ranges[stat]["min"] = min_stat
                self.stat_ranges[stat]["max"] = max_stat
                
                # Also set speed-specific fields
                if stat == "spe":
                    self.base_speed = base
                    self.actual_speed_range["min"] = min_stat
                    self.actual_speed_range["max"] = max_stat
    
    def get_speed_range_at_level(self, base_speed: int, level: int = 50,
                                  assume_invested: bool = False,
                                  assume_positive_nature: bool = False) -> Dict[str, int]:
        """
        Calculate speed range from base speed with assumptions.
        
        Args:
            base_speed: Base speed stat
            level: Pokemon level
            assume_invested: If True, assume 252 EVs
            assume_positive_nature: If True, assume +Spe nature
            
        Returns:
            {"min": X, "max": X} speed range
        """
        # No investment: (2*base + 31) * 50/100 + 5 = base + 20 (neutral nature)
        # Full investment: (2*base + 31 + 63) * 50/100 + 5 = base + 52 (neutral nature)
        
        if assume_invested:
            ev = 252
        else:
            ev = 0 if not assume_invested else 252
        
        iv = 31  # Assume competitive IVs
        
        base_calc = int((2 * base_speed + iv + int(ev / 4)) * level / 100) + 5
        
        if assume_positive_nature:
            return {"min": int(base_calc * 1.1), "max": int(base_calc * 1.1)}
        else:
            return {
                "min": int(base_calc * 0.9),  # -nature
                "max": int(base_calc * 1.1)   # +nature
            }
    
    def update_hp(self, hp_percent: float, hp_exact: Optional[int] = None):
        """Update HP."""
        self.hp_percent = hp_percent
        if hp_exact is not None:
            self.hp_exact = hp_exact
        if hp_percent <= 0:
            self.fainted = True
            self.is_active = False
    
    def add_move(self, move_id: str, name: str, turn: int, max_pp: int = 0):
        """Record a move usage."""
        if move_id in self.moves:
            self.moves[move_id].record_use(turn)
        else:
            self.moves[move_id] = MoveState.confirmed(move_id, name, max_pp, turn)
    
    def add_predicted_move(self, move_id: str, name: str, probability: float):
        """Add a predicted move for opponent."""
        if move_id not in self.moves:
            self.moves[move_id] = MoveState.predicted(move_id, name, probability)
        elif not self.moves[move_id].is_confirmed:
            # Update probability if higher
            if probability > self.moves[move_id].probability:
                self.moves[move_id].probability = probability
    
    def add_predicted_moves_from_usage(self, usage_moves: List[Dict]):
        """
        Add predicted moves from usage statistics.
        usage_moves: [{"move": "Fake Out", "usage": 85.5}, ...]
        """
        for move_data in usage_moves:
            move_name = move_data.get("move", "")
            usage = move_data.get("usage", 0)
            if move_name and usage > 0:
                move_id = move_name.lower().replace(" ", "").replace("-", "")
                self.add_predicted_move(move_id, move_name, usage)
    
    def get_top_moves(self, limit: int = 10) -> List[MoveState]:
        """
        Get top N moves by priority:
        1. Confirmed moves (is_confirmed=True) - always included first
        2. Predicted moves sorted by probability (descending)
        """
        confirmed = [m for m in self.moves.values() if m.is_confirmed]
        predicted = [m for m in self.moves.values() if not m.is_confirmed]
        
        # Sort predicted by probability descending
        predicted.sort(key=lambda x: x.probability, reverse=True)
        
        # Confirmed first, then top predicted to fill up to limit
        result = confirmed.copy()
        remaining_slots = limit - len(confirmed)
        if remaining_slots > 0:
            result.extend(predicted[:remaining_slots])
        
        return result
    
    def get_moves_for_gpt(self, limit: int = 10) -> List[Dict]:
        """Get top moves formatted for GPT context."""
        top_moves = self.get_top_moves(limit)
        return [m.to_dict() for m in top_moves]
    
    def apply_stat_boost(self, stat: str, stages: int):
        """Apply stat boost."""
        if stat in self.stat_boosts:
            new_val = self.stat_boosts[stat] + stages
            self.stat_boosts[stat] = max(-6, min(6, new_val))
    
    # =========================================================================
    # Observation Recording Methods
    # =========================================================================
    
    def record_speed_relation(self, other_species: str, is_faster: bool, turn: int, 
                               context: str = "", priority_diff: int = 0):
        """
        Record a speed relation observation.
        
        Args:
            other_species: Species compared against
            is_faster: True if this pokemon moved first
            turn: Turn number
            context: Additional context (e.g., "tailwind active", "trick room")
            priority_diff: Priority difference (0 if same priority moves)
        """
        if priority_diff != 0:
            # Different priority, can't infer speed
            return
            
        relation = {
            "faster_than" if is_faster else "slower_than": other_species,
            "turn": turn,
            "context": context
        }
        
        # Avoid duplicates
        for existing in self.speed_relations:
            key = "faster_than" if is_faster else "slower_than"
            if existing.get(key) == other_species:
                return
        
        self.speed_relations.append(relation)
    
    def record_damage_dealt(self, move: str, target: str, damage_percent: float,
                            crit: bool, turn: int, target_hp_before: float = 100.0,
                            weather: str = None, terrain: str = None,
                            attacker_boosts: Dict = None, defender_boosts: Dict = None):
        """
        Record damage dealt for stat inference.
        
        Args:
            move: Move used
            target: Target pokemon species
            damage_percent: Damage dealt as percentage
            crit: Whether it was a critical hit
            turn: Turn number
            target_hp_before: Target's HP% before attack
            weather: Active weather (if any)
            terrain: Active terrain (if any)
            attacker_boosts: Attacker's stat boosts
            defender_boosts: Defender's stat boosts
        """
        self.damage_dealt.append({
            "turn": turn,
            "move": move,
            "target": target,
            "damage_percent": damage_percent,
            "crit": crit,
            "target_hp_before": target_hp_before,
            "weather": weather,
            "terrain": terrain,
            "attacker_boosts": attacker_boosts or {},
            "defender_boosts": defender_boosts or {}
        })
    
    def record_damage_received(self, move: str, attacker: str, damage_percent: float,
                                crit: bool, turn: int, my_hp_before: float = 100.0,
                                weather: str = None, terrain: str = None,
                                attacker_boosts: Dict = None, defender_boosts: Dict = None):
        """
        Record damage received for stat inference.
        """
        self.damage_received.append({
            "turn": turn,
            "move": move,
            "attacker": attacker,
            "damage_percent": damage_percent,
            "crit": crit,
            "my_hp_before": my_hp_before,
            "weather": weather,
            "terrain": terrain,
            "attacker_boosts": attacker_boosts or {},
            "defender_boosts": defender_boosts or {}
        })
    
    def record_item_effect(self, effect: str, turn: int):
        """
        Record observed item effect.
        
        Args:
            effect: Effect type (e.g., "life_orb_recoil", "focus_sash", 
                   "choice_locked", "leftovers_heal", "sitrus_heal")
            turn: Turn observed
        """
        self.item_effect_observed = effect
        
        # Infer item from effect
        item_map = {
            "life_orb_recoil": "Life Orb",
            "focus_sash": "Focus Sash",
            "choice_locked": None,  # Could be band/specs/scarf
            "leftovers_heal": "Leftovers",
            "sitrus_heal": "Sitrus Berry",
            "berry_consumed": None,  # Generic berry
            "assault_vest": "Assault Vest",  # From failed status move
            "air_balloon": "Air Balloon",
            "balloon_popped": "Air Balloon",
            "eviolite": "Eviolite",  # Inferred from bulk
        }
        
        if effect in item_map and item_map[effect]:
            self.item = item_map[effect]
    
    def record_ability_effect(self, effect: str, turn: int):
        """
        Record observed ability effect.
        
        Args:
            effect: Effect type (e.g., "intimidate_activated", "levitate_immune")
        """
        self.ability_effect_observed = effect
        
        # Infer ability from effect
        ability_map = {
            "intimidate_activated": "Intimidate",
            "levitate_immune": "Levitate",
            "flash_fire_immune": "Flash Fire",
            "water_absorb_heal": "Water Absorb",
            "volt_absorb_heal": "Volt Absorb",
            "lightning_rod_immune": "Lightning Rod",
            "storm_drain_immune": "Storm Drain",
            "magic_bounce": "Magic Bounce",
            "clear_body_blocked": "Clear Body",
            "defiant_activated": "Defiant",
            "competitive_activated": "Competitive",
            "beast_boost": "Beast Boost",
            "moxie_activated": "Moxie",
            "protean_activated": "Protean",
            "libero_activated": "Libero",
        }
        
        if effect in ability_map:
            self.ability = ability_map[effect]
    
    # =========================================================================
    # Stat Range Update Methods
    # =========================================================================
    
    def update_stat_range(self, stat: str, new_min: int = None, new_max: int = None,
                          evidence: str = None):
        """
        Update stat range based on observation.
        
        Args:
            stat: Stat name (hp, atk, def, spa, spd, spe)
            new_min: New minimum value (if known)
            new_max: New maximum value (if known)
            evidence: Description of evidence
        """
        if stat not in self.stat_ranges:
            return
        
        if new_min is not None:
            self.stat_ranges[stat]["min"] = max(self.stat_ranges[stat]["min"], new_min)
        if new_max is not None:
            self.stat_ranges[stat]["max"] = min(self.stat_ranges[stat]["max"], new_max)
        
        # Ensure min <= max
        if self.stat_ranges[stat]["min"] > self.stat_ranges[stat]["max"]:
            # Contradiction - reset to last known good values
            pass
    
    def update_speed_from_relation(self, other_species: str, other_speed: int, 
                                    is_faster: bool, paralysis: bool = False,
                                    tailwind_self: bool = False, tailwind_opp: bool = False,
                                    trick_room: bool = False):
        """
        Update speed range based on speed relation observation.
        
        Args:
            other_species: Species compared against
            other_speed: Other pokemon's speed value
            is_faster: Whether this pokemon moved first
            paralysis: If this pokemon is paralyzed (0.5x speed)
            tailwind_self: If tailwind is active for this pokemon (2x speed)
            tailwind_opp: If tailwind is active for opponent
            trick_room: If trick room is active (slower goes first)
        """
        # Calculate effective speeds
        other_effective = other_speed
        if tailwind_opp:
            other_effective *= 2
        
        multiplier = 1.0
        if paralysis:
            multiplier *= 0.5
        if tailwind_self:
            multiplier *= 2
        
        if trick_room:
            # In trick room, slower wins
            if is_faster:
                # We moved first = we are slower = our speed < other
                new_max = int(other_effective / multiplier) - 1
                self.update_stat_range("spe", new_max=new_max)
            else:
                # We moved second = we are faster = our speed > other
                new_min = int(other_effective / multiplier) + 1
                self.update_stat_range("spe", new_min=new_min)
        else:
            # Normal: faster wins
            if is_faster:
                # We moved first = we are faster = our speed > other
                new_min = int(other_effective / multiplier) + 1
                self.update_stat_range("spe", new_min=new_min)
            else:
                # We moved second = we are slower = our speed < other
                new_max = int(other_effective / multiplier) - 1
                self.update_stat_range("spe", new_max=new_max)
    
    def update_ev_estimate(self, stat: str, min_ev: int = None, max_ev: int = None):
        """
        Update EV estimate for a stat, respecting 508 total and 252 per stat limits.
        
        Args:
            stat: Stat name
            min_ev: Minimum EVs in this stat
            max_ev: Maximum EVs in this stat
        """
        if stat not in self.ev_estimates:
            return
        
        # Per-stat cap is 252
        if min_ev is not None:
            self.ev_estimates[stat]["min"] = max(0, min(252, min_ev))
        if max_ev is not None:
            self.ev_estimates[stat]["max"] = max(0, min(252, max_ev))
        
        # Update total EV tracking
        self._recalculate_ev_totals()
    
    def _recalculate_ev_totals(self):
        """Recalculate total EV bounds based on individual stat estimates."""
        # Sum of all minimums = minimum EVs definitely used
        self.ev_used_min = sum(self.ev_estimates[stat]["min"] for stat in self.ev_estimates)
        
        # If we've confirmed high EVs in some stats, other stats have less room
        # Maximum possible = min(508 - other_mins, sum of maxes)
        total_max = min(508, sum(self.ev_estimates[stat]["max"] for stat in self.ev_estimates))
        
        # Update individual maxes based on 508 constraint
        for stat in self.ev_estimates:
            other_mins = sum(
                self.ev_estimates[s]["min"] for s in self.ev_estimates if s != stat
            )
            remaining = 508 - other_mins
            self.ev_estimates[stat]["max"] = min(self.ev_estimates[stat]["max"], remaining)
        
        self.ev_used_max = min(508, total_max)
    
    def infer_speed_ev_from_stat(self, base_speed: int, known_speed: int, 
                                  level: int = 50, iv: int = 31):
        """
        Infer speed EV from known speed stat.
        
        Formula at Lv50: stat = floor((2*base + IV + floor(EV/4)) * level/100 + 5) * nature
        Solving for EV: EV = (((stat / nature) - 5) * 100 / level - 2*base - IV) * 4
        
        Args:
            base_speed: Base speed stat
            known_speed: Known actual speed
            level: Pokemon level (default 50 for VGC)
            iv: Assumed IV (default 31)
        """
        # Try different natures
        for nature_mult, nature_type in [(1.1, "positive"), (1.0, "neutral"), (0.9, "negative")]:
            try:
                # Reverse calculate EV
                stat_pre_nature = known_speed / nature_mult
                ev = ((stat_pre_nature - 5) * 100 / level - 2 * base_speed - iv) * 4
                
                if 0 <= ev <= 252:
                    # Valid EV found
                    ev_int = int(round(ev))
                    self.update_ev_estimate("spe", min_ev=max(0, ev_int - 4), 
                                           max_ev=min(252, ev_int + 4))
                    
                    # Update nature candidates
                    if nature_type == "positive":
                        self.nature_candidates.append({"nature": "+Spe nature", "probability": 70})
                    elif nature_type == "negative":
                        self.nature_candidates.append({"nature": "-Spe nature", "probability": 50})
                    return
            except:
                pass
    
    def get_speed_summary(self) -> Dict:
        """Get summary of speed information for GPT context."""
        summary = {
            "range": self.stat_ranges["spe"],
            "ev_range": self.ev_estimates["spe"]
        }
        
        # Include actual speed if known/narrowed
        if self.actual_speed:
            summary["actual_speed"] = self.actual_speed
        elif self.actual_speed_range["min"] > 0 or self.actual_speed_range["max"] < 999:
            summary["actual_speed_range"] = self.actual_speed_range
        
        if self.speed_relations:
            summary["relations"] = self.speed_relations
        
        if self.speed_tier:
            summary["tier"] = self.speed_tier
        
        return summary
    
    def calculate_effective_speed(
        self, 
        tailwind: bool = False,
        paralysis: bool = False,
        choice_scarf: bool = False,
        speed_boost_stages: int = 0,
        trick_room: bool = False
    ) -> Dict[str, int]:
        """
        Calculate effective speed with modifiers.
        
        Modifiers stack multiplicatively:
        - Tailwind: 2x
        - Paralysis: 0.5x (0.25x in some gens, 0.5x in gen 7+)
        - Choice Scarf: 1.5x
        - Speed stages: +1 = 1.5x, +2 = 2x, -1 = 0.67x, etc.
        - Trick Room: Doesn't change value, but inverts comparison
        
        Args:
            tailwind: Tailwind active on this Pokemon's side
            paralysis: Pokemon is paralyzed
            choice_scarf: Pokemon is holding Choice Scarf
            speed_boost_stages: Current speed stat stages (-6 to +6)
            trick_room: Trick Room active (for context, doesn't change value)
        
        Returns:
            {"min": X, "max": X, "trick_room": bool} effective speed range
        """
        # Get base speed range
        base_min = self.actual_speed_range["min"] if self.actual_speed_range["min"] > 0 else self.stat_ranges["spe"]["min"]
        base_max = self.actual_speed_range["max"] if self.actual_speed_range["max"] < 999 else self.stat_ranges["spe"]["max"]
        
        if self.actual_speed:
            base_min = base_max = self.actual_speed
        
        # Calculate multiplier
        multiplier = 1.0
        
        if tailwind:
            multiplier *= 2.0
        
        if paralysis:
            multiplier *= 0.5  # Gen 7+ paralysis speed
        
        if choice_scarf:
            multiplier *= 1.5
        
        # Stat stages: +1 = 1.5x, +2 = 2x, +3 = 2.5x, etc. / -1 = 0.67x, -2 = 0.5x
        if speed_boost_stages != 0:
            if speed_boost_stages > 0:
                stage_mult = (2 + speed_boost_stages) / 2
            else:
                stage_mult = 2 / (2 - speed_boost_stages)
            multiplier *= stage_mult
        
        return {
            "min": int(base_min * multiplier),
            "max": int(base_max * multiplier),
            "trick_room": trick_room
        }
    
    def compare_speed(
        self, 
        other_speed: int,
        my_modifiers: Dict = None,
        other_modifiers: Dict = None,
        trick_room: bool = False
    ) -> str:
        """
        Compare speed with another Pokemon.
        
        Args:
            other_speed: Other Pokemon's actual speed stat
            my_modifiers: {"tailwind": bool, "paralysis": bool, "scarf": bool, "stages": int}
            other_modifiers: Same structure for opponent
            trick_room: Trick Room active
            
        Returns:
            "faster" | "slower" | "speed_tie" | "uncertain"
        """
        my_mods = my_modifiers or {}
        other_mods = other_modifiers or {}
        
        # Calculate my effective speed
        my_eff = self.calculate_effective_speed(
            tailwind=my_mods.get("tailwind", False),
            paralysis=my_mods.get("paralysis", False) or self.status == "par",
            choice_scarf=my_mods.get("scarf", False),
            speed_boost_stages=my_mods.get("stages", self.stat_boosts.get("spe", 0))
        )
        
        # Calculate other's effective speed
        other_mult = 1.0
        if other_mods.get("tailwind"):
            other_mult *= 2.0
        if other_mods.get("paralysis"):
            other_mult *= 0.5
        if other_mods.get("scarf"):
            other_mult *= 1.5
        stages = other_mods.get("stages", 0)
        if stages > 0:
            other_mult *= (2 + stages) / 2
        elif stages < 0:
            other_mult *= 2 / (2 - stages)
        
        other_eff = int(other_speed * other_mult)
        
        # Compare (trick room inverts)
        if trick_room:
            # In trick room, lower speed wins
            if my_eff["max"] < other_eff:
                return "faster"  # We're slower = we go first in TR
            elif my_eff["min"] > other_eff:
                return "slower"  # We're faster = we go second in TR
            elif my_eff["min"] == my_eff["max"] == other_eff:
                return "speed_tie"
            else:
                return "uncertain"
        else:
            # Normal: higher speed wins
            if my_eff["min"] > other_eff:
                return "faster"
            elif my_eff["max"] < other_eff:
                return "slower"
            elif my_eff["min"] == my_eff["max"] == other_eff:
                return "speed_tie"
            else:
                return "uncertain"
    
    def get_stat_inference_summary(self) -> Dict:
        """Get summary of stat inferences for GPT context."""
        summary = {}
        
        # Only include stats with meaningful constraints
        for stat in self.stat_ranges:
            if (self.stat_ranges[stat]["min"] > 0 or 
                self.stat_ranges[stat]["max"] < 999):
                summary[stat] = {
                    "range": self.stat_ranges[stat],
                    "ev_range": self.ev_estimates[stat]
                }
        
        if self.damage_dealt:
            summary["damage_dealt_count"] = len(self.damage_dealt)
        if self.damage_received:
            summary["damage_received_count"] = len(self.damage_received)
        
        return summary
    
    def to_dict(self, move_limit: int = 10) -> Dict:
        """
        Convert to dictionary for GPT context.
        
        Args:
            move_limit: Max number of moves to include (default 10).
                       Confirmed moves are always included first.
        """
        result = {
            "species": self.species,
            "hp": f"{self.hp_percent:.0f}%",
            "status": self.status or "healthy",
        }
        
        if self.is_active:
            result["active"] = True
            result["slot"] = self.slot
        
        if self.fainted:
            result["fainted"] = True
        
        # Add non-zero boosts
        boosts = {k: v for k, v in self.stat_boosts.items() if v != 0}
        if boosts:
            result["stat_boosts"] = boosts
        
        if self.volatile_status:
            result["volatile_status"] = self.volatile_status
        
        if self.has_terastallized:
            result["terastallized"] = self.tera_type
        
        if self.item:
            result["item"] = self.item
            if self.item_consumed:
                result["item_consumed"] = True
        
        if self.ability:
            result["ability"] = self.ability
        
        # Observed effects (for opponent inference)
        if self.item_effect_observed:
            result["item_effect_observed"] = self.item_effect_observed
        if self.ability_effect_observed:
            result["ability_effect_observed"] = self.ability_effect_observed
        
        # Speed information (critical for VGC decisions)
        speed_info = self.get_speed_summary()
        if speed_info.get("relations") or speed_info.get("tier"):
            result["speed_info"] = speed_info
        
        # Stat inferences (if any meaningful constraints exist)
        stat_inferences = self.get_stat_inference_summary()
        if stat_inferences:
            result["stat_inferences"] = stat_inferences
        
        # Nature candidates (if inferred)
        if self.nature_candidates:
            result["nature_candidates"] = self.nature_candidates
        
        # Moves summary - TOP N only (confirmed first, then by probability)
        if self.moves:
            result["known_moves"] = self.get_moves_for_gpt(move_limit)
        
        return result


@dataclass
class TeamState:
    """Track state for an entire team."""
    pokemon: Dict[str, PokemonState] = field(default_factory=dict)  # species -> PokemonState
    tera_used: bool = False  # Team has used tera this battle
    
    def get_active(self) -> List[PokemonState]:
        """Get currently active pokemon."""
        return [p for p in self.pokemon.values() if p.is_active and not p.fainted]
    
    def get_selected(self) -> List[PokemonState]:
        """Get pokemon selected for battle."""
        return [p for p in self.pokemon.values() if p.selected_for_battle]
    
    def get_available(self) -> List[PokemonState]:
        """Get pokemon that can still battle (selected, not fainted, not active)."""
        return [p for p in self.pokemon.values() 
                if p.selected_for_battle and not p.fainted and not p.is_active]
    
    def add_pokemon(self, species: str, base_stats: Dict[str, int] = None, 
                    **kwargs) -> PokemonState:
        """
        Add a pokemon to track.
        
        Args:
            species: Pokemon species name
            base_stats: Base stats dict for stat range initialization
            **kwargs: Additional PokemonState fields
        """
        poke = PokemonState(species=species, **kwargs)
        
        # Initialize stat ranges if base stats provided
        if base_stats:
            poke.initialize_stat_ranges_from_base(base_stats, level=50)
        
        self.pokemon[species] = poke
        return poke
    
    def get(self, species: str) -> Optional[PokemonState]:
        """Get pokemon by species name."""
        return self.pokemon.get(species)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for GPT context."""
        active = [p.to_dict() for p in self.get_active()]
        bench = [p.to_dict() for p in self.get_available()]
        fainted = [p.species for p in self.pokemon.values() if p.fainted]
        
        return {
            "active": active,
            "bench": bench,
            "fainted": fainted,
            "tera_used": self.tera_used
        }


@dataclass
class StrategyContext:
    """Track strategic context throughout battle."""
    # Team preview decisions
    team_analysis: Optional[Dict] = None  # Full team analysis from preview
    my_strategy: Optional[Dict] = None  # Our planned strategy
    opponent_predictions: Optional[Dict] = None  # Predicted opponent strategy
    
    # Per-turn strategy notes
    turn_plans: Dict[int, Dict] = field(default_factory=dict)  # turn -> plan
    
    # Strategy adjustments
    adjustments: List[Dict] = field(default_factory=list)  # List of strategy changes
    
    # Key observations
    confirmed_threats: List[str] = field(default_factory=list)  # Confirmed threat pokemon
    neutralized_threats: List[str] = field(default_factory=list)  # Threats we've handled
    
    # Win condition tracking
    initial_win_condition: Optional[str] = None
    current_win_condition: Optional[str] = None
    win_condition_changes: List[Dict] = field(default_factory=list)
    
    def set_initial_strategy(self, analysis: Dict):
        """Set initial strategy from team preview."""
        self.team_analysis = analysis
        if "my_strategy" in analysis:
            self.my_strategy = analysis["my_strategy"]
            self.initial_win_condition = analysis.get("my_strategy", {}).get("game_plan")
            self.current_win_condition = self.initial_win_condition
        if "opponent_analysis" in analysis:
            self.opponent_predictions = analysis["opponent_analysis"]
    
    def record_turn_plan(self, turn: int, plan: Dict):
        """Record the plan for a turn."""
        self.turn_plans[turn] = plan
    
    def add_adjustment(self, turn: int, reason: str, new_focus: str):
        """Record a strategy adjustment."""
        self.adjustments.append({
            "turn": turn,
            "reason": reason,
            "new_focus": new_focus,
            "previous_win_condition": self.current_win_condition
        })
        self.current_win_condition = new_focus
        self.win_condition_changes.append({
            "turn": turn,
            "from": self.current_win_condition,
            "to": new_focus
        })
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for GPT context."""
        return {
            "initial_strategy": self.my_strategy,
            "current_win_condition": self.current_win_condition,
            "adjustments_made": len(self.adjustments),
            "recent_adjustments": self.adjustments[-3:] if self.adjustments else [],
            "confirmed_threats": self.confirmed_threats,
            "neutralized_threats": self.neutralized_threats
        }


@dataclass
class BattleState:
    """
    Master battle state tracker.
    Persists throughout the entire battle, updating as observations come in.
    """
    battle_tag: str
    format: str
    
    # Team states
    my_team: TeamState = field(default_factory=TeamState)
    opponent_team: TeamState = field(default_factory=TeamState)
    
    # Strategic context
    strategy: StrategyContext = field(default_factory=StrategyContext)
    
    # Battle progress
    current_turn: int = 0
    phase: str = "team_preview"  # team_preview, battle, ended
    
    # Field conditions
    weather: Optional[str] = None
    weather_turns: int = 0
    terrain: Optional[str] = None
    terrain_turns: int = 0
    trick_room: bool = False
    trick_room_turns: int = 0
    
    # Side conditions (my side)
    my_side_conditions: Dict[str, int] = field(default_factory=dict)  # condition -> turns remaining
    # Side conditions (opponent side)
    opp_side_conditions: Dict[str, int] = field(default_factory=dict)
    
    # Turn-by-turn history (for context)
    turn_history: List[Dict] = field(default_factory=list)
    
    # Inference data (opponent pokemon detail tracking)
    opponent_inferred: Dict[str, "PokemonInferredInfo"] = field(default_factory=dict)
    
    # ==========================================================================
    # Team Preview Cache - stores data collected during team preview
    # Used to avoid redundant tool calls during battle phase
    # ==========================================================================
    team_preview_cache: TeamPreviewCache = field(default_factory=TeamPreviewCache)
    
    def cache_pokemon_data(self, pokemon: str, data_type: str, data: Any) -> None:
        """
        Store data collected during team preview for later use.
        
        Args:
            pokemon: Pokemon name/species
            data_type: Type of data (usage_stats, base_stats, moves, items, speed_tier)
            data: The data to cache
        """
        self.team_preview_cache.store(pokemon, data_type, data)
    
    def get_cached_pokemon_data(self, pokemon: str, data_type: str) -> Optional[Any]:
        """Retrieve cached data for a Pokemon."""
        return self.team_preview_cache.get(pokemon, data_type)
    
    def has_cached_data(self, pokemon: str) -> bool:
        """Check if we have any cached data for this Pokemon."""
        return self.team_preview_cache.has_data(pokemon)
    
    def get_cache_summary(self) -> Dict[str, List[str]]:
        """Get summary of what's cached for debugging."""
        return self.team_preview_cache.summary()
    
    def advance_turn(self, turn: int):
        """Called at start of new turn."""
        self.current_turn = turn
        self.phase = "battle"
        
        # Decrement field condition turns
        if self.weather_turns > 0:
            self.weather_turns -= 1
            if self.weather_turns == 0:
                self.weather = None
        if self.terrain_turns > 0:
            self.terrain_turns -= 1
            if self.terrain_turns == 0:
                self.terrain = None
        if self.trick_room_turns > 0:
            self.trick_room_turns -= 1
            if self.trick_room_turns == 0:
                self.trick_room = False
        
        # Increment turn counts for active pokemon
        for poke in self.my_team.get_active():
            poke.turns_on_field += 1
        for poke in self.opponent_team.get_active():
            poke.turns_on_field += 1
    
    def record_turn_actions(self, turn: int, summary: Dict):
        """Record summary of what happened in a turn."""
        self.turn_history.append({
            "turn": turn,
            **summary
        })
        # Keep last 10 turns
        if len(self.turn_history) > 10:
            self.turn_history.pop(0)
    
    def set_weather(self, weather: str, turns: int = 5):
        """Set weather condition."""
        self.weather = weather
        self.weather_turns = turns
    
    def set_terrain(self, terrain: str, turns: int = 5):
        """Set terrain."""
        self.terrain = terrain
        self.terrain_turns = turns
    
    def set_trick_room(self, turns: int = 5):
        """Set trick room."""
        self.trick_room = True
        self.trick_room_turns = turns
    
    def get_field_summary(self) -> Dict:
        """Get current field conditions."""
        result = {}
        if self.weather:
            result["weather"] = f"{self.weather} ({self.weather_turns} turns)"
        if self.terrain:
            result["terrain"] = f"{self.terrain} ({self.terrain_turns} turns)"
        if self.trick_room:
            result["trick_room"] = f"active ({self.trick_room_turns} turns)"
        if self.my_side_conditions:
            result["my_side"] = self.my_side_conditions
        if self.opp_side_conditions:
            result["opponent_side"] = self.opp_side_conditions
        return result
    
    def to_gpt_context(self) -> Dict:
        """Generate full context dictionary for GPT."""
        context = {
            "turn": self.current_turn,
            "phase": self.phase,
            "field": self.get_field_summary(),
            "my_team": self.my_team.to_dict(),
            "opponent_team": self.opponent_team.to_dict(),
            "strategy": self.strategy.to_dict(),
            "recent_turns": self.turn_history[-5:]  # Last 5 turns
        }
        
        # Add speed tier analysis if trick room context matters
        if self.trick_room:
            context["speed_mode"] = "trick_room"
        
        # Add opponent inference summary
        opponent_inferences = {}
        for species, poke_state in self.opponent_team.pokemon.items():
            inference = poke_state.get_stat_inference_summary()
            if inference:
                opponent_inferences[species] = inference
            
            # Add speed relations
            speed_summary = poke_state.get_speed_summary()
            if speed_summary.get("relations"):
                if species not in opponent_inferences:
                    opponent_inferences[species] = {}
                opponent_inferences[species]["speed"] = speed_summary
        
        if opponent_inferences:
            context["opponent_stat_inferences"] = opponent_inferences
        
        return context
    
    def to_summary_string(self) -> str:
        """Generate human-readable summary."""
        lines = [f"=== Turn {self.current_turn} ==="]
        
        # Field
        field = self.get_field_summary()
        if field:
            lines.append(f"Field: {field}")
        
        # My team
        lines.append("\n[My Team]")
        for poke in self.my_team.get_active():
            lines.append(f"  Active: {poke.species} ({poke.hp_percent:.0f}%)")
        for poke in self.my_team.get_available():
            lines.append(f"  Bench: {poke.species} ({poke.hp_percent:.0f}%)")
        
        # Opponent team
        lines.append("\n[Opponent Team]")
        for poke in self.opponent_team.get_active():
            status = f", {poke.status}" if poke.status else ""
            lines.append(f"  Active: {poke.species} ({poke.hp_percent:.0f}%{status})")
        for poke in self.opponent_team.get_available():
            lines.append(f"  Bench: {poke.species} ({poke.hp_percent:.0f}%)")
        
        # Strategy
        if self.strategy.current_win_condition:
            lines.append(f"\n[Strategy] {self.strategy.current_win_condition}")
        
        return "\n".join(lines)


# =============================================================================
# Probabilistic Info Structures (for detailed inference)
# =============================================================================

@dataclass
class ProbabilisticValue:
    """
    Represents a value with associated probability and evidence.
    
    Probability levels:
    - 100.0: Directly observed (explicit in battle message)
    - 99.9: Certain inference (e.g., Life Orb recoil seen)
    - 95.0+: Very high confidence (strong evidence)
    - 80.0+: High confidence (good evidence)
    - 60.0+: Medium confidence (some evidence)
    - <60.0: Low confidence (weak evidence)
    """
    value: Any
    probability: float  # 0.0 - 100.0
    source: str  # "observed", "inferred", "predicted"
    evidence: str = ""  # Reason for this probability
    turn_observed: int = 0  # Turn when this was observed/inferred
    
    def to_dict(self) -> Dict:
        return {
            "value": self.value,
            "probability": self.probability,
            "source": self.source,
            "evidence": self.evidence,
            "turn": self.turn_observed
        }
    
    @staticmethod
    def observed(value: Any, turn: int, evidence: str = "Directly observed") -> "ProbabilisticValue":
        """Create a 100% observed value."""
        return ProbabilisticValue(value, 100.0, "observed", evidence, turn)
    
    @staticmethod
    def inferred(value: Any, probability: float, turn: int, evidence: str) -> "ProbabilisticValue":
        """Create an inferred value with probability."""
        return ProbabilisticValue(value, probability, "inferred", evidence, turn)


@dataclass
class PokemonInferredInfo:
    """
    Stores all known and inferred information about an opponent Pokemon.
    Each field can have multiple candidates with probabilities.
    """
    species: str
    
    # Single-value fields (only one can be true)
    ability: Optional[ProbabilisticValue] = None
    item: Optional[ProbabilisticValue] = None
    tera_type: Optional[ProbabilisticValue] = None
    nature: Optional[ProbabilisticValue] = None
    
    # Multi-candidate fields (could be one of several)
    ability_candidates: List[ProbabilisticValue] = field(default_factory=list)
    item_candidates: List[ProbabilisticValue] = field(default_factory=list)
    tera_type_candidates: List[ProbabilisticValue] = field(default_factory=list)
    
    # Known moves (set of move IDs with observation data)
    moves: Dict[str, ProbabilisticValue] = field(default_factory=dict)
    
    # EV spread inference (could be multiple possibilities)
    ev_spreads: List[ProbabilisticValue] = field(default_factory=list)
    
    # Speed tier observations
    speed_observations: List[Dict] = field(default_factory=list)  # {"outsped": bool, "opponent": str, "turn": int}
    
    # Damage observations for inference
    damage_log: List[Dict] = field(default_factory=list)
    
    # HP tracking
    last_hp_percent: float = 100.0
    
    def set_observed(self, field_name: str, value: Any, turn: int, evidence: str = "Directly observed"):
        """Set a field as directly observed (100% probability)."""
        pv = ProbabilisticValue.observed(value, turn, evidence)
        if field_name == "ability":
            self.ability = pv
            self.ability_candidates = [pv]  # Clear other candidates
        elif field_name == "item":
            self.item = pv
            self.item_candidates = [pv]
        elif field_name == "tera_type":
            self.tera_type = pv
            self.tera_type_candidates = [pv]
        elif field_name == "nature":
            self.nature = pv
    
    def add_candidate(self, field_name: str, value: Any, probability: float, turn: int, evidence: str):
        """Add a candidate with probability."""
        pv = ProbabilisticValue.inferred(value, probability, turn, evidence)
        
        if field_name == "ability":
            # Update existing or add new
            self._update_candidates(self.ability_candidates, pv)
            # Set as main if highest probability
            if not self.ability or probability > self.ability.probability:
                self.ability = pv
        elif field_name == "item":
            self._update_candidates(self.item_candidates, pv)
            if not self.item or probability > self.item.probability:
                self.item = pv
        elif field_name == "tera_type":
            self._update_candidates(self.tera_type_candidates, pv)
            if not self.tera_type or probability > self.tera_type.probability:
                self.tera_type = pv
        elif field_name == "ev_spread":
            self._update_candidates(self.ev_spreads, pv)
    
    def _update_candidates(self, candidates: List[ProbabilisticValue], new_pv: ProbabilisticValue):
        """Update candidate list, replacing if same value exists."""
        for i, c in enumerate(candidates):
            if c.value == new_pv.value:
                # Update if new probability is higher
                if new_pv.probability > c.probability:
                    candidates[i] = new_pv
                return
        candidates.append(new_pv)
        # Sort by probability descending
        candidates.sort(key=lambda x: x.probability, reverse=True)
        # Keep top 5 candidates
        if len(candidates) > 5:
            candidates.pop()
    
    def add_move(self, move_id: str, turn: int):
        """Add an observed move."""
        if move_id not in self.moves:
            self.moves[move_id] = ProbabilisticValue.observed(move_id, turn, "Move used in battle")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for GPT context."""
        result = {"species": self.species}
        
        if self.ability:
            result["ability"] = self.ability.to_dict()
        if self.ability_candidates and len(self.ability_candidates) > 1:
            result["ability_candidates"] = [c.to_dict() for c in self.ability_candidates[:3]]
        
        if self.item:
            result["item"] = self.item.to_dict()
        if self.item_candidates and len(self.item_candidates) > 1:
            result["item_candidates"] = [c.to_dict() for c in self.item_candidates[:3]]
        
        if self.tera_type:
            result["tera_type"] = self.tera_type.to_dict()
        
        if self.moves:
            result["known_moves"] = list(self.moves.keys())
        
        if self.ev_spreads:
            result["ev_spread_estimates"] = [e.to_dict() for e in self.ev_spreads[:2]]
        
        if self.nature:
            result["nature"] = self.nature.to_dict()
        
        return result
    
    def get_best_guess(self, field_name: str) -> Optional[Any]:
        """Get the highest probability value for a field."""
        if field_name == "ability":
            return self.ability.value if self.ability else None
        elif field_name == "item":
            return self.item.value if self.item else None
        elif field_name == "tera_type":
            return self.tera_type.value if self.tera_type else None
        return None


class PochampsPlayer(Player):
    """
    Pokemon VGC Battle AI using parallel GPT strategies.
    - Fast: Quick tactical response (configurable, default: backend model)
    - Normal: Balanced with tool usage (backend model)
    - Deep: Iterative analysis (configurable, default: backend model)
    
    For local OSS models (e.g., Ollama with gpt-oss Responses API server):
        - Set base_url to local server (e.g., "http://localhost:8080")
        - Set backend to your model name (e.g., "gpt-oss-20b")
    """

    def __init__(
        self,
        battle_format: str = "gen9vgc2025regi",
        api_key: str = "",
        backend: str = "gpt-4o-2024-08-06",
        fast_model: Optional[str] = None,  # None = use backend
        deep_model: Optional[str] = None,  # None = use backend
        base_url: Optional[str] = None,    # For local OSS models (e.g., http://localhost:8080)
        temperature: float = 0.8,
        log_dir: Optional[str] = None,
        team=None,
        save_replays=None,
        account_configuration=None,
        server_configuration=None,
        open_team_sheets: bool = False,
        debug_mode: bool = False
    ):
        # =========================================================================
        # CRITICAL: Initialize ALL attributes BEFORE super().__init__()
        # super().__init__() may trigger async message handlers that use these
        # =========================================================================
        
        # Game data - initialize defaults first (used by _parse_opponent_team etc.)
        self.gen9_moves = {}
        self.ability_effect = {}
        self.item_effect = {}
        self.gen9_pokedex = {}
        self.gen9_typechart = {}
        
        # Opponent info tracking (species -> PokemonInferredInfo)
        self._opponent_info: Dict[str, PokemonInferredInfo] = {}
        # My Pokemon HP tracking
        self._my_pokemon_hp: Dict[str, float] = {}
        # Token tracking (Responses API uses input_tokens, output_tokens)
        self._tokens = {"input": 0, "output": 0, "reasoning": 0}
        # Battle context tracking
        self._battle_response_ids: Dict[str, str] = {}
        self._strategy_progress: Dict[str, Dict] = {}
        
        # =========================================================================
        # Persistent Battle State Tracking (NEW)
        # Each battle gets its own BattleState object that persists throughout
        # =========================================================================
        self._battle_states: Dict[str, BattleState] = {}  # battle_tag -> BattleState
        
        # My team info (parsed once when team is set)
        self._my_team_info: List[Dict] = []
        
        # =========================================================================
        # Battle Turn Logs - Comprehensive visibility into each turn
        # =========================================================================
        self._turn_logs: Dict[str, List[BattleTurnLog]] = {}  # battle_tag -> List[BattleTurnLog]
        self._current_turn_log: Optional[BattleTurnLog] = None  # Current turn being logged
        
        # Debug mode: cache GPT responses to avoid API costs
        self.debug_mode = debug_mode
        self._response_cache: Dict[str, Dict] = {}  # cache key -> response
        self._cache_file = "debug_gpt_cache.json" if debug_mode else None

        # API Configuration
        self.backend = backend
        self.fast_model = fast_model or backend  # Default to backend if not specified
        self.deep_model = deep_model or backend  # Default to backend if not specified
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.log_dir = log_dir
        self.global_timeout = 120.0  # 2 minutes for deep strategy (gpt-5.1)

        # OpenAI client & thread pool
        # For local OSS models, set base_url (e.g., gpt-oss Responses API server)
        # Use dummy API key for local servers that don't require authentication
        if self.base_url:
            # Local server with OpenAI-compatible API
            self.client = OpenAI(api_key=self.api_key or "dummy-key", base_url=self.base_url)
        elif self.api_key:
            # OpenAI API - requires real key
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None
        
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # Load game data before super().__init__() since it may be needed
        self.gen = GenData.from_format(battle_format)
        self._load_game_data()

        # =========================================================================
        # NOW call super().__init__() - handlers may use initialized attributes
        # =========================================================================
        super().__init__(
            battle_format=battle_format,
            team=team,
            save_replays=save_replays,
            account_configuration=account_configuration,
            server_configuration=server_configuration,
            open_team_sheets=open_team_sheets
        )
        
        # Log configuration after super().__init__() so logger is available
        if self.base_url:
            self.logger.info(f"Using custom API endpoint: {self.base_url}")
        
        # Load debug cache after initialization
        if debug_mode:
            self._load_response_cache()

        # Tool executor - will be linked to BattleState cache per battle
        self.tool_executor = ToolExecutor(game_data={
            "moves": self.gen9_moves,
            "pokedex": self.gen9_pokedex,
            "items": self.item_effect,
            "abilities": self.ability_effect,
            "typechart": self.gen9_typechart
        })

    # =========================================================================
    # Debug Mode - Response Cache
    # =========================================================================
    
    def _load_response_cache(self):
        """Load cached GPT responses from file."""
        if self._cache_file:
            try:
                with open(self._cache_file, "r") as f:
                    self._response_cache = json.load(f)
                self.logger.debug(f"Loaded {len(self._response_cache)} cached responses")
            except FileNotFoundError:
                self._response_cache = {}
                self.logger.debug("No cache file found, starting fresh")
            except json.JSONDecodeError:
                self._response_cache = {}
                self.logger.debug("Cache file corrupted, starting fresh")
    
    def _save_response_cache(self):
        """Save GPT responses to cache file."""
        if self._cache_file and self._response_cache:
            try:
                with open(self._cache_file, "w") as f:
                    json.dump(self._response_cache, f, indent=2)
                self.logger.debug(f"Saved {len(self._response_cache)} responses to cache")
            except Exception as e:
                self.logger.debug(f"Failed to save cache: {e}")
    
    def _get_cache_key(self, context_type: str, data: Dict) -> str:
        """Generate a cache key from context data."""
        # Use sorted JSON string for consistent hashing
        import hashlib
        data_str = json.dumps(data, sort_keys=True)
        hash_val = hashlib.md5(data_str.encode()).hexdigest()[:12]
        return f"{context_type}_{hash_val}"
    
    def _get_cached_response(self, cache_key: str) -> Optional[Dict]:
        """Get cached response if available."""
        if self.debug_mode and cache_key in self._response_cache:
            self.logger.debug(f"Using cached response for {cache_key}")
            return self._response_cache[cache_key]
        return None
    
    def _cache_response(self, cache_key: str, response: Dict):
        """Cache a GPT response."""
        if self.debug_mode:
            self._response_cache[cache_key] = response
            self._save_response_cache()
            self.logger.debug(f"Cached response for {cache_key}")

    def _call_llm(self, model: str, messages: List[Dict], temperature: float = 0.5, 
                  max_tokens: int = 2000, json_format: bool = False, tools: List = None) -> Dict:
        """
        Unified LLM call method.
        - Uses ollama for local models (when base_url is set)
        - Uses OpenAI Responses API for OpenAI models
        
        Returns a dict with 'content' (text output) and optionally 'tool_calls'.
        """
        if self.base_url:
            return {
            }
        else:
            # Use OpenAI Responses API
            if not self.client:
                raise RuntimeError("OpenAI client not initialized. Provide api_key.")
            
            params = {
                "model": model,
                "input": messages,
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            
            if json_format:
                params["text"] = {"format": {"type": "json_object"}}
            
            if tools:
                params["tools"] = tools
            
            # Log params for debugging (hide full messages)
            debug_params = {k: (f"<{len(v)} items>" if k == "input" else v) for k, v in params.items()}
            print(f"[API CALL] params: {debug_params}")
            self.logger.info(f"[API CALL] Creating response with params: {debug_params}")
            
            try:
                response = self.client.responses.create(**params)
            except Exception as e:
                # Log full stack so 500 errors show the call path
                self.logger.exception("LLM call failed (model=%s): %s", model, e)
                raise
            
            # Track tokens
            usage = None
            if hasattr(response, 'usage') and response.usage:
                self._tokens['input'] += getattr(response.usage, 'input_tokens', 0)
                self._tokens['output'] += getattr(response.usage, 'output_tokens', 0)
                usage = {
                    "input_tokens": getattr(response.usage, 'input_tokens', 0),
                    "output_tokens": getattr(response.usage, 'output_tokens', 0)
                }
            
            # Extract content and tool calls
            content = None
            tool_calls = []
            
            if response.output:
                for item in response.output:
                    if item.type == "message":
                        for c in item.content:
                            if c.type == "output_text":
                                content = c.text
                                break
                    elif item.type == "function_call":
                        tool_calls.append({
                            "id": item.call_id,
                            "name": item.name,
                            "arguments": item.arguments
                        })
            
            # Also check output_text shorthand
            if not content and hasattr(response, 'output_text'):
                content = response.output_text
            
            return {
                "content": content,
                "tool_calls": tool_calls if tool_calls else None,
                "usage": usage,
                "response_id": response.id if hasattr(response, 'id') else None,
                "raw_response": response  # Keep raw for follow-up calls
            }

    def _call_llm_followup(self, model: str, previous_response_id: str, tool_results: List[Dict],
                           tools: List = None, temperature: float = 0.5, max_tokens: int = 2000) -> Dict:
        """
        Follow-up LLM call for tool result processing (OpenAI only).
        Ollama doesn't support this pattern, so returns None for Ollama.
        """
        if self.base_url:
            # Ollama doesn't support follow-up calls with previous_response_id
            return None
        
        if not self.client:
            raise RuntimeError("OpenAI client not initialized.")
        
        params = {
            "model": model,
            "previous_response_id": previous_response_id,
            "input": tool_results,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        
        if tools:
            params["tools"] = tools
        
        # Log params for debugging
        debug_params = {k: (f"<{len(v)} items>" if k == "input" else v) for k, v in params.items()}
        print(f"[API FOLLOWUP] params: {debug_params}")
        self.logger.info(f"[API FOLLOWUP] Creating response with params: {debug_params}")
        
        try:
            response = self.client.responses.create(**params)
        except Exception as e:
            # Log full stack for troubleshooting follow-up failures
            self.logger.exception("Follow-up LLM call failed (model=%s, prev_id=%s): %s", model, previous_response_id, e)
            raise
        
        # Track tokens
        usage = None
        if hasattr(response, 'usage') and response.usage:
            self._tokens['input'] += getattr(response.usage, 'input_tokens', 0)
            self._tokens['output'] += getattr(response.usage, 'output_tokens', 0)
            usage = {
                "input_tokens": getattr(response.usage, 'input_tokens', 0),
                "output_tokens": getattr(response.usage, 'output_tokens', 0)
            }
        
        # Extract content and tool calls
        content = None
        tool_calls = []
        
        if response.output:
            for item in response.output:
                if item.type == "message":
                    for c in item.content:
                        if c.type == "output_text":
                            content = c.text
                            break
                elif item.type == "function_call":
                    tool_calls.append({
                        "id": item.call_id,
                        "name": item.name,
                        "arguments": item.arguments
                    })
        
        if not content and hasattr(response, 'output_text'):
            content = response.output_text
        
        return {
            "content": content,
            "tool_calls": tool_calls if tool_calls else None,
            "usage": usage,
            "response_id": response.id if hasattr(response, 'id') else None,
            "raw_response": response
        }

    def _load_game_data(self):
        """Load game data files."""
        try:
            with open("src/data/static/moves/gen9moves.json", "r") as f:
                self.gen9_moves = json.load(f)
            with open("src/data/static/abilities/ability_effect.json", "r") as f:
                self.ability_effect = json.load(f)
            with open("src/data/static/items/item_effect.json", "r") as f:
                self.item_effect = json.load(f)
            with open("src/data/static/pokedex/gen9pokedex.json", "r") as f:
                self.gen9_pokedex = json.load(f)
            with open("src/data/static/typechart/gen9typechart.json", "r") as f:
                self.gen9_typechart = json.load(f)
        except FileNotFoundError as e:
            self.logger.warning(f"Could not load game data: {e}")
            self.gen9_moves = {}
            self.ability_effect = {}
            self.item_effect = {}
            self.gen9_pokedex = {}
            self.gen9_typechart = {}

    def _get_item_vgc_notes(self, item_name: str) -> str:
        """VGC에서 아이템의 전략적 의미 반환."""
        item_lower = item_name.lower().replace(" ", "").replace("-", "")
        
        vgc_notes = {
            # Choice 아이템
            "choicescarf": "⚡ Speed boost (1.5x) but locked into one move. Great for revenge killing or outspeeding threats. Check if opponent might have this!",
            "choiceband": "💪 Attack boost (1.5x) but locked into one move. Huge damage but predictable.",
            "choicespecs": "🔮 Sp.Atk boost (1.5x) but locked into one move. Can't use Protect!",
            
            # 생존 아이템
            "focussash": "🛡️ Survives one hit at full HP with 1 HP. Common on frail setup sweepers or leads. Broken by multi-hit moves, weather, hazards.",
            "assaultvest": "🧱 Sp.Def boost (1.5x) but CAN'T USE STATUS MOVES (no Protect!). Check before assuming they have Protect.",
            "eviolite": "🛡️ Def/SpD boost (1.5x) for NFE Pokemon. Makes Pokemon like Dusclops very bulky.",
            
            # 부스트 아이템
            "lifeorb": "💥 All attacks do 1.3x damage but lose 10% HP per attack. High damage output with flexibility.",
            "expertbelt": "🎯 Super effective moves do 1.2x damage. No drawback, good for coverage moves.",
            "weaknesspolicy": "📈 +2 Atk/SpA when hit by super effective move. Common with Tera to bait weaknesses.",
            
            # 유틸리티 아이템
            "safetygoggles": "👓 Immune to weather damage AND powder moves (Spore, Sleep Powder). Great vs Amoonguss!",
            "covertcloak": "🎭 Blocks secondary effects (flinch, stat drops, status from moves). Counters Fake Out flinch!",
            "clearamulet": "🔒 Prevents stat drops from opponent's moves/abilities. Counters Intimidate!",
            "protectivepads": "🧤 No contact effects (Rocky Helmet, Rough Skin, etc.).",
            
            # 열매
            "sitrusberry": "🍓 Heals 25% HP at 50% or less. Standard bulk berry.",
            "lumberry": "✨ Cures any status once. Great vs status-heavy teams.",
            "aguavberry": "🍇 Heals 33% HP at 25% or less (Gluttony activates at 50%).",
            "figyberry": "🍑 Heals 33% HP at 25% or less (Gluttony activates at 50%).",
            "wikiberry": "🍊 Heals 33% HP at 25% or less (Gluttony activates at 50%).",
            "iapapaberry": "🍋 Heals 33% HP at 25% or less (Gluttony activates at 50%).",
            "magoberry": "🍒 Heals 33% HP at 25% or less (Gluttony activates at 50%).",
            
            # 스위칭 아이템
            "ejectbutton": "🔄 Switches out when hit by a damaging move. Can disrupt opponent's plans.",
            "shedshell": "🐚 Can always switch out (ignores trapping). Counters Shadow Tag/Arena Trap.",
            
            # 트릭룸 아이템
            "ironball": "⚫ Halves Speed, grounds Flying-types. Used for Trick Room or to be slower.",
            "roomservice": "🚪 -1 Speed when Trick Room is set. Helps in Trick Room teams.",
            
            # 시그니처 오브
            "souldev": "👻 Giratina's orb. 1.2x Ghost/Dragon moves.",
            "adamantorb": "💎 Dialga's orb. 1.2x Steel/Dragon moves.",
            "lustrousorb": "🌊 Palkia's orb. 1.2x Water/Dragon moves.",
            "griseousorb": "😈 Giratina's orb (Origin). 1.2x Ghost/Dragon moves.",
            "boosterenergy": "⚡ Activates Protosynthesis/Quark Drive without weather/terrain. One-time boost.",
            
            # 기타 유용한 아이템
            "leftovers": "🍖 Heals 1/16 HP each turn. Sustained recovery.",
            "blacksludge": "🧪 Like Leftovers but damages non-Poison types if Tricked.",
            "lightclay": "🏗️ Extends Light Screen/Reflect/Aurora Veil to 8 turns.",
            "terrainextender": "🌍 Extends terrain to 8 turns.",
            "mentalherb": "🌿 Cures Taunt, Encore, Disable, etc. once. Good on support Pokemon.",
            "redcard": "🃏 Forces opponent to switch when holder is hit. Disruption tool.",
        }
        
        return vgc_notes.get(item_lower, "Standard item. Check effect description for details.")

    # =========================================================================
    # Battle State Management
    # =========================================================================
    
    def _get_or_create_battle_state(self, battle: AbstractBattle) -> BattleState:
        """Get existing BattleState or create new one for this battle."""
        battle_tag = battle.battle_tag
        
        if battle_tag not in self._battle_states:
            # Create new BattleState
            state = BattleState(
                battle_tag=battle_tag,
                format=self._format
            )
            self._battle_states[battle_tag] = state
            self.logger.debug(f"Created new BattleState for {battle_tag}")
        
        return self._battle_states[battle_tag]
    
    def _init_battle_state_from_preview(
        self, 
        battle: AbstractBattle, 
        state: BattleState,
        my_team_pokemon: List[str],
        opponent_pokemon: List[str]
    ):
        """Initialize battle state from team preview."""
        state.phase = "team_preview"
        
        # Initialize my team
        for species in my_team_pokemon:
            poke_state = state.my_team.add_pokemon(species=species)
            # If we have detailed team data, populate moves etc.
            for mon in battle.team.values():
                if mon.species == species or self._normalize_name(mon.species) == species:
                    # Set known moves
                    for move in mon.moves.values():
                        poke_state.moves[move.id] = MoveState(
                            move_id=move.id,
                            name=move.id,
                            max_pp=move.max_pp,
                            pp_remaining=move.current_pp,
                            is_confirmed=True,
                            probability=100.0
                        )
                    # Set item if known
                    if mon.item:
                        poke_state.item = mon.item
                    break
        
        # Initialize opponent team
        for species in opponent_pokemon:
            state.opponent_team.add_pokemon(species=species)
            # Also init inference tracking
            if species not in state.opponent_inferred:
                state.opponent_inferred[species] = PokemonInferredInfo(species=species)
        
        self.logger.debug(f"Initialized teams - My: {my_team_pokemon}, Opp: {opponent_pokemon}")
    
    def _update_battle_state_selections(
        self, 
        state: BattleState, 
        selected_pokemon: List[str],
        lead_pokemon: List[str]
    ):
        """Update state after team selection."""
        for species, poke in state.my_team.pokemon.items():
            normalized = self._normalize_name(species)
            poke.selected_for_battle = normalized in [self._normalize_name(s) for s in selected_pokemon]
            poke.is_lead = normalized in [self._normalize_name(s) for s in lead_pokemon]
        
        self.logger.debug(f"Selections - Bring: {selected_pokemon}, Lead: {lead_pokemon}")
    
    def _sync_battle_state(self, battle: AbstractBattle, state: BattleState):
        """
        Synchronize BattleState with current battle object.
        Called at the start of each turn to ensure state is up-to-date.
        """
        state.current_turn = battle.turn
        state.phase = "battle"
        
        # Sync field conditions
        if battle.weather:
            state.weather = str(battle.weather)
        else:
            state.weather = None
        
        if hasattr(battle, 'fields') and battle.fields:
            # Check for terrain
            for field_effect in battle.fields:
                field_name = str(field_effect).lower()
                if 'terrain' in field_name:
                    state.terrain = field_name
                    break
        
        # Sync my active pokemon
        active_species = set()
        if isinstance(battle, DoubleBattle):
            for slot, mon in enumerate(battle.active_pokemon, 1):
                if mon and not mon.fainted:
                    species = self._normalize_name(mon.species)
                    active_species.add(species)
                    poke = state.my_team.get(species)
                    if poke:
                        poke.set_active(slot)
                        poke.update_hp(
                            hp_percent=mon.current_hp_fraction * 100 if mon.current_hp_fraction else 0,
                            hp_exact=mon.current_hp
                        )
                        poke.status = str(mon.status) if mon.status else None
                        # Update stat boosts
                        for stat, val in mon.boosts.items():
                            if stat in poke.stat_boosts:
                                poke.stat_boosts[stat] = val
        else:
            # Singles
            if battle.active_pokemon and not battle.active_pokemon.fainted:
                mon = battle.active_pokemon
                species = self._normalize_name(mon.species)
                active_species.add(species)
                poke = state.my_team.get(species)
                if poke:
                    poke.set_active(1)
                    poke.update_hp(
                        hp_percent=mon.current_hp_fraction * 100 if mon.current_hp_fraction else 0,
                        hp_exact=mon.current_hp
                    )
                    poke.status = str(mon.status) if mon.status else None
        
        # Mark non-active pokemon as inactive
        for species, poke in state.my_team.pokemon.items():
            if species not in active_species and poke.is_active:
                poke.set_inactive()
        
        # Sync opponent active pokemon
        opp_active_species = set()
        if isinstance(battle, DoubleBattle):
            for slot, mon in enumerate(battle.opponent_active_pokemon, 1):
                if mon and not mon.fainted:
                    species = self._normalize_name(mon.species)
                    opp_active_species.add(species)
                    poke = state.opponent_team.get(species)
                    if not poke:
                        poke = state.opponent_team.add_pokemon(species=species)
                    poke.selected_for_battle = True  # If active, definitely selected
                    poke.set_active(slot)
                    poke.update_hp(hp_percent=mon.current_hp_fraction * 100 if mon.current_hp_fraction else 100)
                    poke.status = str(mon.status) if mon.status else None
                    # Sync observed ability/item
                    if mon.ability:
                        poke.ability = mon.ability
                    if mon.item:
                        poke.item = mon.item
        else:
            if battle.opponent_active_pokemon and not battle.opponent_active_pokemon.fainted:
                mon = battle.opponent_active_pokemon
                species = self._normalize_name(mon.species)
                opp_active_species.add(species)
                poke = state.opponent_team.get(species)
                if not poke:
                    poke = state.opponent_team.add_pokemon(species=species)
                poke.selected_for_battle = True
                poke.set_active(1)
                poke.update_hp(hp_percent=mon.current_hp_fraction * 100 if mon.current_hp_fraction else 100)
                poke.status = str(mon.status) if mon.status else None
        
        # Mark opponent non-active as inactive
        for species, poke in state.opponent_team.pokemon.items():
            if species not in opp_active_species and poke.is_active:
                poke.set_inactive()
        
        # Check fainted pokemon
        for mon in battle.team.values():
            if mon.fainted:
                species = self._normalize_name(mon.species)
                poke = state.my_team.get(species)
                if poke:
                    poke.fainted = True
                    poke.is_active = False
    
    def _record_move_used(
        self, 
        state: BattleState, 
        user_species: str, 
        move_id: str,
        move_name: str,
        is_opponent: bool,
        turn: int
    ):
        """Record a move was used."""
        team = state.opponent_team if is_opponent else state.my_team
        poke = team.get(user_species)
        if poke:
            poke.add_move(move_id, move_name, turn)
            poke.last_action = f"used {move_name}"
            poke.last_action_turn = turn
    
    def _cleanup_battle_state(self, battle_tag: str):
        """Clean up battle state when battle ends."""
        if battle_tag in self._battle_states:
            state = self._battle_states[battle_tag]
            state.phase = "ended"
            # Optionally keep for post-battle analysis, or delete
            # del self._battle_states[battle_tag]
            self.logger.debug(f"Battle ended: {battle_tag}")
    
    def _normalize_name(self, name: str) -> str:
        """Normalize pokemon/move names for comparison."""
        if not name:
            return ""
        return name.lower().replace(" ", "").replace("-", "").replace("'", "")

    def _sanitize_for_json(self, obj):
        """Recursively sanitize objects for JSON serialization."""
        from src.environment.pokemon_type import PokemonType
        
        if isinstance(obj, PokemonType):
            return obj.name
        elif isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._sanitize_for_json(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            # Handle objects with __dict__ but skip internal/complex objects
            if type(obj).__name__ in ('Pokemon', 'Move', 'Field', 'Weather'):
                return str(obj)
            return obj
        else:
            return obj

    def _build_prompt_payload(
        self,
        battle_context: Dict,
        state: Optional[BattleState]
    ) -> Dict:
        """Build enriched payload for the fast strategy prompt."""
        # Work on a deep copy to avoid mutating shared battle_context
        bc = copy.deepcopy(battle_context)
        
        # Sanitize for JSON serialization (convert PokemonType enums, etc.)
        bc = self._sanitize_for_json(bc)

        # Fill missing move/item info for active mons using available_switches_detailed
        switches_detail = bc.get("available_switches_detailed") or []
        flat_candidates = []
        for side in switches_detail:
            if side:
                flat_candidates.extend(side)

        if flat_candidates:
            for mon in bc.get("my_active", []) or []:
                if mon.get("moves"):
                    continue
                species = str(mon.get("species", "")).lower()
                if not species:
                    continue
                candidate = next((c for c in flat_candidates if str(c.get("species", "")).lower() == species), None)
                if candidate:
                    if candidate.get("moves"):
                        mon["moves"] = candidate["moves"]
                    # Backfill useful fields if missing/empty
                    for key in ("item", "ability", "types", "tera_type"):
                        if mon.get(key) in (None, [], "") and key in candidate:
                            mon[key] = candidate[key]

        # Secondary backfill: use parsed team builder info for my team
        if getattr(self, "_my_team_info", None):
            for mon in bc.get("my_active", []) or []:
                if mon.get("moves"):
                    continue
                species = str(mon.get("species", "")).lower()
                if not species:
                    continue
                team_entry = next((c for c in self._my_team_info if str(c.get("species", "")).lower() == species), None)
                if team_entry:
                    if team_entry.get("moves"):
                        mon["moves"] = team_entry["moves"]
                    for key in ("item", "ability", "types", "tera_type"):
                        if mon.get(key) in (None, [], "") and key in team_entry:
                            mon[key] = team_entry[key]

        payload: Dict[str, Any] = {"battle_context": bc}

        # Add structured battle state (strategy, history, field)
        if state:
            try:
                payload["battle_state"] = state.to_gpt_context()
            except Exception as e:
                self.logger.debug(f"Fast payload battle_state error: {e}")

            # Include team preview cache contents so fast prompt can see scouting data
            try:
                cache_summary = state.team_preview_cache.summary()
                payload["team_preview_cache_summary"] = cache_summary

                cache_data: Dict[str, Any] = {}
                for species_key in cache_summary.keys():
                    cache_data[species_key] = state.team_preview_cache.get_all(species_key)
                payload["team_preview_cache"] = cache_data
            except Exception as e:
                self.logger.debug(f"Fast payload cache error: {e}")

        # Add opponent inference map (probabilistic info)
        if self._opponent_info:
            try:
                payload["opponent_inferred"] = [info.to_dict() for info in self._opponent_info.values()]
            except Exception as e:
                self.logger.debug(f"Fast payload opponent_inferred error: {e}")

        # Lightweight human-readable summary
        try:
            payload["context_summary"] = self._summarize_gpt_context(battle_context)
        except Exception as e:
            payload["context_summary"] = f"summary_error: {e}"

        return payload

    # =========================================================================
    # Parallel GPT Strategies
    # =========================================================================

    def call_parallel_strategies(
        self,
        battle_tag: str,
        battle_context: Dict,
        state: Optional[BattleState] = None,
        tools: Optional[List[Dict]] = None,
        timeout: float = None
    ) -> Dict[str, str]:
        """
        Call 3 strategies in parallel: fast, normal, deep.
        Fast strategy receives enriched payload (battle_context + caches + state).
        Returns responses that complete within timeout.
        """
        # Removed duplicate log - strategies log their own execution
        if not self.client:
            self.logger.error("OpenAI client not initialized.")
            return {}
        
        timeout = timeout or self.global_timeout

        # Enrich prompt for fast strategy with full player state
        try:
            battle_payload = self._build_prompt_payload(battle_context, state)
        except Exception as e:
            self.logger.error(f"Failed to build battle payload: {e}", exc_info=True)
            # Fallback to minimal payload
            battle_payload = {"battle_context": self._sanitize_for_json(battle_context)}
        
        # Submit all strategies
        futures = {
            self.executor.submit(
                call_fast_strategy,
                client=self.client,
                battle_tag=battle_tag,
                battle_context=battle_payload,
                model=self.fast_model,  # Configurable (default: backend)
                temperature=0.3,
                max_tokens=1500,
                battle_response_ids=self._battle_response_ids,
                token_tracker=self._tokens,
                tools=tools
            ): "fast",
            self.executor.submit(
                call_normal_strategy,
                client=self.client,
                battle_tag=battle_tag,
                battle_context=battle_payload,
                model=self.backend,
                temperature=self.temperature,
                max_tokens=3000,
                battle_response_ids=self._battle_response_ids,
                token_tracker=self._tokens,
                tool_executor=self.tool_executor,
                tools=tools
            ): "normal",
            self.executor.submit(
                call_deep_strategy,
                client=self.client,
                battle_tag=battle_tag,
                battle_context=battle_payload,
                model=self.deep_model,  # Configurable (default: backend)
                temperature=self.temperature,
                max_tokens=50000,
                battle_response_ids=self._battle_response_ids,
                token_tracker=self._tokens,
                tool_executor=self.tool_executor,
                progress_tracker=self._strategy_progress,
                tools=tools,
            ): "deep"
        }
        
        results = {}
        try:
            for future in as_completed(futures.keys(), timeout=timeout):
                name = futures[future]
                try:
                    results[name] = future.result()
                    self.logger.info(f"Strategy '{name}' completed")
                except Exception as e:
                    self.logger.warning(f"Strategy '{name}' failed: {e}")
        except TimeoutError:
            self.logger.warning(f"Timeout ({timeout}s) reached")
            for f in futures:
                if not f.done():
                    f.cancel()
        
        return results

    def select_best_response(self, responses: Dict[str, str], battle_context: Dict = None) -> Optional[Dict[str, Any]]:
        """Select best response by synthesizing all strategy responses with GPT.
        
        Instead of simple priority selection, uses GPT to analyze all available
        strategy responses and make an informed final decision.
        
        Returns dict with 'strategy' name and 'content' (parsed JSON).
        """
        # Collect all valid parsed responses
        valid_responses = {}
        for strategy in ["fast", "normal", "deep"]:
            if strategy in responses and responses[strategy]:
                try:
                    parsed = json.loads(responses[strategy])
                    valid_responses[strategy] = parsed
                except json.JSONDecodeError:
                    continue
        
        if not valid_responses:
            self.logger.warning("No valid responses to select from")
            return None
        
        # If only one valid response, use it directly
        if len(valid_responses) == 1:
            strategy_name = list(valid_responses.keys())[0]
            self.logger.info(f"Using only available '{strategy_name}' response")
            return {
                "strategy": strategy_name,
                "content": valid_responses[strategy_name],
                "raw": responses[strategy_name]
            }
        
        # Multiple responses - use GPT to synthesize final decision
        return self._synthesize_turn_decision(valid_responses, battle_context)
    
    def _synthesize_turn_decision(
        self, 
        strategy_responses: Dict[str, Dict], 
        battle_context: Dict = None
    ) -> Optional[Dict[str, Any]]:
        """
        Use GPT to synthesize the best decision from multiple strategy responses.
        
        This is the TURN DECISION phase - GPT analyzes all strategy outputs
        and makes a final, informed decision considering:
        - Fast strategy: Quick tactical assessment
        - Normal strategy: Balanced analysis with tool usage
        - Deep strategy: Thorough iterative analysis
        """
        if not self.client:
            # Fallback to priority selection if no client
            for strategy in ["deep", "normal", "fast"]:
                if strategy in strategy_responses:
                    return {
                        "strategy": strategy,
                        "content": strategy_responses[strategy],
                        "raw": json.dumps(strategy_responses[strategy])
                    }
            return None
        
        # Build synthesis prompt
        synthesis_prompt = self._build_turn_decision_prompt(strategy_responses, battle_context)
        
        try:
            # Use fast model for quick synthesis (this should be fast)
            # Note: OpenAI structured outputs don't support oneOf, so we use a flat schema
            # with all fields and action as discriminator (move/switch)
            slot_action_schema = {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["move", "switch"]},
                    "move": {"type": ["string", "null"]},
                    "target": {"type": ["integer", "null"]},
                    "terastallize": {"type": ["boolean", "null"]},
                    "pokemon": {"type": ["string", "null"]}
                },
                "required": ["action"],
                "additionalProperties": False
            }
            
            request_params = {
                "model": self.fast_model,
                "input": [{"role": "user", "content": synthesis_prompt}],
                "max_output_tokens": 400,
                "temperature": 0.2,  # Low temp for consistent decisions
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "turn_decision",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "synthesis_reasoning": {"type": "string"},
                                "slot1": slot_action_schema,
                                "slot2": slot_action_schema
                            },
                            "required": ["synthesis_reasoning", "slot1", "slot2"],
                            "additionalProperties": False
                        }
                    }
                }
            }
            
            response = self.client.responses.create(**request_params)
            
            # Track tokens
            if hasattr(response, 'usage') and response.usage:
                self._tokens["input"] += getattr(response.usage, 'input_tokens', 0)
                self._tokens["output"] += getattr(response.usage, 'output_tokens', 0)
            
            # Extract response text
            response_text = None
            if hasattr(response, 'output_text') and response.output_text:
                response_text = response.output_text
            elif response.output:
                for item in response.output:
                    if item.type == "message":
                        for c in item.content:
                            if hasattr(c, 'text'):
                                response_text = c.text
                                break
            
            if response_text:
                # Parse the final decision
                final_decision = self._parse_synthesis_response(response_text, strategy_responses)
                if final_decision:
                    return final_decision
            
            # Fallback to deep > normal > fast
            self.logger.warning("Synthesis failed, falling back to priority selection")
            for strategy in ["deep", "normal", "fast"]:
                if strategy in strategy_responses:
                    return {
                        "strategy": strategy,
                        "content": strategy_responses[strategy],
                        "raw": json.dumps(strategy_responses[strategy])
                    }
            
        except Exception as e:
            self.logger.warning(f"Turn decision synthesis failed: {e}")
            # Fallback
            for strategy in ["deep", "normal", "fast"]:
                if strategy in strategy_responses:
                    return {
                        "strategy": strategy,
                        "content": strategy_responses[strategy],
                        "raw": json.dumps(strategy_responses[strategy])
                    }
        
        return None
    
    def _build_turn_decision_prompt(
        self, 
        strategy_responses: Dict[str, Dict],
        battle_context: Dict = None
    ) -> str:
        """Build the TURN DECISION synthesis prompt."""
        
        # Format each strategy's response
        strategy_summaries = []
        for strategy_name in ["fast", "normal", "deep"]:
            if strategy_name not in strategy_responses:
                continue
            
            resp = strategy_responses[strategy_name]
            summary = f"=== {strategy_name.upper()} STRATEGY ===\n"
            
            for slot in ["slot1", "slot2"]:
                if slot not in resp:
                    continue
                slot_data = resp[slot]
                
                # Handle case where slot_data is a string instead of dict
                if isinstance(slot_data, str):
                    summary += f"  {slot}: {slot_data}\n"
                    continue
                
                action = slot_data.get("action", "?")
                
                if action == "move":
                    move = slot_data.get("move", "?")
                    target = slot_data.get("target", "?")
                    tera = " [TERA]" if slot_data.get("terastallize") else ""
                    summary += f"  {slot}: {action} {move} -> target {target}{tera}\n"
                elif action == "switch":
                    pokemon = slot_data.get("pokemon", "?")
                    summary += f"  {slot}: {action} to {pokemon}\n"
                else:
                    summary += f"  {slot}: {action}\n"
            
            # Include reasoning if available
            if resp.get("reasoning"):
                summary += f"  Reasoning: {resp['reasoning'][:200]}...\n" if len(str(resp.get('reasoning', ''))) > 200 else f"  Reasoning: {resp.get('reasoning')}\n"
            
            strategy_summaries.append(summary)
        
        # Build context summary with DETAILED move information per slot
        context_summary = ""
        if battle_context:
            # Work on a copy and backfill missing move info from switches/team builder
            bc = copy.deepcopy(battle_context)
            switches_detail = bc.get("available_switches_detailed") or []
            flat_candidates = []
            for side in switches_detail:
                if side:
                    flat_candidates.extend(side)

            if flat_candidates:
                for mon in bc.get("my_active", []) or []:
                    if mon.get("moves"):
                        continue
                    species = str(mon.get("species", "")).lower()
                    if not species:
                        continue
                    candidate = next((c for c in flat_candidates if str(c.get("species", "")).lower() == species), None)
                    if candidate:
                        if candidate.get("moves"):
                            mon["moves"] = candidate["moves"]
                        for key in ("item", "ability", "types", "tera_type"):
                            if mon.get(key) in (None, [], "") and key in candidate:
                                mon[key] = candidate[key]

            if getattr(self, "_my_team_info", None):
                for mon in bc.get("my_active", []) or []:
                    if mon.get("moves"):
                        continue
                    species = str(mon.get("species", "")).lower()
                    if not species:
                        continue
                    team_entry = next((c for c in self._my_team_info if str(c.get("species", "")).lower() == species), None)
                    if team_entry:
                        if team_entry.get("moves"):
                            mon["moves"] = team_entry["moves"]
                        for key in ("item", "ability", "types", "tera_type"):
                            if mon.get(key) in (None, [], "") and key in team_entry:
                                mon[key] = team_entry[key]

            context_summary = f"""
CURRENT BATTLE STATE:
- Turn: {bc.get('turn', '?')}
- Force Switch: {bc.get('force_switch', [False, False])}
"""
            # Add DETAILED active Pokemon info with available moves
            if bc.get('my_active'):
                context_summary += "\n=== YOUR ACTIVE POKEMON (USE ONLY THESE MOVES!) ===\n"
                for p in bc['my_active']:
                    if isinstance(p, dict):
                        slot_num = p.get('slot', '?')
                        species = p.get('species', '?')
                        hp = p.get('hp_percent', '?')
                        context_summary += f"\n[SLOT {slot_num}] {species} ({hp}%)\n"
                        
                        # List available moves with target info
                        moves = p.get('moves', [])
                        if moves:
                            context_summary += "  AVAILABLE MOVES:\n"
                            for m in moves:
                                if isinstance(m, dict):
                                    move_id = m.get('id', m.get('name', '?'))
                                    move_type = m.get('type', '?')
                                    category = m.get('category', '?')
                                    bp = m.get('base_power', 0) or '-'
                                    targets = m.get('valid_targets', [0])
                                else:
                                    move_id = str(m)
                                    move_type = '?'
                                    category = '?'
                                    bp = '-'
                                    targets = [0]
                                
                                # Determine target type from move data
                                target_info = ""
                                if targets == [0] or targets == []:
                                    target_info = "⚠️ NO TARGET (spread/self)"
                                elif -2 in targets and len(targets) == 1:
                                    target_info = "SELF only (target=-2 or omit)"
                                elif -1 in targets:
                                    target_info = f"targets={targets}"
                                else:
                                    target_info = f"target=1(opp L) or 2(opp R)"
                                
                                context_summary += f"    - {move_id}: {move_type}/{category} BP={bp} | {target_info}\n"
                        else:
                            context_summary += "  (No moves available - must switch)\n"
            
            if bc.get('opponent_active'):
                context_summary += "\n=== OPPONENT ACTIVE ===\n"
                for p in battle_context['opponent_active']:
                    if isinstance(p, dict):
                        slot_num = p.get('slot', '?')
                        species = p.get('species', '?')
                        hp = p.get('hp_percent', '?')
                        types = "/".join(p.get('types', [])) if p.get('types') else "?"
                        context_summary += f"  [Slot {slot_num}] {species} ({types}) {hp}%\n"
            
            # Available switches
            switches = battle_context.get('available_switches', [])
            if switches:
                context_summary += "\n=== AVAILABLE SWITCHES ===\n"
                for slot_idx, slot_switches in enumerate(switches):
                    if isinstance(slot_switches, list) and slot_switches:
                        context_summary += f"  Slot {slot_idx + 1}: {', '.join(slot_switches)}\n"
        
        prompt = f"""══════════════════════════════════════════════════════════════
                    🎯 TURN DECISION - FINAL SYNTHESIS 🎯
══════════════════════════════════════════════════════════════

You have received recommendations from 3 different analysis strategies.
Synthesize these into ONE optimal final decision.

- Fast strategy: Quick tactical assessment (low cost, shallow).
- Normal strategy: Balanced analysis with limited tool usage.
- Deep strategy: Thorough iterative analysis (highest ceiling but can time out or be unstable).

STRATEGY PRIORITY:
1) If the Deep strategy recommendation is well-formed, legal, and does not break any rules below,
   you SHOULD PREFER Deep, since it explores future turns more thoroughly.
2) If Deep is missing, invalid, clearly illegal, or self-destructive, FALL BACK to Normal,
   as the main, stable default.
3) Only if BOTH Deep and Normal produce unusable/illegal decisions, THEN fall back to Fast.
4) If multiple valid strategies remain, choose the one that:
   - Respects all legality and ally-safety rules below, and
   - Keeps the best overall board position (not throwing away the game for small gain).

{context_summary}
{chr(10).join(strategy_summaries)}

══════════════════════════════════════════════════════════════
⚠️ CRITICAL VALIDATION RULES ⚠️

1. USE ONLY AVAILABLE MOVES:
   - Each slot can ONLY use moves listed under its own AVAILABLE MOVES.
   - Slot 1 can ONLY use Slot 1’s moves.
   - Slot 2 can ONLY use Slot 2’s moves.
   - DO NOT MIX UP which Pokemon has which moves.
   - If a slot has no usable moves or force_switch for that slot is True, that slot MUST switch.

2. TARGET RULES:
   - Single-target offensive moves:
     • target = 1 (left opponent)
     • target = 2 (right opponent)
   - Ally target:
     • target = -1 (ally partner)
   - Spread / self / targetless moves:
     • Use target = 0 OR OMIT the target field entirely.
   - NEVER use any string like "omit" for target; target MUST be an integer when present.

3. SPREAD MOVES (NO TARGET):
   - Moves that naturally hit multiple Pokemon (e.g., hypervoice, makeitrain, earthquake,
     heatwave, dazzlinggleam, etc.) must NOT specify a single-opponent target.
   - For these, use target: 0 OR omit the target field entirely.
   - The server will reject choices like: “You can't choose a target for X” if you specify target 1 or 2.

4. FORCE SWITCH:
   - If force_switch = [True, X] or force_switch = [X, True] for a given slot,
     that slot MUST use action="switch".
   - The "pokemon" chosen for a switch MUST be one of the valid bench options listed for that slot.

5. ALLY-SAFETY RULE (VERY IMPORTANT):
   - You MUST NOT choose single-target damaging moves that intentionally target the ally (target = -1).
   - Allowed ally targets:
     • Legitimate ally-targeting support/status moves (e.g., Helping Hand, Follow Me/Rage Powder,
       Ally Switch, healing, defensive buffs, etc.).
     • Spread moves where hitting the ally is an unavoidable side effect (earthquake, surf, etc.)
       and you are using target = 0 or omitting the target.
   - If a move is single-target and damaging, its target MUST be 1 or 2 (opponents), NOT -1.
   - When in doubt, prefer attacking opponents or using safe support/self moves over harming allies.

6. GENERAL LEGALITY:
   - Do NOT invent moves, Pokemon, items, or targets that are not present in the provided context.
   - Do NOT mix move and switch fields in the same slot.
   - Ensure both slot1 and slot2 decisions are internally consistent and executable by the battle server.

══════════════════════════════════════════════════════════════

OUTPUT FORMAT (JSON only, no explanation outside JSON):
{{
    "synthesis_reasoning": "<brief explanation of which strategy you followed and why>",
    "slot1": {{"action": "move", "move": "<move_id>", "target": 0|1|2|-1}}
             OR {{"action": "switch", "pokemon": "<bench_name>"}},
    "slot2": {{"action": "move", "move": "<move_id>", "target": 0|1|2|-1}}
             OR {{"action": "switch", "pokemon": "<bench_name>"}}
}}

"""
        return prompt
    
    def _parse_synthesis_response(
        self, 
        response_text: str, 
        strategy_responses: Dict[str, Dict]
    ) -> Optional[Dict[str, Any]]:
        """Parse the synthesis response and extract final decision."""
        try:
            # Try direct JSON parse
            parsed = json.loads(response_text)
            
            # Log the synthesis decision
            print(f"\n[TURN DECISION] Synthesized from {len(strategy_responses)} strategies:")
            if parsed.get("synthesis_reasoning"):
                print(f"  Reasoning: {parsed['synthesis_reasoning']}")
            for slot in ["slot1", "slot2"]:
                if slot in parsed:
                    slot_data = parsed[slot]
                    action = slot_data.get("action", "?")
                    if action == "move":
                        print(f"  {slot}: {action} {slot_data.get('move', '?')} -> target {slot_data.get('target', '?')}")
                    elif action == "switch":
                        print(f"  {slot}: {action} to {slot_data.get('pokemon', '?')}")
            
            self.logger.info(f"[TURN DECISION] Synthesized decision from {list(strategy_responses.keys())}")
            
            return {
                "strategy": "synthesized",
                "content": parsed,
                "raw": response_text,
                "source_strategies": list(strategy_responses.keys())
            }
            
        except json.JSONDecodeError:
            # Try to extract JSON from text
            import re
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    print(f"\n[TURN DECISION] Synthesized (extracted JSON):")
                    self.logger.info(f"[TURN DECISION] Synthesized decision")
                    return {
                        "strategy": "synthesized",
                        "content": parsed,
                        "raw": json_match.group(),
                        "source_strategies": list(strategy_responses.keys())
                    }
                except json.JSONDecodeError:
                    pass
        
        return None

    # =========================================================================
    # Battle Logic
    # =========================================================================

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """Main decision function - routes to doubles or singles."""
        self.logger.warning("Battle %s - Turn %d", battle.battle_tag, battle.turn)
        
        # Get or create BattleState for this battle
        state = self._get_or_create_battle_state(battle)
        
        # Advance turn and sync state
        state.advance_turn(battle.turn)
        self._sync_battle_state(battle, state)
        
        # Print state summary for debugging
        if self.debug_mode:
            print(state.to_summary_string())
        
        # Forfeit for testing
        #return ForfeitBattleOrder()
        if isinstance(battle, DoubleBattle):
            return self._choose_doubles_move_async(battle, state)
        # Singles is rarely used in VGC - fallback to random
        return self.choose_random_move(battle)

    async def _choose_doubles_move_async(self, battle: DoubleBattle, state: Optional[BattleState] = None) -> BattleOrder:
        """
        Doubles battle logic - main VGC format (ASYNC version).
        
        This is async to prevent blocking the event loop during GPT calls,
        which allows WebSocket ping/pong to continue and prevents timeout.
        
        Flow:
        1. Update confirmed info from last message
        2. Build battle context for GPT (including BattleState)
        3. Call GPT with parallel strategies (fast/normal/deep) - in thread
        4. Log turn details for visibility
        """
        import asyncio
        
        # Get state if not provided
        if state is None:
            state = self._get_or_create_battle_state(battle)
        
        # =====================================================================
        # Initialize Turn Log for this decision
        # =====================================================================
        turn_log = BattleTurnLog(
            turn_number=battle.turn,
            timestamp=datetime.now().strftime("%H:%M:%S.%f")[:-3],
            phase="battle"
        )
        
        # Capture field state
        turn_log.field_state = self._capture_field_state(battle)
        
        # Capture active Pokemon states
        turn_log.my_active = [
            BattleTurnLog.format_pokemon_state(p) 
            for p in battle.team.values() if p.active
        ]
        turn_log.opponent_active = [
            BattleTurnLog.format_pokemon_state(p) 
            for p in battle.opponent_team.values() if p.active
        ]
        
        # =====================================================================
        # Link ToolExecutor's cache to BattleState's TeamPreviewCache
        # This allows battle-phase tools to access data collected in team preview
        # =====================================================================
        self.tool_executor.set_cache(state.team_preview_cache)
        
        # =====================================================================
        # Step 1: Update confirmed information from last battle message
        # =====================================================================
        self._update_battle_info(battle, state)
        
        # =====================================================================
        # Step 2: Build battle context for GPT (with BattleState)
        # =====================================================================
        battle_context = self._build_doubles_context(battle, state)
        
        # =====================================================================
        # LOG: Battle state details for debugging
        # =====================================================================
        force_switch = getattr(battle, 'force_switch', [False, False])
        if not isinstance(force_switch, list):
            force_switch = [False, False]
        
        self.logger.info(f"Turn {battle.turn}: force_switch={force_switch}")
        
        # Log active Pokemon
        for i, poke in enumerate(battle.active_pokemon):
            if poke:
                self.logger.info(f"  Active Slot{i+1}: {poke.species} ({round(poke.current_hp_fraction*100,1)}%)")
            else:
                self.logger.info(f"  Active Slot{i+1}: EMPTY (fainted/switched)")
        
        # Log available switches per slot
        if hasattr(battle, 'available_switches') and battle.available_switches:
            for slot_idx, slot_switches in enumerate(battle.available_switches):
                if isinstance(slot_switches, list):
                    switch_names = [p.species for p in slot_switches]
                    must_switch = " [MUST SWITCH]" if force_switch[slot_idx] else ""
                    self.logger.info(f"  Slot{slot_idx+1} switches{must_switch}: {switch_names}")
        
        # Capture GPT context summary for logging
        turn_log.gpt_context_summary = self._summarize_gpt_context(battle_context)
        
        # =====================================================================
        # Step 3: GPT inference with 3 parallel strategies (non-blocking)
        # Run in thread to prevent blocking event loop (keeps WebSocket alive)
        # =====================================================================
        responses = await asyncio.to_thread(
            self.call_parallel_strategies,
            battle_tag=battle.battle_tag,
            battle_context=battle_context,
            state=state,
            tools=BATTLE_TOOLS,
            timeout=self.global_timeout
        )
        
        # Capture all strategy responses
        turn_log.gpt_responses = self._capture_strategy_responses(responses)
        
        # Use GPT to synthesize the best decision from all strategies
        best = self.select_best_response(responses, battle_context=battle_context)
        if best:
            # Capture selected strategy
            turn_log.selected_strategy = best.get("strategy", "unknown")
            
            # =====================================================================
            # LOG: GPT Decision for debugging
            # =====================================================================
            raw_content = best.get("content", {})
            self.logger.info(f"[GPT DECISION] Strategy: {best.get('strategy', 'unknown')}")
            if isinstance(raw_content, dict):
                for slot_key in ["slot1", "slot2"]:
                    if slot_key in raw_content:
                        slot_data = raw_content[slot_key]
                        action = slot_data.get("action", "?")
                        if action == "move":
                            move = slot_data.get("move", "?")
                            target = slot_data.get("target", "?")
                            tera = slot_data.get("terastallize", False)
                            tera_str = " [TERA]" if tera else ""
                            self.logger.info(f"  {slot_key}: MOVE {move} -> target={target}{tera_str}")
                        elif action == "switch":
                            pokemon = slot_data.get("pokemon", "?")

                            # If GPT omitted the switch target, infer the first available switch for this slot
                            if (not pokemon or pokemon == "?") and hasattr(battle, "available_switches"):
                                slot_switches = []
                                if isinstance(battle.available_switches, list) and battle.available_switches:
                                    if isinstance(battle.available_switches[0], list):
                                        slot_switches = battle.available_switches[i] if i < len(battle.available_switches) else []
                                    else:
                                        slot_switches = battle.available_switches
                                if slot_switches:
                                    pokemon = slot_switches[0].species
                                    slot_data["pokemon"] = pokemon
                            self.logger.info(f"  {slot_key}: SWITCH to {pokemon}")
                        else:
                            self.logger.info(f"  {slot_key}: {action} {slot_data}")
            
            # Pass the content (parsed dict) to parser
            print(f"[DEBUG] Calling _parse_doubles_decision with content: {best.get('content', {})}")
            orders = self._parse_doubles_decision(best.get("content", {}), battle, turn_log)
            print(f"[DEBUG] _parse_doubles_decision returned: {orders}")
            if orders:
                print(f"[DEBUG] Orders valid, finalizing turn log")
                # Finalize and store turn log
                self._finalize_turn_log(battle.battle_tag, turn_log, orders)
                return orders
            else:
                print(f"[DEBUG] Orders is None/False - parse failed")
            
            # =====================================================================
            # Step 3.5: RETRY - If parse failed, send feedback and get new response
            # (Does NOT re-run analysis - just asks for corrected decision)
            # =====================================================================
            validation = turn_log.decision_validation or {}
            print(f"[DEBUG] Validation dict: {validation}")
            if validation.get("needs_retry"):
                print(f"  [RETRY] Invalid decision - requesting correction (not re-running analysis)")
                retry_feedback = self._build_retry_feedback(validation, battle)
                
                # Run in thread to prevent blocking event loop
                corrected = await asyncio.to_thread(
                    self._request_decision_correction,
                    battle_tag=battle.battle_tag,
                    strategy=best.get("strategy", "normal"),
                    feedback=retry_feedback
                )
                
                if corrected:
                    orders = self._parse_doubles_decision(corrected, battle, turn_log)
                    if orders:
                        turn_log.selected_strategy = f"{best.get('strategy', 'unknown')}_corrected"
                        self._finalize_turn_log(battle.battle_tag, turn_log, orders)
                        return orders
        
        # Fallback to random
        print(f"[DEBUG] Falling back to random move - GPT decision/parse failed")
        self.logger.error("[CRITICAL] All strategies failed - using random fallback")
        turn_log.final_decision = "FALLBACK: Random move (GPT failed)"
        turn_log.selected_strategy = "random_fallback"
        
        # Use safe random that respects ally-safety rules
        try:
            random_order = self.choose_random_doubles_move(battle)
        except Exception as e:
            self.logger.error(f"Random fallback also failed: {e}")
            # Ultimate fallback - default order
            random_order = self.create_order(battle.available_moves[0]) if battle.available_moves else ForfeitBattleOrder()
        
        self._finalize_turn_log(battle.battle_tag, turn_log, random_order)
        return random_order
    
    def _request_decision_correction(
        self,
        battle_tag: str,
        strategy: str,
        feedback: str
    ) -> Optional[Dict]:
        """
        Request a corrected decision from GPT without re-running analysis.
        Uses the existing response_id to continue the conversation.
        """
        try:
            if not self.client:
                return None
            
            # Use appropriate model based on strategy
            model = self.backend
            if strategy == "deep":
                model = self.deep_model
            elif strategy == "fast":
                model = self.fast_model
            
            request_params = {
                "model": model,
                "input": [{"role": "user", "content": feedback}],
                "max_output_tokens": 300
            }
            
            # NOTE: Don't use previous_response_id for error correction
            # Previous response may have pending tool calls which causes 400 error
            # Start a fresh conversation with just the feedback
            
            # Temperature for non-reasoning models
            if not (model.startswith("o") or "gpt-5" in model):
                request_params["temperature"] = 0.3
            
            response = self.client.responses.create(**request_params)
            
            # Update response ID
            if hasattr(response, 'id'):
                self._battle_response_ids[battle_tag] = response.id
            
            # Track tokens
            if hasattr(response, 'usage') and response.usage:
                if hasattr(response.usage, 'input_tokens'):
                    self._tokens["input"] += response.usage.input_tokens
                if hasattr(response.usage, 'output_tokens'):
                    self._tokens["output"] += response.usage.output_tokens
            
            # Extract text content
            text = None
            for item in response.output:
                if item.type == "message":
                    for content in item.content:
                        if content.type == "output_text":
                            text = content.text
                            break
            
            if text:
                print(f"  [OK] Received corrected decision")
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    # Try to extract JSON from text
                    import re
                    json_match = re.search(r'\{[\s\S]*\}', text)
                    if json_match:
                        return json.loads(json_match.group())
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Decision correction failed: {e}")
            return None
    
    def _build_retry_feedback(self, validation: Dict, battle: DoubleBattle) -> str:
        """Build feedback message for GPT retry when decision was invalid."""
        reasons = []
        
        if validation.get("retry_reason"):
            reasons.append(validation["retry_reason"])
        
        # Check force_switch status
        force_switch = getattr(battle, 'force_switch', [False, False])
        if not isinstance(force_switch, list):
            force_switch = [False, False]
        
        # List active Pokemon to clarify
        active_species = []
        for poke in battle.active_pokemon:
            if poke:
                active_species.append(poke.species)
        
        # List available switches
        available = []
        if hasattr(battle, 'available_switches') and battle.available_switches:
            for slot_switches in battle.available_switches:
                if slot_switches:
                    for p in slot_switches:
                        if p.species not in available:
                            available.append(p.species)
        
        # Build force_switch info
        force_switch_info = ""
        if force_switch[0] or force_switch[1]:
            force_switch_info = "\n⚡ FORCE SWITCH STATUS:\n"
            if force_switch[0]:
                force_switch_info += "  - SLOT 1 MUST SWITCH (cannot use moves!)\n"
            else:
                force_switch_info += "  - Slot 1 can use moves normally\n"
            if force_switch[1]:
                force_switch_info += "  - SLOT 2 MUST SWITCH (cannot use moves!)\n"
            else:
                force_switch_info += "  - Slot 2 can use moves normally\n"
        
        feedback = f"""YOUR PREVIOUS DECISION WAS INVALID. Provide corrected decision.

REASON: {'; '.join(reasons) if reasons else 'Invalid action'}
{force_switch_info}
ALREADY ACTIVE (CANNOT switch to these):
- {', '.join(active_species)}

VALID SWITCH OPTIONS:
- {', '.join(available) if available else 'None'}

Respond with ONLY the corrected JSON. Do NOT switch to a Pokemon already active!
If a slot has force_switch=True, you MUST use action="switch" for that slot!"""
        
        return feedback
    
    def _capture_field_state(self, battle: DoubleBattle) -> Dict[str, Any]:
        """Capture current field conditions for logging."""
        field_state = {
            "weather": str(battle.weather) if battle.weather else None,
            "terrain": None,
            "trick_room": False,
            "my_side": [],
            "opponent_side": []
        }
        
        # Field effects
        for field in battle.fields:
            field_str = str(field).lower()
            if "trickroom" in field_str:
                field_state["trick_room"] = True
            elif "terrain" in field_str:
                field_state["terrain"] = str(field)
        
        # Side conditions
        for cond in battle.side_conditions:
            field_state["my_side"].append(str(cond))
        for cond in battle.opponent_side_conditions:
            field_state["opponent_side"].append(str(cond))
        
        return field_state
    
    def _summarize_gpt_context(self, context) -> str:
        """Create a readable summary of what GPT received."""
        try:
            # Handle dict context (from _build_doubles_context)
            if isinstance(context, dict):
                summary_parts = []
                
                # Turn info
                summary_parts.append(f"Turn: {context.get('turn', '?')}")
                
                # Field conditions
                field = context.get('field', {})
                if isinstance(field, dict) and (field.get('weather') or field.get('terrain') or field.get('trick_room')):
                    conditions = []
                    if field.get('weather'): conditions.append(f"Weather: {field['weather']}")
                    if field.get('terrain'): conditions.append(f"Terrain: {field['terrain']}")
                    if field.get('trick_room'): conditions.append("Trick Room")
                    if field.get('my_tailwind'): conditions.append("My Tailwind")
                    if field.get('opp_tailwind'): conditions.append("Opp Tailwind")
                    summary_parts.append(f"Field: {', '.join(conditions)}")
                
                # My active Pokemon
                my_active = context.get('my_active', [])
                if my_active and isinstance(my_active, list):
                    summary_parts.append("\n[My Active]")
                    for poke in my_active:
                        if isinstance(poke, dict):
                            moves = [m.get('id', '?') for m in poke.get('moves', []) if isinstance(m, dict)]
                            summary_parts.append(f"  Slot{poke.get('slot', '?')}: {poke.get('species', '?')} HP:{poke.get('hp_percent', '?')}% Moves: {', '.join(moves[:4])}")
                
                # Opponent active Pokemon
                opp_active = context.get('opponent_active', [])
                if opp_active and isinstance(opp_active, list):
                    summary_parts.append("\n[Opponent Active]")
                    for poke in opp_active:
                        if isinstance(poke, dict):
                            summary_parts.append(f"  Slot{poke.get('slot', '?')}: {poke.get('species', '?')} HP:{poke.get('hp_percent', '?')}%")
                
                return "\n".join(summary_parts)
            
            # Handle string context (legacy)
            if isinstance(context, str):
                lines = context.split('\n')
                summary_parts = []
                
                # Find key sections
                in_section = None
                for line in lines:
                    if "MY ACTIVE POKEMON" in line or "내 활성 포켓몬" in line:
                        in_section = "my_active"
                        summary_parts.append("\n[My Active Pokemon]")
                    elif "OPPONENT ACTIVE" in line or "상대 활성" in line:
                        in_section = "opp_active"
                        summary_parts.append("\n[Opponent Active Pokemon]")
                    elif "AVAILABLE MOVES" in line or "사용 가능한 기술" in line:
                        in_section = "moves"
                        summary_parts.append("\n[Available Moves]")
                    elif "BENCH POKEMON" in line or "벤치 포켓몬" in line:
                        in_section = "bench"
                        summary_parts.append("\n[Bench Pokemon]")
                    elif line.strip() and in_section:
                        # Add relevant lines
                        if len(line.strip()) < 200:  # Skip very long lines
                            summary_parts.append(f"  {line.strip()}")
                
                if not summary_parts:
                    # Fallback: return truncated context
                    return context[:1000] + "..." if len(context) > 1000 else context
                
                return "\n".join(summary_parts)
            
            # Unknown type - convert to string
            return str(context)[:500]
        except Exception as e:
            return f"Error summarizing context: {e}"
    
    def _capture_strategy_responses(self, responses: Dict[str, Any]) -> Dict[str, Any]:
        """Capture all strategy responses for logging."""
        captured = {}
        
        for strategy_name, resp in responses.items():
            # Handle string responses (raw JSON string)
            if isinstance(resp, str):
                captured[strategy_name] = {
                    "status": "success",
                    "reasoning": resp[:500] + "..." if len(resp) > 500 else resp,
                    "decision": None,
                    "tokens": {"input": 0, "output": 0}
                }
            elif isinstance(resp, dict):
                if resp.get("error"):
                    captured[strategy_name] = {
                        "status": "error",
                        "error": resp["error"]
                    }
                else:
                    content = resp.get("content", "")
                    captured[strategy_name] = {
                        "status": "success",
                        "reasoning": content[:500] + "..." if len(content) > 500 else content,
                        "decision": resp.get("decision"),
                        "tokens": {
                            "input": resp.get("input_tokens", 0),
                            "output": resp.get("output_tokens", 0)
                        }
                    }
            else:
                captured[strategy_name] = {
                    "status": "unknown",
                    "raw": str(resp)[:200]
                }
        
        return captured
    
    def _finalize_turn_log(self, battle_tag: str, turn_log: BattleTurnLog, orders: BattleOrder) -> None:
        """Finalize turn log and store it (no console output - saved to file at battle end)."""
        # Format final decision
        turn_log.final_decision = str(orders) if orders else "None"
        
        # Store in logs
        if battle_tag not in self._turn_logs:
            self._turn_logs[battle_tag] = []
        self._turn_logs[battle_tag].append(turn_log)

    def _update_battle_info(self, battle: DoubleBattle, state: Optional[BattleState] = None) -> None:
        """
        Update opponent information using GPT-based inference.
        Also updates BattleState with observations.
        
        Process:
        1. Extract turn events (damage, moves, abilities, items)
        2. Calculate expected damage and compare with actual
        3. Use GPT to infer EVs, items, abilities from discrepancies
        4. Update confirmed info database and BattleState
        """
        # Get state if not provided
        if state is None:
            state = self._get_or_create_battle_state(battle)
        
        # Step 1: Extract direct observations from battle state
        self._extract_direct_observations(battle, state)
        
        # Step 2: Extract turn events for analysis
        turn_events = self._extract_turn_events(battle)
        if not turn_events:
            return
        
        # Step 3: Analyze with GPT for deeper inference
        if self.client and battle.turn > 1:
            self._analyze_turn_with_gpt(battle, turn_events)
        
        # Step 4: Record turn summary in state
        if turn_events:
            state.record_turn_actions(battle.turn, {
                "events_count": len(turn_events),
                "summary": turn_events[:3]  # First 3 events as sample
            })

    def _get_or_create_pokemon_info(self, species: str) -> PokemonInferredInfo:
        """Get or create PokemonInferredInfo for a species."""
        if species not in self._opponent_info:
            self._opponent_info[species] = PokemonInferredInfo(species=species)
        return self._opponent_info[species]

    def _extract_direct_observations(self, battle: DoubleBattle, state: Optional[BattleState] = None) -> None:
        """Extract directly observable information from battle state and update BattleState."""
        for pokemon in battle.opponent_team.values():
            info = self._get_or_create_pokemon_info(pokemon.species)
            species_normalized = self._normalize_name(pokemon.species)
            
            # Get PokemonState from BattleState
            poke_state = state.opponent_team.get(species_normalized) if state else None
            if state and not poke_state:
                # Get base stats from pokedex for stat range initialization
                base_stats = None
                species_key = pokemon.species.lower().replace(" ", "").replace("-", "")
                if hasattr(self, 'gen9_pokedex') and species_key in self.gen9_pokedex:
                    base_stats = self.gen9_pokedex[species_key].get("baseStats", {})
                poke_state = state.opponent_team.add_pokemon(
                    species=species_normalized, 
                    base_stats=base_stats
                )
            
            # Confirmed ability (if revealed) - 100% probability
            if pokemon.ability:
                info.set_observed("ability", pokemon.ability, battle.turn, 
                                  "Ability revealed in battle")
                if poke_state:
                    poke_state.ability = pokemon.ability
            
            # Confirmed item (if revealed or consumed) - 100% probability
            if pokemon.item:
                info.set_observed("item", pokemon.item, battle.turn,
                                  "Item revealed/consumed in battle")
                if poke_state:
                    poke_state.item = pokemon.item
            
            # Confirmed moves (from usage) - 100% probability
            if pokemon.moves:
                for move_id in pokemon.moves.keys():
                    info.add_move(move_id, battle.turn)
                    if poke_state:
                        poke_state.add_move(move_id, move_id, battle.turn)
            
            # Confirmed tera type (if terastallized) - 100% probability
            if pokemon.terastallized and pokemon._terastallized_type:
                tera_type_str = pokemon._terastallized_type.name if hasattr(pokemon._terastallized_type, 'name') else str(pokemon._terastallized_type)
                info.set_observed("tera_type", tera_type_str, battle.turn, "Pokemon terastallized")
                if poke_state:
                    poke_state.tera_type = tera_type_str
                    poke_state.has_terastallized = True
                if state:
                    state.opponent_team.tera_used = True
            
            # Track HP changes for damage analysis
            current_hp = round(pokemon.current_hp_fraction * 100, 1)
            if poke_state:
                poke_state.update_hp(current_hp)
            
            if info.last_hp_percent != current_hp:
                hp_change = info.last_hp_percent - current_hp
                if hp_change > 0:
                    info.damage_log.append({
                        "turn": battle.turn,
                        "damage_percent": hp_change,
                        "hp_before": info.last_hp_percent,
                        "hp_after": current_hp
                    })
                info.last_hp_percent = current_hp
            
            # Track status
            if pokemon.status and poke_state:
                poke_state.status = str(pokemon.status)
                
        # Record speed order observations from this turn's action order
        if state:
            self._record_speed_order_observations(battle, state)
            self._record_item_ability_effects(battle, state)
    
    def _record_speed_order_observations(self, battle: DoubleBattle, state: BattleState) -> None:
        """
        Record speed order observations from turn events.
        
        In VGC doubles, action order reveals relative speeds when same priority.
        This tracks which Pokemon moved before others.
        """
        # Get active Pokemon on both sides
        my_active = [p for p in battle.team.values() if p.active]
        opp_active = [p for p in battle.opponent_team.values() if p.active]
        
        if not my_active or not opp_active:
            return
        
        # Check battle.observations if available (some implementations have this)
        # Otherwise we can infer from the damage log timing
        
        # For now, we can at least record that active Pokemon on field 
        # share the same turn context
        turn = battle.turn
        
        # Get field conditions for context
        context_parts = []
        if battle.fields:
            for field in battle.fields:
                if "trickroom" in str(field).lower():
                    context_parts.append("trick_room")
        
        for side in [battle.side_conditions, battle.opponent_side_conditions]:
            if side:
                for cond in side:
                    if "tailwind" in str(cond).lower():
                        if side == battle.side_conditions:
                            context_parts.append("my_tailwind")
                        else:
                            context_parts.append("opp_tailwind")
        
        context = ", ".join(context_parts) if context_parts else ""
        
        # TODO: Parse battle log to get exact move order
        # For now, this is a placeholder that can be enhanced with
        # proper battle log parsing
    
    def _record_item_ability_effects(self, battle: DoubleBattle, state: BattleState) -> None:
        """
        Record item and ability effect observations from battle events.
        
        Checks for:
        - Life Orb recoil (10% self-damage after attack)
        - Focus Sash activation (survived with 1 HP)
        - Leftovers/Black Sludge healing
        - Berry consumption
        - Ability activations (Intimidate, etc.)
        """
        turn = battle.turn
        
        for pokemon in battle.opponent_team.values():
            species_normalized = self._normalize_name(pokemon.species)
            poke_state = state.opponent_team.get(species_normalized)
            if not poke_state:
                continue
            
            info = self._opponent_info.get(pokemon.species)
            if not info:
                continue
            
            # Check for Life Orb recoil (10% HP loss after attacking)
            # Need to check damage log for 10% self-damage patterns
            current_hp = round(pokemon.current_hp_fraction * 100, 1)
            
            # Check for Focus Sash (survived at 1 HP when would have fainted)
            if current_hp <= 1 and current_hp > 0:
                # Look at damage log to see if they took fatal damage
                recent_damage = [d for d in info.damage_log if d.get("turn") == turn]
                for d in recent_damage:
                    if d.get("hp_before", 100) - d.get("damage_percent", 0) < 0:
                        # Would have fainted but didn't
                        poke_state.record_item_effect("focus_sash", turn)
                        break
            
            # Check ability from pokemon object
            if pokemon.ability and not poke_state.ability_effect_observed:
                ability_str = str(pokemon.ability).lower()
                if "intimidate" in ability_str:
                    poke_state.record_ability_effect("intimidate_activated", turn)
                elif "levitate" in ability_str:
                    poke_state.record_ability_effect("levitate_immune", turn)

    def _extract_turn_events(self, battle: DoubleBattle) -> List[Dict]:
        """
        Extract relevant events from the current turn.
        Returns list of event dictionaries for analysis.
        """
        events = []
        
        # Track damage events for each opponent Pokemon
        for pokemon in battle.opponent_team.values():
            info = self._opponent_info.get(pokemon.species)
            if info:
                # Get damage from current turn
                current_turn_damage = [
                    d for d in info.damage_log 
                    if d.get("turn") == battle.turn
                ]
                if current_turn_damage:
                    events.append({
                        "type": "damage_to_opponent",
                        "target": pokemon.species,
                        "damage_percent": sum(d["damage_percent"] for d in current_turn_damage),
                        "hp_after": round(pokemon.current_hp_fraction * 100, 1)
                    })
                
                # Track new moves
                if pokemon.moves:
                    for move_id in pokemon.moves.keys():
                        if move_id not in info.moves:
                            events.append({
                                "type": "move_revealed",
                                "pokemon": pokemon.species,
                                "move": move_id
                            })
        
        # Track damage to my Pokemon
        for pokemon in battle.team.values():
            current_hp = round(pokemon.current_hp_fraction * 100, 1)
            last_hp = self._my_pokemon_hp.get(pokemon.species, 100.0)
            
            if last_hp > current_hp:
                events.append({
                    "type": "damage_to_me",
                    "target": pokemon.species,
                    "damage_percent": last_hp - current_hp,
                    "hp_after": current_hp
                })
            
            self._my_pokemon_hp[pokemon.species] = current_hp
        
        return events

    def _analyze_turn_with_gpt(self, battle: DoubleBattle, events: List[Dict]) -> None:
        """
        Use GPT to analyze turn events and infer hidden information.
        """
        if not events:
            return
        
        # Build analysis context
        analysis_context = {
            "turn": battle.turn,
            "events": events,
            "opponent_pokemon": [],
            "my_pokemon": []
        }
        
        # Add opponent Pokemon with current inferred info
        for species, info in self._opponent_info.items():
            pokemon_data = info.to_dict()
            # Add base stats from pokedex if available
            if species.lower() in self.gen9_pokedex:
                pokemon_data["base_stats"] = self.gen9_pokedex[species.lower()].get("baseStats", {})
            analysis_context["opponent_pokemon"].append(pokemon_data)
        
        # Add my Pokemon stats for damage reference
        for pokemon in battle.team.values():
            analysis_context["my_pokemon"].append({
                "species": pokemon.species,
                "hp_percent": round(pokemon.current_hp_fraction * 100, 1),
                "stats": pokemon.stats if hasattr(pokemon, 'stats') else None
            })
        
        try:
            messages = [
                {"role": "system", "content": self._get_analysis_system_prompt()},
                {"role": "user", "content": json.dumps(analysis_context, indent=2)}
            ]
            
            result = self._call_llm(
                model=self.fast_model,
                messages=messages,
                temperature=0.3,
                max_tokens=400,
                json_format=True
            )
            
            text = result.get("content")
            if text:
                inferences = json.loads(text)
                self._apply_inferences(inferences, battle.turn)
                
        except Exception as e:
            self.logger.debug(f"Turn analysis error: {e}")

    def _get_analysis_system_prompt(self) -> str:
        """System prompt for turn analysis."""
        return """You are a Pokemon VGC battle analyst. Analyze turn events to infer opponent info.

PROBABILITY GUIDELINES:
- 100.0: Directly observed (don't output, already tracked)
- 99.9: Certain (e.g., Life Orb recoil seen = Life Orb)
- 95.0: Very high confidence (strong mechanical evidence)
- 85.0: High confidence (good evidence, few alternatives)
- 70.0: Medium-high (reasonable evidence)
- 60.0: Medium (some evidence, multiple possibilities)
- <50.0: Don't include (too uncertain)

ANALYSIS TASKS:
1. Item Inference:
   - Life Orb recoil (10% self-damage after attack) = 99.9%
   - 1.3x damage boost observed = 95% Life Orb
   - Focus Sash (1 HP from full) = 99.9%
   - Leftovers heal (6.25%) = 99.9%
   - Choice lock pattern = 85% Choice item

2. Ability Inference:
   - Weather on switch = 99.9% weather ability
   - Terrain on switch = 99.9% terrain ability
   - Unexpected immunity = 95% type immunity ability

3. EV Spread Inference:
   - Survived OHKO = 85% defensive investment
   - Outsped when shouldn't = 90% speed investment
   - High damage roll = 80% offensive investment

4. Speed Tier:
   - Outsped X = "faster than X"
   - Undersped X = "slower than X"

OUTPUT JSON:
{
    "inferences": [
        {
            "pokemon": "<species>",
            "field": "item|ability|ev_spread|nature|speed_tier",
            "value": "<inferred_value>",
            "probability": <float 50.0-99.9>,
            "evidence": "<specific observation that led to this>"
        }
    ]
}

Be precise. Only include inferences with probability >= 60.0."""

    def _apply_inferences(self, inferences: Dict, turn: int) -> None:
        """Apply GPT inferences to PokemonInferredInfo."""
        for inference in inferences.get("inferences", []):
            species = inference.get("pokemon")
            field = inference.get("field")
            value = inference.get("value")
            probability = inference.get("probability", 50.0)
            evidence = inference.get("evidence", "GPT inference")
            
            if not species or not field or not value:
                continue
            
            # Skip low probability inferences
            if probability < 60.0:
                continue
            
            info = self._get_or_create_pokemon_info(species)
            
            # Add as candidate with probability
            if field in ["item", "ability", "tera_type", "ev_spread"]:
                info.add_candidate(field, value, probability, turn, evidence)
                self.logger.debug(f"Inferred {species}.{field}={value} ({probability}%): {evidence}")
            
            elif field == "nature":
                if not info.nature or probability > info.nature.probability:
                    info.nature = ProbabilisticValue.inferred(value, probability, turn, evidence)
            
            elif field == "speed_tier":
                info.speed_observations.append({
                    "value": value,
                    "probability": probability,
                    "evidence": evidence,
                    "turn": turn
                })

    def _build_doubles_context(self, battle: DoubleBattle, state: Optional[BattleState] = None) -> Dict:
        """
        Build doubles battle context for GPT.
        
        Combines:
        - BattleState (persistent tracking: strategy, history, selections)
        - Battle object (real-time: available moves, targets, can_tera)
        """
        # Get state if available
        if state is None:
            state = self._battle_states.get(battle.battle_tag)
        
        # =====================================================================
        # CRITICAL: Check force_switch status
        # =====================================================================
        force_switch = getattr(battle, 'force_switch', [False, False])
        if not isinstance(force_switch, list):
            force_switch = [False, False]
        
        # Check if we've already used Terastallization this battle
        team_tera_used = any(p.terastallized for p in battle.team.values())
        
        context = {
            "turn": battle.turn,
            "force_switch": force_switch,  # [slot1_must_switch, slot2_must_switch]
            "my_active": [],
            "opponent_active": [],
            "my_team": [],
            "opponent_revealed": [],
            "tera_available": not team_tera_used,  # Can we still Terastallize this battle?
            "field": {
                "weather": str(battle.weather) if battle.weather else None,
                "terrain": str(battle.fields) if battle.fields else None
            }
        }
        
        # Parse field conditions more explicitly for speed calculations
        trick_room_active = False
        my_tailwind = False
        opp_tailwind = False
        
        if battle.fields:
            for field in battle.fields:
                if "trickroom" in str(field).lower():
                    trick_room_active = True
                    context["field"]["trick_room"] = True
        
        if battle.side_conditions:
            for cond in battle.side_conditions:
                if "tailwind" in str(cond).lower():
                    my_tailwind = True
                    context["field"]["my_tailwind"] = True
        
        if battle.opponent_side_conditions:
            for cond in battle.opponent_side_conditions:
                if "tailwind" in str(cond).lower():
                    opp_tailwind = True
                    context["field"]["opp_tailwind"] = True
        
        # Add speed mode hint for GPT
        if trick_room_active:
            context["speed_mode"] = "trick_room"  # Slower pokemon move first
        elif my_tailwind or opp_tailwind:
            context["speed_mode"] = "tailwind_active"  # 2x speed for affected side
        else:
            context["speed_mode"] = "normal"
        
        # Add strategy context from BattleState (if available)
        if state:
            context["strategy"] = state.strategy.to_dict()
            context["turn_history"] = state.turn_history[-5:]  # Last 5 turns
        
        # My active Pokemon (up to 2) - need real-time move/target info
        for i, pokemon in enumerate(battle.active_pokemon):
            if pokemon:
                moves_info = []
                if i < len(battle.available_moves) and battle.available_moves[i]:
                    for move in battle.available_moves[i]:
                        # Get valid targets for this move
                        try:
                            targets = battle.get_possible_showdown_targets(move, pokemon)
                        except:
                            targets = [0]  # Fallback: no specific target
                        
                        moves_info.append({
                            "id": move.id,
                            "type": move.type.name if move.type else None,
                            "category": move.category.name if move.category else None,
                            "base_power": move.base_power,
                            "valid_targets": targets  # Showdown target indices
                        })
                
                # Check if team has already used Terastallization
                team_tera_used = any(
                    p.terastallized for p in battle.team.values()
                )
                
                pokemon_data = {
                    "slot": i + 1,
                    "species": pokemon.species,
                    "hp_percent": round(pokemon.current_hp_fraction * 100, 1),
                    "status": str(pokemon.status) if pokemon.status else None,
                    "moves": moves_info,
                    "item": pokemon.item,
                    "ability": pokemon.ability,
                    "types": [t.name for t in pokemon.types if t],
                    "tera_type": pokemon._terastallized_type.name if pokemon._terastallized_type else None,
                    "terastallized": pokemon.terastallized,
                    "can_terastallize": (battle.can_tera[i] if i < len(battle.can_tera) else False) and not team_tera_used
                }
                
                # Add state tracking info
                if state:
                    poke_state = state.my_team.get(self._normalize_name(pokemon.species))
                    if poke_state:
                        pokemon_data["turns_on_field"] = poke_state.turns_on_field
                
                context["my_active"].append(pokemon_data)
        
        # Opponent active Pokemon (up to 2) - combine battle data + BattleState + PokemonInferredInfo
        for i, pokemon in enumerate(battle.opponent_active_pokemon):
            if pokemon:
                species_norm = self._normalize_name(pokemon.species)
                inferred_info = self._opponent_info.get(pokemon.species)
                poke_state = state.opponent_team.get(species_norm) if state else None
                
                opp_data = {
                    "slot": i + 1,
                    "species": pokemon.species,
                    "hp_percent": round(pokemon.current_hp_fraction * 100, 1),
                    "status": str(pokemon.status) if pokemon.status else None,
                    "known_moves": list(pokemon.moves.keys()) if pokemon.moves else [],
                    # Type info (critical for deep strategy)
                    "types": [t.name for t in pokemon.types if t] if pokemon.types else [],
                    # Tera info
                    "tera_type": pokemon._terastallized_type.name if pokemon._terastallized_type else None,
                    "terastallized": pokemon.terastallized,
                    # Confirmed values (100% probability if present)
                    "item": {"value": pokemon.item, "probability": 100.0} if pokemon.item else None,
                    "ability": {"value": pokemon.ability, "probability": 100.0} if pokemon.ability else None
                }
                
                # =========================================================
                # Add BattleState (PokemonState) tracking info - NEW SYSTEM
                # =========================================================
                if poke_state:
                    opp_data["turns_on_field"] = poke_state.turns_on_field
                    
                    # Moves with usage count from state (TOP 10 only)
                    if poke_state.moves:
                        opp_data["predicted_moves"] = poke_state.get_moves_for_gpt(limit=10)
                    
                    # Speed tracking - critical for VGC decisions
                    speed_summary = poke_state.get_speed_summary()
                    if speed_summary:
                        opp_data["speed"] = speed_summary
                    
                    # Calculate current effective speed with field conditions
                    if poke_state.actual_speed_range["min"] > 0:
                        # Check field conditions
                        opp_tailwind = any("tailwind" in str(c).lower() 
                                          for c in battle.opponent_side_conditions) if battle.opponent_side_conditions else False
                        opp_paralyzed = pokemon.status and "par" in str(pokemon.status).lower()
                        trick_room = any("trickroom" in str(f).lower() 
                                        for f in battle.fields) if battle.fields else False
                        
                        eff_speed = poke_state.calculate_effective_speed(
                            tailwind=opp_tailwind,
                            paralysis=opp_paralyzed,
                            speed_boost_stages=poke_state.stat_boosts.get("spe", 0),
                            trick_room=trick_room
                        )
                        opp_data["effective_speed"] = eff_speed
                    
                    # Item/Ability effects observed
                    if poke_state.item_effect_observed:
                        opp_data["item_effect_observed"] = poke_state.item_effect_observed
                    if poke_state.ability_effect_observed:
                        opp_data["ability_effect_observed"] = poke_state.ability_effect_observed
                    
                    # Stat range inferences
                    stat_inferences = poke_state.get_stat_inference_summary()
                    if stat_inferences:
                        opp_data["stat_inferences"] = stat_inferences
                    
                    # Stat boosts (non-zero only)
                    boosts = {k: v for k, v in poke_state.stat_boosts.items() if v != 0}
                    if boosts:
                        opp_data["stat_boosts"] = boosts
                
                # =========================================================
                # Add legacy PokemonInferredInfo (for backward compatibility)
                # =========================================================
                if inferred_info:
                    # Item candidates if not confirmed
                    if not opp_data["item"] and inferred_info.item:
                        opp_data["item"] = inferred_info.item.to_dict()
                    if inferred_info.item_candidates and len(inferred_info.item_candidates) > 1:
                        opp_data["item_candidates"] = [c.to_dict() for c in inferred_info.item_candidates[:3]]
                    
                    # Ability candidates if not confirmed
                    if not opp_data["ability"] and inferred_info.ability:
                        opp_data["ability"] = inferred_info.ability.to_dict()
                    if inferred_info.ability_candidates and len(inferred_info.ability_candidates) > 1:
                        opp_data["ability_candidates"] = [c.to_dict() for c in inferred_info.ability_candidates[:3]]
                    
                    # EV spread estimates (from GPT inference)
                    if inferred_info.ev_spreads:
                        opp_data["ev_estimates_legacy"] = [e.to_dict() for e in inferred_info.ev_spreads[:2]]
                    
                    # Nature
                    if inferred_info.nature:
                        opp_data["nature"] = inferred_info.nature.to_dict()
                    
                    # Speed tier observations (legacy)
                    if inferred_info.speed_observations:
                        opp_data["speed_info_legacy"] = inferred_info.speed_observations[-3:]
                
                context["opponent_active"].append(opp_data)
        
        # My full team - add selection info from state
        for p in battle.team.values():
            species_norm = self._normalize_name(p.species)
            poke_state = state.my_team.get(species_norm) if state else None
            
            team_data = {
                "species": p.species,
                "hp_percent": round(p.current_hp_fraction * 100, 1),
                "fainted": p.fainted,
                "active": p.active,
                "tera_type": p._terastallized_type.name if p._terastallized_type else None,
                "status": str(p.status) if p.status else None
            }
            
            if poke_state:
                team_data["is_lead"] = poke_state.is_lead
                team_data["selected"] = poke_state.selected_for_battle
            
            context["my_team"].append(team_data)
        
        # Opponent revealed team - combine state + inferred info
        for p in battle.opponent_team.values():
            species_norm = self._normalize_name(p.species)
            inferred_info = self._opponent_info.get(p.species)
            poke_state = state.opponent_team.get(species_norm) if state else None
            
            opp_data = {
                "species": p.species,
                "hp_percent": round(p.current_hp_fraction * 100, 1),
                "fainted": p.fainted,
                "active": p.active,
                "known_moves": list(p.moves.keys()) if p.moves else [],
                # Type info (important for deep strategy)
                "types": [t.name for t in p.types if t] if p.types else [],
                "tera_type": p._terastallized_type.name if p._terastallized_type else None,
                "item": {"value": p.item, "probability": 100.0} if p.item else None,
                "ability": {"value": p.ability, "probability": 100.0} if p.ability else None
            }
            
            # =========================================================
            # Add BattleState (PokemonState) tracking info
            # =========================================================
            if poke_state:
                opp_data["was_lead"] = poke_state.is_lead
                
                # Predicted moves (TOP 10, confirmed first)
                if poke_state.moves:
                    opp_data["predicted_moves"] = poke_state.get_moves_for_gpt(limit=10)
                
                # Speed info (for switch-in planning)
                if poke_state.actual_speed_range["min"] > 0 or poke_state.speed_relations:
                    opp_data["speed"] = poke_state.get_speed_summary()
                
                # Item/Ability observed effects
                if poke_state.item and not opp_data["item"]:
                    opp_data["item"] = {"value": poke_state.item, "probability": 100.0}
                if poke_state.item_effect_observed:
                    opp_data["item_effect"] = poke_state.item_effect_observed
                if poke_state.ability:
                    opp_data["ability"] = {"value": poke_state.ability, "probability": 100.0}
                if poke_state.ability_effect_observed:
                    opp_data["ability_effect"] = poke_state.ability_effect_observed
                
                # Stat inferences
                stat_inferences = poke_state.get_stat_inference_summary()
                if stat_inferences:
                    opp_data["stat_inferences"] = stat_inferences
            
            # =========================================================
            # Add legacy PokemonInferredInfo
            # =========================================================
            if inferred_info:
                if not opp_data.get("item") and inferred_info.item:
                    opp_data["item"] = inferred_info.item.to_dict()
                if not opp_data.get("ability") and inferred_info.ability:
                    opp_data["ability"] = inferred_info.ability.to_dict()
                if inferred_info.ev_spreads:
                    opp_data["ev_estimates_legacy"] = [e.to_dict() for e in inferred_info.ev_spreads[:2]]
            
            context["opponent_revealed"].append(opp_data)
        
        # Available switches - doubles format: List[List[Pokemon]] (per slot)
        available_switches = []
        available_switches_detailed = []
        try:
            if hasattr(battle, 'available_switches') and battle.available_switches:
                # Doubles: available_switches is List[List[Pokemon]] 
                # Index 0 = switches for slot 1, Index 1 = switches for slot 2
                for slot_idx, slot_switches in enumerate(battle.available_switches):
                    slot_list = []
                    detailed_list = []
                    if slot_switches:
                        for p in slot_switches:
                            slot_list.append(p.species)
                            # Detailed info for deep strategy
                            detailed_list.append({
                                "species": p.species,
                                "hp_percent": round(p.current_hp_fraction * 100, 1),
                                "types": [t.name for t in p.types if t] if p.types else [],
                                "ability": p.ability,
                                "item": p.item,
                                "status": str(p.status) if p.status else None,
                                "moves": [m.id for m in p.moves.values()] if p.moves else []
                            })
                    available_switches.append(slot_list)
                    available_switches_detailed.append(detailed_list)
        except Exception as e:
            self.logger.debug(f"Error getting available switches: {e}")
            # Fallback to empty lists per slot
            available_switches = [[], []]
            available_switches_detailed = [[], []]
        
        context["available_switches"] = available_switches  # Simple: [["species1", "species2"], ["species1", "species2"]]
        context["available_switches_detailed"] = available_switches_detailed  # Detailed for deep strategy
        
        return context

    def _parse_doubles_decision(
        self, 
        response, 
        battle: DoubleBattle,
        turn_log: Optional[BattleTurnLog] = None
    ) -> Optional[BattleOrder]:
        """
        Parse GPT doubles decision to DoubleBattleOrder.
        
        CRITICAL: Target validation
        - In VGC doubles, targets are position-based:
          - 1 = opponent's position 1 (their left, our right from their view)
          - 2 = opponent's position 2 (their right, our left from their view)
          - -1 = ally (partner Pokemon) - for support moves like Heal Pulse
          - -2 = self - for moves like Protect, stat boosts
          - 0 = no specific target (spread moves, field moves)
        
        GPT sometimes confuses -1 (ally) as a valid attack target, which is WRONG
        for damage-dealing moves!
        
        CRITICAL: Force switch handling
        - When battle.force_switch is [True, False], only slot 1 needs to switch
        - When battle.force_switch is [False, True], only slot 2 needs to switch
        - GPT might send switch commands for both slots - we must ignore extras!
        """
        print(f"[DEBUG _parse_doubles_decision] Entry: response type={type(response)}, response={response}")
        validation_log = []  # Track validation actions
        
        try:
            # Handle both dict and str input
            if isinstance(response, dict):
                decision = response
            else:
                decision = json.loads(response)
            print(f"[DEBUG _parse_doubles_decision] Parsed decision: {decision}")
            
            orders: List[Optional[BattleOrder]] = [None, None]
            used_switches: set = set()  # Track Pokemon already chosen for switch
            
            # =====================================================================
            # CRITICAL: Detect force_switch situation
            # =====================================================================
            force_switch = getattr(battle, 'force_switch', [False, False])
            if not isinstance(force_switch, list):
                force_switch = [False, False]
            
            # =====================================================================
            # CRITICAL: Determine which slots need actions
            # =====================================================================
            # If ANY force_switch is True, ONLY those slots need action!
            # This is a mid-turn replacement (e.g., after KO), not a normal turn.
            any_force_switch = force_switch[0] or force_switch[1]
            
            slots_needing_action = []
            if any_force_switch:
                # Force switch mode: ONLY slots with force_switch=True need action
                for i in range(2):
                    if force_switch[i]:
                        slots_needing_action.append(i)
                print(f"  [FORCE SWITCH MODE] Only slots {[s+1 for s in slots_needing_action]} need action")
            else:
                # Normal turn: all alive active Pokemon need action
                for i in range(2):
                    if i < len(battle.active_pokemon) and battle.active_pokemon[i] and not battle.active_pokemon[i].fainted:
                        slots_needing_action.append(i)
            
            if force_switch[0] or force_switch[1]:
                print(f"  [FORCE SWITCH] {force_switch}")
                validation_log.append({
                    "force_switch": force_switch,
                    "slots_needing_action": slots_needing_action
                })
            
            # Log raw decision from GPT
            if turn_log:
                turn_log.decision_validation = {
                    "raw_decision": decision,
                    "force_switch": force_switch,
                    "validation_actions": validation_log
                }
            
            for i, slot_key in enumerate(["slot1", "slot2"]):
                # Skip slots that don't need action
                if i not in slots_needing_action:
                    validation_log.append({
                        "slot": i + 1,
                        "action": "skipped",
                        "reason": "Slot not in action list (force_switch or fainted)"
                    })
                    continue
                
                if slot_key not in decision:
                    continue
                
                slot_data = decision[slot_key]
                action_type = slot_data.get("action", "").lower()
                
                if action_type == "move":
                    # =========================================================
                    # CRITICAL: If force_switch is True for this slot, CANNOT move!
                    # =========================================================
                    if force_switch[i]:
                        validation_log.append({
                            "slot": i + 1,
                            "action": "move",
                            "error": "force_switch_active",
                            "message": f"Slot {i+1} must switch (force_switch=True), cannot use move!"
                        })
                        print(f"  [ERR] Slot{i+1} tried to MOVE but force_switch=True! Must switch.")
                        # Will fallback to switch later
                        continue
                    
                    move_id = slot_data.get("move", "").lower()
                    target = slot_data.get("target", 0)
                    tera_requested = slot_data.get("terastallize", False)
                    
                    # =====================================================
                    # CRITICAL: Validate Terastallization
                    # =====================================================
                    # Check if we can actually Terastallize:
                    # 1. Team hasn't used tera yet this battle
                    # 2. This specific Pokemon can tera (can_tera flag)
                    # 3. Pokemon isn't already terastallized
                    team_tera_used = any(p.terastallized for p in battle.team.values())
                    can_tera_now = (
                        not team_tera_used and
                        i < len(battle.can_tera) and
                        battle.can_tera[i] and
                        i < len(battle.active_pokemon) and
                        battle.active_pokemon[i] and
                        not battle.active_pokemon[i].terastallized
                    )
                    
                    tera = tera_requested and can_tera_now
                    
                    if tera_requested and not can_tera_now:
                        validation_log.append({
                            "slot": i + 1,
                            "action": "terastallize_blocked",
                            "reason": "team_tera_used" if team_tera_used else "cannot_tera",
                            "message": f"Slot {i+1} requested tera but cannot (team_used={team_tera_used}, can_tera={battle.can_tera[i] if i < len(battle.can_tera) else False})"
                        })
                        print(f"  [TERA BLOCK] Slot{i+1} requested terastallize but cannot (team_used={team_tera_used})")
                    
                    # Find the move in available moves for this slot
                    if i < len(battle.available_moves) and battle.available_moves[i]:
                        for move in battle.available_moves[i]:
                            if move_id in move.id.lower() or move.id.lower() in move_id:
                                # =====================================================
                                # CRITICAL: Validate and fix target
                                # =====================================================
                                original_target = int(target) if target else 0
                                validated_target = self._validate_move_target(
                                    move=move,
                                    requested_target=original_target,
                                    slot=i,
                                    battle=battle
                                )
                                
                                # Log validation
                                validation_entry = {
                                    "slot": i + 1,
                                    "move": move.id,
                                    "original_target": original_target,
                                    "validated_target": validated_target,
                                    "changed": original_target != validated_target
                                }
                                validation_log.append(validation_entry)
                                
                                print(f"  [TARGET] Slot{i+1} {move.id}: requested={target}, validated={validated_target}")
                                
                                orders[i] = BattleOrder(
                                    move, 
                                    move_target=validated_target,
                                    terastallize=tera
                                )
                                break
                
                elif action_type == "switch":
                    pokemon_name = slot_data.get("pokemon", "").lower()

                    # If GPT omitted pokemon, pick the first available switch for this slot
                    if not pokemon_name:
                        inferred_switches = []
                        if isinstance(battle.available_switches, list) and battle.available_switches:
                            if isinstance(battle.available_switches[0], list):
                                inferred_switches = battle.available_switches[i] if i < len(battle.available_switches) else []
                            else:
                                inferred_switches = battle.available_switches
                        if inferred_switches:
                            pokemon_name = inferred_switches[0].species.lower()
                            # Also propagate back to decision dict for logging/debug
                            slot_data["pokemon"] = inferred_switches[0].species
                    
                    # =========================================================
                    # CRITICAL: Check if trying to switch to already-active Pokemon
                    # =========================================================
                    active_species = set()
                    for active_poke in battle.active_pokemon:
                        if active_poke:
                            active_species.add(active_poke.species.lower())
                    
                    if pokemon_name in active_species:
                        validation_log.append({
                            "slot": i + 1,
                            "action": "switch",
                            "pokemon": pokemon_name,
                            "error": "already_active",
                            "message": f"Cannot switch to {pokemon_name} - already on field!"
                        })
                        print(f"  [ERR] Slot{i+1} switch to {pokemon_name} INVALID (already active!)")
                        # Don't set order - will fallback to default move
                        continue
                    
                    # In doubles, available_switches can be a list of lists (per slot)
                    # or a flat list. Handle both cases.
                    switches = battle.available_switches
                    if isinstance(switches, list) and len(switches) > 0:
                        # Check if it's a list of lists (per-slot switches)
                        if isinstance(switches[0], list):
                            slot_switches = switches[i] if i < len(switches) else []
                        else:
                            # Flat list - all Pokemon available for any slot
                            slot_switches = switches
                    else:
                        slot_switches = []
                    
                    found_switch = False
                    for pokemon in slot_switches:
                        pokemon_species_lower = pokemon.species.lower()
                        if pokemon_name in pokemon_species_lower or pokemon_species_lower in pokemon_name:
                            # Check if this Pokemon is already used by another slot
                            if pokemon.species in used_switches:
                                validation_log.append({
                                    "slot": i + 1,
                                    "action": "switch",
                                    "pokemon": pokemon.species,
                                    "error": "duplicate_switch_target"
                                })
                                print(f"  [WARN] Slot{i+1} switch to {pokemon.species} BLOCKED (already used)")
                                continue  # Try next Pokemon
                            
                            found_switch = True
                            used_switches.add(pokemon.species)
                            validation_log.append({
                                "slot": i + 1,
                                "action": "switch",
                                "pokemon": pokemon.species
                            })
                            orders[i] = BattleOrder(pokemon)
                            break
                    
                    if not found_switch:
                        validation_log.append({
                            "slot": i + 1,
                            "action": "switch",
                            "pokemon": pokemon_name,
                            "error": "not_found",
                            "message": f"Pokemon {pokemon_name} not in available switches"
                        })
                        print(f"  [ERR] Slot{i+1} switch to {pokemon_name} FAILED (not available)")
            
            # Update turn_log with validation
            if turn_log:
                turn_log.decision_validation["validation_actions"] = validation_log
            
            # =====================================================================
            # FALLBACK: If any slot has no valid order, use first available move/switch
            # This prevents the game from hanging on invalid GPT decisions
            # CRITICAL: Respect force_switch - must switch if force_switch[i] is True!
            # =====================================================================
            for i in range(2):
                # Skip slots that don't need action
                if i not in slots_needing_action:
                    continue
                    
                if orders[i] is None:
                    # If force_switch is True, MUST switch - don't try moves!
                    if force_switch[i]:
                        # Must switch - try available switches
                        switches = []
                        if isinstance(battle.available_switches, list) and len(battle.available_switches) > 0:
                            if isinstance(battle.available_switches[0], list):
                                switches = battle.available_switches[i] if i < len(battle.available_switches) else []
                            else:
                                switches = battle.available_switches
                        
                        if switches:
                            for sw in switches:
                                if sw.species not in used_switches:
                                    orders[i] = BattleOrder(sw)
                                    used_switches.add(sw.species)
                                    validation_log.append({
                                        "slot": i + 1,
                                        "action": "force_switch_fallback",
                                        "pokemon": sw.species,
                                        "reason": "force_switch=True, must switch"
                                    })
                                    print(f"  [FALLBACK] Slot{i+1} FORCE SWITCH: Switch to {sw.species}")
                                    break
                    else:
                        # Normal fallback - try move first, then switch
                        # Try to use first available move for this slot
                        if i < len(battle.available_moves) and battle.available_moves[i]:
                            fallback_move = battle.available_moves[i][0]
                            # Get a valid target
                            fallback_target = 1  # Default: opponent slot 1
                            try:
                                targets = battle.get_possible_showdown_targets(fallback_move, battle.active_pokemon[i])
                                if targets:
                                    fallback_target = targets[0]
                            except:
                                pass
                            
                            orders[i] = BattleOrder(fallback_move, move_target=fallback_target)
                            validation_log.append({
                                "slot": i + 1,
                                "action": "fallback_move",
                                "move": fallback_move.id,
                                "target": fallback_target,
                                "reason": "GPT decision invalid - using first available move"
                            })
                            print(f"  [FALLBACK] Slot{i+1}: Using {fallback_move.id} -> target {fallback_target}")
                        
                        # If no moves available, try switch
                        elif battle.available_switches:
                            switches = []
                            if isinstance(battle.available_switches, list) and len(battle.available_switches) > 0:
                                if isinstance(battle.available_switches[0], list):
                                    switches = battle.available_switches[i] if i < len(battle.available_switches) else []
                                else:
                                    switches = battle.available_switches
                            
                            if switches:
                                # Find a switch that's not already used
                                for sw in switches:
                                    if sw.species not in used_switches:
                                        orders[i] = BattleOrder(sw)
                                        used_switches.add(sw.species)
                                        validation_log.append({
                                            "slot": i + 1,
                                            "action": "fallback_switch",
                                            "pokemon": sw.species,
                                            "reason": "GPT decision invalid - using first available switch"
                                        })
                                        print(f"  [FALLBACK] Slot{i+1}: Switch to {sw.species}")
                                        break
            
            # Build DoubleBattleOrder - also validate no duplicate switches
            print(f"[DEBUG _parse_doubles_decision] Building final order: orders[0]={orders[0]}, orders[1]={orders[1]}")
            if orders[0] and orders[1]:
                # Final check: if both are switches to same Pokemon, invalidate second
                first_is_switch = hasattr(orders[0].order, 'species') and not hasattr(orders[0].order, 'base_power')
                second_is_switch = hasattr(orders[1].order, 'species') and not hasattr(orders[1].order, 'base_power')
                if first_is_switch and second_is_switch:
                    if orders[0].order.species == orders[1].order.species:
                        print(f"  [WARN] Both slots switching to same Pokemon! Clearing slot2.")
                        orders[1] = None
                
                if orders[1]:
                    print(f"[DEBUG _parse_doubles_decision] Returning DoubleBattleOrder with both orders")
                    return DoubleBattleOrder(first_order=orders[0], second_order=orders[1])
                else:
                    print(f"[DEBUG _parse_doubles_decision] Returning DoubleBattleOrder with first_order only")
                    return DoubleBattleOrder(first_order=orders[0])
            elif orders[0]:
                print(f"[DEBUG _parse_doubles_decision] Returning DoubleBattleOrder with first_order")
                return DoubleBattleOrder(first_order=orders[0])
            elif orders[1]:
                print(f"[DEBUG _parse_doubles_decision] Returning DoubleBattleOrder with second_order")
                return DoubleBattleOrder(second_order=orders[1])
            
            print(f"[DEBUG _parse_doubles_decision] No valid orders built - falling through to return None")
                
        except Exception as e:
            print(f"[DEBUG _parse_doubles_decision] Exception caught: {e}")
            self.logger.error(f"Parse doubles error: {e}")
            import traceback
            traceback.print_exc()
            if turn_log:
                turn_log.decision_validation = {"error": str(e)}
        print(f"[DEBUG _parse_doubles_decision] Returning None at end")
        return None
    
    def _validate_move_target(
        self, 
        move, 
        requested_target: int, 
        slot: int,
        battle: DoubleBattle
    ) -> int:
        """
        Validate and fix move target to prevent illegal/stupid targeting.
        
        Args:
            move: Move object
            requested_target: GPT's requested target
            slot: Slot index (0 or 1)
            battle: Battle object
            
        Returns:
            Validated target index
        """
        # Get move's valid targets from battle context
        try:
            valid_targets = battle.get_possible_showdown_targets(move, slot)
        except:
            valid_targets = [1, 2]  # Default to opponent targets
        
        # Get move category
        category = move.category.name if hasattr(move, 'category') and move.category else "Physical"
        is_damaging = category in ["Physical", "Special"]
        
        # =========================================================================
        # Rule 0: NO-TARGET MOVES - Check move's target property from game data
        # These moves cannot have a target specified (server rejects with error)
        # =========================================================================
        # Check move's target property (from game data)
        # Possible values: "self", "allySide", "allAdjacent", "allAdjacentFoes", "all", "foeSide", etc.
        # NOTE: All values are lowercase for case-insensitive comparison
        NO_TARGET_TYPES = {
            "self",             # Swords Dance, Calm Mind, Protect, etc.
            "allyside",         # Tailwind, Light Screen, Reflect, etc.
            "alladjacent",      # Earthquake, Discharge, etc. (hits all adjacent)
            "alladjacentfoes",  # Hyper Voice, Dazzling Gleam, Heat Wave, Make It Rain, etc.
            "all",              # Perish Song, etc.
            "foeside",          # Stealth Rock, Spikes, etc.
            "allies",           # Helping Hand targets ally but no target selection
            "randomnormal",     # Sleep Talk, Metronome, etc.
            "scripted",         # Counter, Mirror Coat, etc.
            "allyteam",         # Heal Bell, Aromatherapy, etc.
        }
        
        move_target_type = None
        if hasattr(move, 'target') and move.target:
            move_target_type = str(move.target).lower().replace("_", "").replace("-", "")
        
        # Check valid_targets - if only [0] or empty, it's a no-target move
        is_no_target_move = (
            valid_targets == [0] or 
            valid_targets == [] or
            move_target_type in NO_TARGET_TYPES
        )
        
        if is_no_target_move:
            if requested_target != 0 and requested_target is not None:
                print(f"    [NO-TARGET] Move '{move.id}' (target_type={move_target_type}) doesn't take a target - ignoring target={requested_target}")
            return 0  # No-target moves use 0 or None
        
        # =========================================================================
        # Rule 1: Damaging moves should NEVER target ally (-1) in normal situations
        # =========================================================================
        if is_damaging and requested_target == -1:
            print(f"    [WARN] Damaging move '{move.id}' targeting ally! Fixing...")
            
            # Check if opponent's positions are available
            opp_targets = [t for t in valid_targets if t > 0]
            if opp_targets:
                # Target the first available opponent
                fixed_target = opp_targets[0]
                print(f"    → Fixed target: {requested_target} -> {fixed_target} (opponent)")
                return fixed_target
            elif 0 in valid_targets:
                # Spread move - no specific target
                print(f"    → Fixed target: {requested_target} -> 0 (spread)")
                return 0
        
        # =========================================================================
        # Rule 2: If requested target not in valid targets, pick best alternative
        # =========================================================================
        if requested_target not in valid_targets and valid_targets:
            print(f"    [WARN] Target {requested_target} not in valid targets {valid_targets}")
            
            if is_damaging:
                # Prefer opponent targets for damaging moves
                opp_targets = [t for t in valid_targets if t > 0]
                if opp_targets:
                    fixed_target = opp_targets[0]
                    print(f"    → Fixed target: {requested_target} -> {fixed_target}")
                    return fixed_target
            
            # Just use first valid target
            fixed_target = valid_targets[0]
            print(f"    → Fixed target: {requested_target} -> {fixed_target}")
            return fixed_target
        
        # =========================================================================
        # Rule 3: Check if target Pokemon actually exists (not fainted/empty)
        # =========================================================================
        if requested_target in [1, 2]:
            # Check opponent's position
            opp_pokemon = battle.opponent_active_pokemon
            target_idx = requested_target - 1  # Convert to 0-indexed
            
            if target_idx < len(opp_pokemon):
                target_mon = opp_pokemon[target_idx]
                if target_mon is None or target_mon.fainted:
                    print(f"    [WARN] Target slot {requested_target} is empty/fainted!")
                    # Find alternative opponent target
                    other_target = 2 if requested_target == 1 else 1
                    if other_target in valid_targets:
                        other_idx = other_target - 1
                        if other_idx < len(opp_pokemon) and opp_pokemon[other_idx] and not opp_pokemon[other_idx].fainted:
                            print(f"    → Fixed target: {requested_target} -> {other_target}")
                            return other_target
        
        return requested_target

    async def teampreview(self, battle: AbstractBattle) -> str:
        """
        Handle team preview - GPT 기반 상대 팀 분석 및 선발 선택 (비동기).
        
        Flow:
        1. BattleState 초기화
        2. 내 팀 & 상대 팀 정보 수집
        3. GPT로 상대 팀 전략 분석 (function calling으로 데이터 조회)
        4. GPT가 최적의 선발 + 리드 결정
        5. BattleState에 전략 정보 저장
        
        Note: GPT API 호출은 asyncio.to_thread()로 별도 스레드에서 실행하여
              이벤트 루프를 블로킹하지 않음.
        """
        import asyncio
        
        # =====================================================================
        # BattleState 초기화
        # =====================================================================
        state = self._get_or_create_battle_state(battle)
        
        # 팀 정보 파싱
        my_team_info = self._parse_my_team()
        opponent_team_info = self._parse_opponent_team(battle)
        
        # BattleState에 팀 정보 등록
        my_pokemon_list = [mon["species"] for mon in my_team_info if mon["species"]]
        opp_pokemon_list = [mon["species"] for mon in opponent_team_info if mon["species"]]
        self._init_battle_state_from_preview(battle, state, my_pokemon_list, opp_pokemon_list)
        
        # 콘솔 출력
        self._print_team_preview(my_team_info, opponent_team_info)
        
        # GPT 분석 및 선발 선택 (비동기 - 별도 스레드에서 실행)
        if self.client:
            try:
                # GPT 호출을 별도 스레드에서 실행하여 이벤트 루프 블로킹 방지
                result = await asyncio.to_thread(
                    self._analyze_and_select_team,
                    my_team_info,
                    opponent_team_info,
                    battle,
                    state  # BattleState 전달
                )
                if result:
                    selection, analysis_data = result
                    
                    # BattleState에 전략 정보 저장
                    if analysis_data:
                        state.strategy.set_initial_strategy(analysis_data)
                    
                    # Debug mode: 분석 결과 출력 후 정상 선발 반환
                    if self.debug_mode:
                        print(f"\n[DEBUG] GPT Selection: {selection}")
                        print("[DEBUG] Returning selection (no forfeit in teampreview)")
                    return selection
            except Exception as e:
                self.logger.error(f"Async team analysis error: {e}")
        
        # Fallback: 기본 선발 (앞 4마리, 6-5-4-3번 선발)
        return "/team 6543"
    
    def update_team(self, team):
        """팀 설정 시 파싱해서 저장 (Player 클래스 메서드 오버라이드)."""
        super().update_team(team)
        # 팀이 설정되면 즉시 파싱해서 저장
        self._my_team_info = self._parse_team_from_builder()
    
    def _parse_team_from_builder(self) -> List[Dict]:
        """팀빌더에서 내 팀 정보 파싱 (실제 파싱 로직)."""
        team_info = []
        if not self._team:
            return team_info
        
        packed = self._team.yield_team()
        for i, mon_str in enumerate(packed.split("]"), 1):
            fields = mon_str.split("|")
            # 0:nickname, 1:species, 2:item, 3:ability, 4:moves, 5:nature, 6:evs
            # If species (field 1) is empty, use nickname (field 0) as species
            species = fields[1] if len(fields) > 1 and fields[1] else (fields[0] if len(fields) > 0 else "")
            item = fields[2] if len(fields) > 2 else ""
            ability = fields[3] if len(fields) > 3 else ""
            moves = fields[4].split(",") if len(fields) > 4 and fields[4] else []
            nature = fields[5] if len(fields) > 5 else ""
            evs_str = fields[6] if len(fields) > 6 else ""
            
            # Tera type 파싱 - packed format: field[11] = ",,,,,TeraType" 형태
            tera_type = ""
            if len(fields) > 11 and fields[11]:
                # ",,,,,Water" 형태에서 마지막 값 추출
                endstring_parts = fields[11].split(",")
                for part in reversed(endstring_parts):
                    if part and part not in ["G", "S", ""]:
                        tera_type = part
                        break
            
            # EV 파싱
            ev_names = ["hp", "atk", "def", "spa", "spd", "spe"]
            evs = {}
            ev_parts = evs_str.split(",") if evs_str else []
            for j, ev in enumerate(ev_parts):
                if ev and j < len(ev_names):
                    evs[ev_names[j]] = int(ev) if ev.isdigit() else 0
            
            # 포켓몬 타입 정보 추가 (pokedex에서)
            types = []
            if species.lower() in self.gen9_pokedex:
                types = self.gen9_pokedex[species.lower()].get("types", [])
            
            team_info.append({
                "slot": i,
                "species": species,
                "item": item,
                "ability": ability,
                "moves": moves[:4],
                "nature": nature,
                "evs": evs,
                "tera_type": tera_type,
                "types": types
            })
        
        return team_info
    
    def _parse_my_team(self) -> List[Dict]:
        """내 팀 정보 반환 (이미 파싱된 정보 사용)."""
        return self._my_team_info
    
    def _parse_opponent_team(self, battle: AbstractBattle) -> List[Dict]:
        """상대 팀 정보 파싱 (공개 정보만)."""
        team_info = []
        for i, (species, mon) in enumerate(battle.opponent_team.items(), 1):
            types = [t.name for t in mon.types if t]
            
            # Base stats from pokedex
            base_stats = {}
            species_key = species.lower().replace(" ", "")
            if species_key in self.gen9_pokedex:
                base_stats = self.gen9_pokedex[species_key].get("baseStats", {})
            
            # 가능한 특성들
            possible_abilities = []
            if species_key in self.gen9_pokedex:
                abilities_data = self.gen9_pokedex[species_key].get("abilities", {})
                possible_abilities = list(abilities_data.values())
            
            team_info.append({
                "slot": i,
                "species": species,
                "types": types,
                "base_stats": base_stats,
                "possible_abilities": possible_abilities
            })
        
        return team_info
    
    def _print_team_preview(self, my_team: List[Dict], opponent_team: List[Dict]):
        """팀 프리뷰 콘솔 출력."""
        print("=" * 70)
        print("TEAM PREVIEW")
        print("=" * 70)
        
        print("【 MY TEAM 】")
        for mon in my_team:
            ev_display = " ".join([f"{k.upper()}:{v}" for k, v in mon["evs"].items() if v > 0])
            tera_str = f" | Tera: {mon['tera_type']}" if mon["tera_type"] else ""
            print(f"  {mon['slot']}. {mon['species']:<15} @ {mon['item']:<15} | {mon['ability']:<15}{tera_str}")
            print(f"     Moves: {', '.join(mon['moves'])}")
        
        print("-" * 70)
        
        print("【 OPPONENT TEAM 】")
        for mon in opponent_team:
            types_str = "/".join(mon["types"])
            print(f"  {mon['slot']}. {mon['species']:<20} | Type: {types_str}")
        
        print("=" * 70)
    
    def _analyze_and_select_team(
        self, 
        my_team: List[Dict], 
        opponent_team: List[Dict],
        battle: AbstractBattle,
        state: Optional[BattleState] = None
    ) -> Optional[tuple]:
        """
        GPT로 상대 팀 분석 및 선발 선택.
        
        Function Calling으로:
        - get_pokemon_info: 포켓몬 상세 정보 조회
        - get_move_info: 기술 정보 조회
        - get_type_matchup: 타입 상성 조회
        - get_common_sets: 일반적인 세팅 조회 (usage stats 기반)
        
        Returns:
            tuple: (selection_string, analysis_data) or None
            - selection_string: "/team 1234" 형식
            - analysis_data: GPT 분석 결과 (전략 정보)
        """
        import sys
        print("\n[GPT] Starting team analysis...", flush=True)
        sys.stdout.flush()
        # =====================================================================
        # FUNCTION CALLING - Team Analysis Tools
        # 팀프리뷰 분석용 함수 정의 (수정 시 이 섹션 편집)
        # =====================================================================
        analysis_tools = [
            {
                "type": "function",
                "name": "get_pokemon_info",
                "description": "Get detailed information about a Pokemon including base stats, types, and possible abilities. Use for Stage 1 information collection.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "generation": {"type": "integer", "description": "Pokemon generation (e.g., 9 for Gen 9)"},
                        "pokemon": {"type": "string", "description": "Pokemon name (e.g., 'urshifu', 'incineroar')"}
                    },
                    "required": ["generation","pokemon"]
                }
            },
            {
                "type": "function",
                "name": "get_move_info",
                "description": "Get information about a move including type, power, priority, and effects. Use for Stage 1.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "generation": {"type": "integer", "description": "Pokemon generation (e.g., 9 for Gen 9)"},
                        "move": {"type": "string", "description": "Move name (e.g., 'fakeout', 'closecombat')"}
                    },
                    "required": ["generation","move"]
                }
            },
            {
                "type": "function",
                "name": "get_item_info",
                "description": "Get information about a held item including effects and common users. Use for Stage 1.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "generation": {"type": "integer", "description": "Pokemon generation (e.g., 9 for Gen 9)"},
                        "item": {"type": "string", "description": "Item name (e.g., 'lifeorb', 'focussash')"}
                    },
                    "required": ["generation","item"]
                }
            },
            {
                "type": "function",
                "name": "get_usage_stats",
                "description": "**REQUIRED** - Get VGC usage statistics for opponent Pokemon. Returns: common items (Choice Scarf, Focus Sash, etc.), moves (Fake Out, Protect, Trick Room), abilities, EV spreads, Tera types with usage percentages. You MUST call this for all opponent Pokemon before making decisions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "generation": {"type": "integer", "description": "Pokemon generation (9 for current VGC)"},
                        "regulation": {"type": "string", "description": "Battle regulation letter (e.g., 'H' for Reg H)"},
                        "gametype": {"type": "string", "description": "Game type: 'double' for VGC"},
                        "pokemon": {"type": "string", "description": "Pokemon name to look up"}
                    },
                    "required": ["generation","regulation","gametype","pokemon"]
                }
            },
            {
                "type": "function",
                "name": "get_type_matchup",
                "description": "Get type effectiveness multiplier (attacking type vs defending types). Use for matchup analysis.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "attack_type": {"type": "string", "description": "Attacking type"},
                        "defend_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Defending Pokemon's types"
                        }
                    },
                    "required": ["attack_type", "defend_types"]
                }
            },
            {
                "type": "function",
                "name": "analyze_speed_tiers",
                "description": "Compare speed stats between Pokemon to determine turn order. Returns min/max possible speeds at Lv50.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pokemon_list": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of Pokemon to compare speeds"
                        }
                    },
                    "required": ["pokemon_list"]
                }
            }
        ]
        # =====================================================================
        # END FUNCTION CALLING - Team Analysis Tools
        # =====================================================================
        
        # 시스템 프롬프트
        system_prompt = self._get_team_analysis_system_prompt(battle)
        
        # 유저 프롬프트: 양 팀 정보
        user_prompt = self._build_team_analysis_prompt(my_team, opponent_team, battle)
        
        # =====================================================================
        # Debug Mode: Check cache before API call
        # =====================================================================
        cache_data = {
            "my_team": [m["species"] for m in my_team],
            "opponent_team": [o["species"] for o in opponent_team]
        }
        cache_key = self._get_cache_key("teampreview", cache_data)
        
        cached = self._get_cached_response(cache_key)
        if cached:
            selection_str = self._format_team_selection(cached, my_team, state)
            return (selection_str, cached)  # Return tuple
        
        try:
            # GPT 호출 (function calling 포함)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            result_data = self._call_llm(
                model=self.backend,
                messages=messages,
                tools=analysis_tools,
                temperature=0.5,
                max_tokens=2000
            )
            
            # Function calls 처리 (with state for storing predictions)
            result = self._process_team_analysis_response_v2(result_data, analysis_tools, my_team, opponent_team, state)
            
            if result:
                # Cache the result for debug mode
                self._cache_response(cache_key, result)
                selection_str = self._format_team_selection(result, my_team, state)
                return (selection_str, result)  # Return tuple
            
        except Exception as e:
            self.logger.error(f"Team analysis error: {e}")
        
        return None
    
    def _get_team_analysis_system_prompt(self, battle: AbstractBattle) -> str:
        """팀 분석용 시스템 프롬프트 - 배틀 포맷에 따라 동적 생성."""
        # 배틀 포맷에서 규칙 추출
        team_size = len(battle.team) if battle.team else 6
        max_team_size = battle.max_team_size if hasattr(battle, 'max_team_size') and battle.max_team_size else team_size
        is_doubles = isinstance(battle, DoubleBattle)
        lead_count = 2 if is_doubles else 1
        
        return f"""You are an expert Pokemon VGC battle analyst.

# BATTLE FORMAT
- Team size: {team_size} → Bring: {max_team_size}
- Type: {"Doubles" if is_doubles else "Singles"} (Lead: {lead_count})

# CRITICAL: MANDATORY TOOL USAGE

## Step 1: REQUIRED - Get Usage Stats (call for ALL 6 opponent Pokemon)
**You MUST call `get_usage_stats` for ALL opponent Pokemon first.**

Why? You don't know:
- Opponent's items (Choice Scarf? Focus Sash? Life Orb?)
- Opponent's moves (Do they have Protect? Fake Out? Trick Room?)
- Opponent's abilities (Intimidate? Prankster? Drought?)
- Opponent's EV spreads (How fast are they really?)
- Opponent's Tera types (Water Tera for defensive pivot?)

## Step 2: VERIFY - Confirm Exact Numbers When Needed
After getting stats, if you're unsure about:
- **Speed comparisons**: Use `analyze_speed_tiers` to check exact speed ranges
  - Example: "Is my Dragapult faster than their Scarf Urshifu?"
- **Type matchups**: Use `get_type_matchup` for precise damage calculations
  - Example: "Is Dragon/Ground hit 4x by Ice or 2x?"
- **Move/Item effects**: Use `get_move_info` or `get_item_info` for exact mechanics
  - Example: "Does Assault Vest block status moves?"

Don't guess - verify with tools!

# TOOL CALLING RULES

1. **FIRST BATCH**: Call `get_usage_stats` for ALL 6 opponent Pokemon simultaneously
2. **SECOND BATCH** (if needed): Call verification tools (speed tiers, type matchups)
3. **FINALLY**: Make your selection based on confirmed data

# OUTPUT FORMAT (JSON)

After gathering and verifying data, respond with:

```json
{{
    "opponent_analysis": {{
        "archetype": "<team style based on usage data>",
        "key_threats": ["pokemon1", "pokemon2"],
        "predicted_leads": ["pokemon1", "pokemon2"],
        "expected_strategy": "<based on common moves/items from usage stats>"
    }},
    "my_strategy": {{
        "game_plan": "<how to counter their likely sets>",
        "lead_reasoning": "<why these leads work against their common items/moves>",
        "backup_plan": "<what if they run uncommon sets>"
    }},
    "selection": {{
        "bring": [slot1, slot2, slot3, slot4],
        "lead": [slot1, slot2]
    }}
}}
```"""

    def _build_team_analysis_prompt(
        self, 
        my_team: List[Dict], 
        opponent_team: List[Dict],
        battle: AbstractBattle
    ) -> str:
        """팀 분석용 유저 프롬프트 생성."""
        # 배틀에서 선발 규칙 가져오기
        team_size = len(my_team)
        max_team_size = battle.max_team_size if hasattr(battle, 'max_team_size') and battle.max_team_size else team_size
        
        prompt = f"## MY TEAM (Choose {max_team_size} from these {team_size})\n\n"
        
        for mon in my_team:
            prompt += f"**Slot {mon['slot']}: {mon['species']}**\n"
            prompt += f"- Item: {mon['item']}\n"
            prompt += f"- Ability: {mon['ability']}\n"
            prompt += f"- Tera Type: {mon['tera_type'] or 'Unknown'}\n"
            prompt += f"- Moves: {', '.join(mon['moves'])}\n"
            if mon['evs']:
                ev_str = ", ".join([f"{k}: {v}" for k, v in mon['evs'].items() if v > 0])
                prompt += f"- EVs: {ev_str}\n"
            prompt += "\n"
        
        prompt += "---\n\n## OPPONENT TEAM\n\n"
        
        for mon in opponent_team:
            prompt += f"**{mon['species']}**\n"
            prompt += f"- Types: {'/'.join(mon['types'])}\n"
            if mon['base_stats']:
                stats = mon['base_stats']
                prompt += f"- Base Stats: HP:{stats.get('hp',0)} Atk:{stats.get('atk',0)} Def:{stats.get('def',0)} "
                prompt += f"SpA:{stats.get('spa',0)} SpD:{stats.get('spd',0)} Spe:{stats.get('spe',0)}\n"
            if mon['possible_abilities']:
                prompt += f"- Possible Abilities: {', '.join(mon['possible_abilities'])}\n"
            prompt += "- Item: ❓ UNKNOWN - Use get_usage_stats\n"
            prompt += "- Moves: ❓ UNKNOWN - Use get_usage_stats\n"
            prompt += "- Tera Type: ❓ UNKNOWN - Use get_usage_stats\n"
            prompt += "\n"
        
        prompt += "---\n\n"
        prompt += "⚠️ **IMPORTANT**: You don't know opponent's items, moves, or Tera types!\n"
        prompt += "**FIRST ACTION**: Call `get_usage_stats` for ALL 6 opponent Pokemon to get this critical information.\n"
        prompt += "Only then can you make an informed team selection."
        
        return prompt
    
    def _process_team_analysis_response(
        self, 
        response, 
        tools: List[Dict],
        my_team: List[Dict],
        opponent_team: List[Dict],
        state: Optional[BattleState] = None
    ) -> Optional[Dict]:
        """
        GPT 응답 처리 (function calling).
        
        - 이상적으로 1라운드에 모든 tool call을 병렬 실행
        - 최대 3라운드까지만 허용 (그 이상은 비효율)
        - usage stats 결과는 BattleState에도 저장
        """
        MAX_ITERATIONS = 5  # 안전장치
        
        for iteration in range(MAX_ITERATIONS):
            # 응답에서 function calls 또는 텍스트 추출
            function_calls = []
            final_text = None
            
            if response.output:
                for item in response.output:
                    if item.type == "function_call":
                        function_calls.append({
                            "id": item.call_id,
                            "name": item.name,
                            "arguments": item.arguments
                        })
                    elif item.type == "message":
                        for content in item.content:
                            if content.type == "output_text":
                                final_text = content.text
            
            # Function calls가 없으면 최종 응답
            if not function_calls:
                if final_text:
                    try:
                        import re
                        json_match = re.search(r'\{[\s\S]*\}', final_text)
                        if json_match:
                            result = json.loads(json_match.group())
                            if "selection" in result:
                                return result
                    except json.JSONDecodeError:
                        self.logger.warning(f"Failed to parse team selection JSON")
                return None
            
            # Function calls 실행 (모두 병렬로)
            print(f"\n  [Batch {iteration+1}] Executing {len(function_calls)} tool calls in parallel...")
            tool_results = []
            for fc in function_calls:
                result = self._execute_analysis_tool(fc["name"], fc["arguments"])
                
                # =========================================================
                # Store ALL tool results in TeamPreviewCache for battle phase
                # This prevents redundant API calls during battle
                # =========================================================
                if state:
                    try:
                        args = json.loads(fc["arguments"])
                        pokemon_name = args.get("pokemon", "").lower().replace(" ", "").replace("-", "")
                        
                        if fc["name"] == "get_usage_stats" and "moves" in result:
                            # Store full usage stats for battle phase
                            state.cache_pokemon_data(pokemon_name, "usage_stats", result)
                            
                            # Also store common items separately for quick lookup
                            if "items" in result:
                                state.cache_pokemon_data(pokemon_name, "common_items", result["items"])
                            
                            # Store speed tier from usage stats if available
                            if "spreads" in result:
                                state.cache_pokemon_data(pokemon_name, "speed_tier", {
                                    "spreads": result["spreads"][:3]  # Top 3 spreads
                                })
                            
                            # Find matching opponent pokemon and add predicted moves
                            for opp in opponent_team:
                                opp_norm = self._normalize_name(opp["species"])
                                if opp_norm == pokemon_name or pokemon_name in opp_norm:
                                    poke_state = state.opponent_team.get(opp_norm)
                                    if poke_state:
                                        # Add predicted moves from usage stats
                                        moves_data = result.get("moves", [])
                                        for move in moves_data:
                                            move_name = move.get("name", "")
                                            usage = move.get("usage", 0)
                                            if move_name and usage > 0:
                                                move_id = move_name.lower().replace(" ", "").replace("-", "")
                                                poke_state.add_predicted_move(move_id, move_name, usage)
                                        
                                        # Store common moves for quick lookup
                                        state.cache_pokemon_data(pokemon_name, "common_moves", 
                                            [m["name"] for m in moves_data[:6]])
                                        
                                        print(f"    → Cached usage stats for {opp['species']} (moves, items, spreads)")
                                    break
                        
                        elif fc["name"] == "get_pokemon_info" and "baseStats" in result:
                            # Store base stats for damage calculation during battle
                            state.cache_pokemon_data(pokemon_name, "base_stats", {
                                "types": result.get("types", []),
                                "hp": result["baseStats"].get("hp", 80),
                                "atk": result["baseStats"].get("atk", 80),
                                "def": result["baseStats"].get("def", 80),
                                "spa": result["baseStats"].get("spa", 80),
                                "spd": result["baseStats"].get("spd", 80),
                                "spe": result["baseStats"].get("spe", 80),
                                "abilities": result.get("abilities", {})
                            })
                            print(f"    → Cached base stats for {pokemon_name}")
                        
                        elif fc["name"] == "analyze_speed_tiers":
                            # Store speed tier analysis
                            pokemon_list = args.get("pokemon_list", [])
                            if "speed_tiers" in result:
                                for tier in result["speed_tiers"]:
                                    poke = tier.get("pokemon", "").lower().replace(" ", "").replace("-", "")
                                    if poke:
                                        state.cache_pokemon_data(poke, "speed_tier", {
                                            "min": tier.get("min_speed", 0),
                                            "max": tier.get("max_speed", 0)
                                        })
                                print(f"    → Cached speed tiers for {len(result['speed_tiers'])} Pokemon")
                    
                    except Exception as e:
                        self.logger.debug(f"Failed to cache tool result: {e}")
                
                tool_results.append({
                    "type": "function_call_output",
                    "call_id": fc["id"],
                    "output": json.dumps(result)
                })
            
            # 간략한 요약 출력
            tool_summary = {}
            for fc in function_calls:
                tool_summary[fc["name"]] = tool_summary.get(fc["name"], 0) + 1
            print(f"    Tools: {', '.join([f'{k}×{v}' for k, v in tool_summary.items()])}")
            
            # 다음 요청
            try:
                response = self.client.responses.create(
                    model=self.backend,
                    previous_response_id=response.id,
                    input=tool_results,
                    tools=tools,
                    temperature=0.5,
                    max_output_tokens=2000
                )
                
                if hasattr(response, 'usage') and response.usage:
                    self._tokens['input'] += getattr(response.usage, 'input_tokens', 0)
                    self._tokens['output'] += getattr(response.usage, 'output_tokens', 0)
                    
            except Exception as e:
                self.logger.error(f"Follow-up request error: {e}")
                break
        
        return None

    def _process_team_analysis_response_v2(self, result_data: Dict, tools: List, 
                                            my_team: List[Dict], opponent_team: List[Dict],
                                            state: 'TeamPreviewCache') -> Optional[Dict]:
        """
        Process team analysis response from _call_llm (v2 version).
        Handles both Ollama (no tool calls) and OpenAI (with tool calls).
        """
        MAX_ITERATIONS = 5
        
        for iteration in range(MAX_ITERATIONS):
            content = result_data.get("content")
            tool_calls = result_data.get("tool_calls")
            
            # No tool calls - try to parse final response
            if not tool_calls:
                if content:
                    try:
                        import re
                        json_match = re.search(r'\{[\s\S]*\}', content)
                        if json_match:
                            result = json.loads(json_match.group())
                            if "selection" in result:
                                return result
                    except json.JSONDecodeError:
                        self.logger.warning(f"Failed to parse team selection JSON")
                return None
            
            # For Ollama, we can't do tool calls
            if self.base_url:
                self.logger.warning("Tool calls not supported with Ollama")
                return None
            
            # Process tool calls
            print(f"\n  [Batch {iteration+1}] Executing {len(tool_calls)} tool calls in parallel...")
            tool_results = []
            
            for fc in tool_calls:
                try:
                    args = json.loads(fc["arguments"]) if isinstance(fc["arguments"], str) else fc["arguments"]
                except json.JSONDecodeError:
                    args = {}
                
                # Execute tool
                result = self._execute_analysis_tool(fc["name"], json.dumps(args))
                
                # Cache results in state
                try:
                    if fc["name"] == "get_pokemon_info":
                        pokemon_name = args.get("pokemon", "").lower().replace(" ", "").replace("-", "")
                        if pokemon_name and "baseStats" in result:
                            state.cache_pokemon_data(pokemon_name, "stats", result["baseStats"])
                    elif fc["name"] == "analyze_speed_tiers":
                        if "speed_tiers" in result:
                            for tier in result["speed_tiers"]:
                                poke = tier.get("pokemon", "").lower().replace(" ", "").replace("-", "")
                                if poke:
                                    state.cache_pokemon_data(poke, "speed_tier", {
                                        "min": tier.get("min_speed", 0),
                                        "max": tier.get("max_speed", 0)
                                    })
                except Exception as e:
                    self.logger.debug(f"Failed to cache tool result: {e}")
                
                tool_results.append({
                    "type": "function_call_output",
                    "call_id": fc["id"],
                    "output": json.dumps(result)
                })
            
            # Summary
            tool_summary = {}
            for fc in tool_calls:
                tool_summary[fc["name"]] = tool_summary.get(fc["name"], 0) + 1
            print(f"    Tools: {', '.join([f'{k}×{v}' for k, v in tool_summary.items()])}")
            
            # Follow-up request
            try:
                response_id = result_data.get("response_id")
                if not response_id:
                    self.logger.error("No response_id for follow-up call")
                    break
                
                result_data = self._call_llm_followup(
                    model=self.backend,
                    previous_response_id=response_id,
                    tool_results=tool_results,
                    tools=tools,
                    temperature=0.5,
                    max_tokens=2000
                )
                
                if not result_data:
                    break
                    
            except Exception as e:
                self.logger.error(f"Follow-up request error: {e}")
                break
        
        return None
    
    def _execute_analysis_tool(self, tool_name: str, arguments: str) -> Dict:
        """분석 도구 실행."""
        # =====================================================================
        # FUNCTION CALLING - Tool Execution Handlers
        # 팀프리뷰 분석용 함수 실행 로직 (수정 시 이 섹션 편집)
        # =====================================================================
        try:
            args = json.loads(arguments)
        except json.JSONDecodeError:
            return {"error": "Invalid arguments"}
        
        if tool_name == "get_pokemon_info":
            generation = args.get("generation", 9)
            pokemon = args.get("pokemon", "")

            # PKHeX Core를 통해 실제 포켓몬 데이터 가져오기
            try:
                pokemon_data = pkhex_core.get_pokemon_info(pokemon, language="en")

                if pokemon_data:
                    return pokemon_data
                else:
                    # 포켓몬을 찾지 못한 경우 기본값 반환
                    return {
                        "species": pokemon,
                        "types": ["Unknown"],
                        "baseStats": {"hp": 80, "atk": 80, "def": 80, "spa": 80, "spd": 80, "spe": 80},
                        "abilities": {},
                        "note": "No detailed data available for this Pokemon"
                    }
            except Exception as e:
                # PKHeX Core 오류 발생 시 기본값 반환
                return {
                    "species": pokemon,
                    "types": ["Unknown"],
                    "baseStats": {"hp": 80, "atk": 80, "def": 80, "spa": 80, "spd": 80, "spe": 80},
                    "abilities": {},
                    "note": f"Error retrieving Pokemon data: {str(e)}"
                }
        
        elif tool_name == "get_move_info":
            generation = args.get("generation", 9)
            move_name = args.get("move", "").lower().replace(" ", "").replace("-", "")
            
            # gen9moves.json에서 기술 정보 가져오기
            if hasattr(self, 'gen9_moves') and self.gen9_moves:
                move_data = self.gen9_moves.get(move_name)
                if move_data:
                    # target 필드 해석 추가
                    target_type = move_data.get("target", "normal")
                    target_desc = {
                        "normal": "Single target - must specify target (1=opp left, 2=opp right)",
                        "self": "Self-targeting - no target needed (Protect, Swords Dance, etc.)",
                        "allAdjacentFoes": "SPREAD MOVE - hits all opponents, NO target allowed!",
                        "allAdjacent": "SPREAD MOVE - hits all adjacent (including ally), NO target!",
                        "allySide": "Ally side effect - no target needed (Tailwind, Light Screen)",
                        "foeSide": "Foe side effect - no target needed (Stealth Rock, Spikes)",
                        "all": "Field effect - no target needed (Perish Song)",
                        "adjacentAlly": "Ally only - target=-1 for ally (Helping Hand)",
                        "adjacentAllyOrSelf": "Self or ally - target=-2 for self, -1 for ally",
                        "any": "Can target anyone on field",
                        "adjacentFoe": "Single opponent - target=1 or 2",
                        "randomNormal": "Random target - no target needed",
                        "scripted": "Counter move - no target needed"
                    }.get(target_type, f"Unknown target type: {target_type}")
                    
                    # 카테고리 정규화
                    category = move_data.get("category", "Unknown")
                    
                    return {
                        "name": move_data.get("name", move_name),
                        "type": move_data.get("type", "Unknown"),
                        "category": category,
                        "basePower": move_data.get("basePower", 0),
                        "accuracy": move_data.get("accuracy", 100),
                        "priority": move_data.get("priority", 0),
                        "target": target_type,
                        "target_description": target_desc,
                        "is_spread_move": target_type in ["allAdjacentFoes", "allAdjacent", "all", "allySide", "foeSide"],
                        "pp": move_data.get("pp", 0),
                        "flags": move_data.get("flags", {}),
                        "secondary": move_data.get("secondary"),
                        "desc": move_data.get("desc", move_data.get("shortDesc", "No description available"))
                    }
            
            # Fallback: 기본 응답
            return {
                "name": move_name,
                "type": "Unknown",
                "category": "Unknown",
                "basePower": 0,
                "accuracy": 100,
                "priority": 0,
                "target": "normal",
                "target_description": "Unknown - data not found",
                "is_spread_move": False,
                "desc": "No detailed data available for this move. Check move name spelling."
            }
        
        elif tool_name == "get_item_info":
            generation = args.get("generation", 9)
            item_name = args.get("item", "").lower().replace(" ", "").replace("-", "")
            
            # item_effect.json에서 아이템 정보 가져오기
            if hasattr(self, 'item_effect') and self.item_effect:
                item_data = self.item_effect.get(item_name)
                if item_data:
                    # 아이템 카테고리 분류 (VGC에서 중요한 정보)
                    name_lower = item_name.lower()
                    category = "other"
                    
                    # 카테고리 자동 분류
                    if "choice" in name_lower:
                        category = "choice_lock"  # 기술 고정
                    elif "berry" in name_lower:
                        category = "berry"  # 열매
                    elif name_lower in ["lifeorb", "expertbelt", "muscleband", "wiseglasses", "metronome"]:
                        category = "damage_boost"
                    elif name_lower in ["focussash", "focusband", "airballoon"]:
                        category = "survival"
                    elif name_lower in ["assaultvest", "eviolite", "rockyhelmet", "leftovers"]:
                        category = "bulk"
                    elif name_lower in ["safetygoggles", "covertcloak", "clearamulet", "protectivepads", "abilityshield"]:
                        category = "utility_protection"
                    elif name_lower in ["ejectbutton", "ejectpack", "redcard", "shedshell"]:
                        category = "switching"
                    elif name_lower in ["whiteherb", "mentalherb", "powerherb", "luminousmoss", "snowball", "absorbbulb", "cellbattery", "weaknesspolicy"]:
                        category = "consumable_boost"
                    elif "orb" in name_lower:
                        category = "signature_orb"
                    elif name_lower in ["throatspray", "roomservice", "blunderpolicy"]:
                        category = "consumable_boost"
                    elif name_lower in ["terrainextender", "lightclay", "gripclaw", "bindingband"]:
                        category = "duration_extend"
                    elif "gem" in name_lower:
                        category = "type_gem"
                    elif "plate" in name_lower or "memory" in name_lower:
                        category = "type_change"
                    elif name_lower in ["ironball", "laggingtail", "fullincense", "ringtarget"]:
                        category = "trick_room_support"
                    
                    return {
                        "name": item_data.get("name", item_name),
                        "effect": item_data.get("effect", "No effect description available."),
                        "category": category,
                        "vgc_notes": self._get_item_vgc_notes(item_name)
                    }
            
            # Fallback: 기본 응답
            return {
                "name": item_name,
                "effect": "No detailed information available for this item.",
                "category": "unknown",
                "vgc_notes": "Check item name spelling."
            }
        
        elif tool_name == "get_type_matchup":
            attack_type = args.get("attack_type", "").lower()
            defend_types = [t.lower() for t in args.get("defend_types", [])]
            
            # 타입 상성표 로드
            try:
                with open("src/data/static/typechart/gen9typechart.json", "r") as f:
                    typechart = json.load(f)
                
                multiplier = 1.0
                for def_type in defend_types:
                    if def_type in typechart and attack_type in typechart[def_type].get("damageTaken", {}):
                        effect = typechart[def_type]["damageTaken"][attack_type]
                        if effect == 1:  # Super effective
                            multiplier *= 2
                        elif effect == 2:  # Not very effective
                            multiplier *= 0.5
                        elif effect == 3:  # Immune
                            multiplier *= 0
                
                effectiveness = "neutral"
                if multiplier >= 2:
                    effectiveness = "super effective"
                elif multiplier == 0:
                    effectiveness = "immune"
                elif multiplier < 1:
                    effectiveness = "not very effective"
                
                return {
                    "attack_type": attack_type,
                    "defend_types": defend_types,
                    "multiplier": multiplier,
                    "effectiveness": effectiveness
                }
            except:
                return {"error": "Could not load type chart"}
        
        elif tool_name == "analyze_speed_tiers":
            pokemon_list = args.get("pokemon_list", [])
            # TODO: 실제 pokedex 연동
            # 현재는 하드코딩된 샘플 데이터 반환
            speed_db = {
                "miraidon": 135, "koraidon": 135, "fluttermane": 135, "chienpao": 135,
                "urshifu": 97, "urshifurapidstrike": 97, "landorus": 101,
                "rillaboom": 85, "incineroar": 60, "amoonguss": 30
            }
            speeds = []
            
            for pokemon in pokemon_list:
                pokemon_key = pokemon.lower().replace(" ", "").replace("-", "")
                base_speed = speed_db.get(pokemon_key, 80)
                speeds.append({
                    "pokemon": pokemon,
                    "base_speed": base_speed,
                    "min_speed": int((2 * base_speed + 31) * 0.9),  # 0 EV, -Speed nature
                    "max_speed": int(((2 * base_speed + 31 + 63) + 5) * 1.1)  # 252 EV, +Speed nature, Lv50
                })
            
            speeds.sort(key=lambda x: x["base_speed"], reverse=True)
            return {"speed_tiers": speeds}
        
        elif tool_name == "get_usage_stats":
            generation = args.get("generation", 9)
            regulation = args.get("regulation", "H")
            gametype = args.get("gametype", "double")
            pokemon = args.get("pokemon", "").lower().replace(" ", "")
            
            # TODO: 실제 사용률 데이터 파일 연동
            # 현재는 하드코딩된 샘플 데이터 반환
            usage_stats_db = {
                "urshifurapidstrike": {
                    "items": [
                        {"name": "Choice Scarf", "usage": 45.2},
                        {"name": "Focus Sash", "usage": 32.1},
                        {"name": "Mystic Water", "usage": 12.5}
                    ],
                    "abilities": [
                        {"name": "Unseen Fist", "usage": 100.0}
                    ],
                    "moves": [
                        {"name": "Surging Strikes", "usage": 98.5},
                        {"name": "Close Combat", "usage": 85.2},
                        {"name": "Aqua Jet", "usage": 62.3},
                        {"name": "Protect", "usage": 55.1},
                        {"name": "U-turn", "usage": 42.8}
                    ],
                    "tera_types": [
                        {"type": "Water", "usage": 52.3},
                        {"type": "Stellar", "usage": 28.4},
                        {"type": "Poison", "usage": 11.2}
                    ],
                    "spreads": [
                        {"nature": "Jolly", "evs": "252 Atk / 4 Def / 252 Spe", "usage": 68.5},
                        {"nature": "Adamant", "evs": "252 Atk / 4 SpD / 252 Spe", "usage": 21.3}
                    ]
                },
                "incineroar": {
                    "items": [
                        {"name": "Safety Goggles", "usage": 38.5},
                        {"name": "Sitrus Berry", "usage": 25.2},
                        {"name": "Assault Vest", "usage": 18.7}
                    ],
                    "abilities": [
                        {"name": "Intimidate", "usage": 99.8}
                    ],
                    "moves": [
                        {"name": "Fake Out", "usage": 96.2},
                        {"name": "Flare Blitz", "usage": 89.5},
                        {"name": "Knock Off", "usage": 82.1},
                        {"name": "Parting Shot", "usage": 78.3},
                        {"name": "Protect", "usage": 45.2}
                    ],
                    "tera_types": [
                        {"type": "Ghost", "usage": 42.1},
                        {"type": "Water", "usage": 28.5},
                        {"type": "Grass", "usage": 15.3}
                    ],
                    "spreads": [
                        {"nature": "Careful", "evs": "252 HP / 4 Atk / 252 SpD", "usage": 45.2},
                        {"nature": "Adamant", "evs": "252 HP / 252 Atk / 4 SpD", "usage": 32.1}
                    ]
                },
                "rillaboom": {
                    "items": [
                        {"name": "Assault Vest", "usage": 42.3},
                        {"name": "Miracle Seed", "usage": 28.5},
                        {"name": "Choice Band", "usage": 15.2}
                    ],
                    "abilities": [
                        {"name": "Grassy Surge", "usage": 99.9}
                    ],
                    "moves": [
                        {"name": "Grassy Glide", "usage": 95.2},
                        {"name": "Wood Hammer", "usage": 72.5},
                        {"name": "Fake Out", "usage": 68.3},
                        {"name": "U-turn", "usage": 55.2},
                        {"name": "Protect", "usage": 48.1}
                    ],
                    "tera_types": [
                        {"type": "Fire", "usage": 35.2},
                        {"type": "Poison", "usage": 28.1},
                        {"type": "Stellar", "usage": 18.5}
                    ],
                    "spreads": [
                        {"nature": "Adamant", "evs": "252 HP / 252 Atk / 4 SpD", "usage": 55.2},
                        {"nature": "Brave", "evs": "252 HP / 252 Atk / 4 SpD", "usage": 25.1}
                    ]
                },
                "miraidon": {
                    "items": [
                        {"name": "Life Orb", "usage": 48.5},
                        {"name": "Choice Specs", "usage": 32.1},
                        {"name": "Magnet", "usage": 12.3}
                    ],
                    "abilities": [
                        {"name": "Hadron Engine", "usage": 100.0}
                    ],
                    "moves": [
                        {"name": "Electro Drift", "usage": 98.2},
                        {"name": "Draco Meteor", "usage": 92.5},
                        {"name": "Protect", "usage": 72.3},
                        {"name": "Thunderbolt", "usage": 45.2},
                        {"name": "Volt Switch", "usage": 38.5}
                    ],
                    "tera_types": [
                        {"type": "Fairy", "usage": 42.5},
                        {"type": "Electric", "usage": 28.3},
                        {"type": "Steel", "usage": 15.2}
                    ],
                    "spreads": [
                        {"nature": "Timid", "evs": "4 HP / 252 SpA / 252 Spe", "usage": 62.5},
                        {"nature": "Modest", "evs": "4 HP / 252 SpA / 252 Spe", "usage": 25.3}
                    ]
                },
                "koraidon": {
                    "items": [
                        {"name": "Choice Scarf", "usage": 35.2},
                        {"name": "Life Orb", "usage": 28.5},
                        {"name": "Clear Amulet", "usage": 18.2}
                    ],
                    "abilities": [
                        {"name": "Orichalcum Pulse", "usage": 100.0}
                    ],
                    "moves": [
                        {"name": "Collision Course", "usage": 95.5},
                        {"name": "Flare Blitz", "usage": 88.2},
                        {"name": "Protect", "usage": 68.5},
                        {"name": "Dragon Claw", "usage": 45.2},
                        {"name": "U-turn", "usage": 35.1}
                    ],
                    "tera_types": [
                        {"type": "Fire", "usage": 38.5},
                        {"type": "Flying", "usage": 25.2},
                        {"type": "Steel", "usage": 18.1}
                    ],
                    "spreads": [
                        {"nature": "Jolly", "evs": "4 HP / 252 Atk / 252 Spe", "usage": 55.2},
                        {"nature": "Adamant", "evs": "4 HP / 252 Atk / 252 Spe", "usage": 32.1}
                    ]
                }
            }
            
            if pokemon in usage_stats_db:
                data = usage_stats_db[pokemon]
                return {
                    "pokemon": pokemon,
                    "format": self._format,
                    "items": data.get("items", [])[:5],
                    "abilities": data.get("abilities", []),
                    "moves": data.get("moves", [])[:6],
                    "tera_types": data.get("tera_types", [])[:4],
                    "spreads": data.get("spreads", [])[:3]
                }
            else:
                # 데이터가 없는 포켓몬은 기본 메시지 반환
                return {
                    "pokemon": pokemon,
                    "format": self._format,
                    "message": "No usage data available for this Pokemon in the specified format.",
                    "note": "Consider checking pokedex for base stats instead."
                }
        
        # =====================================================================
        # END FUNCTION CALLING - Tool Execution Handlers
        # =====================================================================
        
        return {"error": f"Unknown tool: {tool_name}"}
    
    def _format_team_selection(
        self, 
        result: Dict, 
        my_team: List[Dict],
        state: Optional[BattleState] = None
    ) -> str:
        """GPT 분석 결과를 showdown 포맷으로 변환하고 BattleState 업데이트."""
        selection = result.get("selection", {})
        
        # 새 포맷 (bring, lead) 또는 구 포맷 (bring_4, lead_2) 지원
        bring = selection.get("bring") or selection.get("bring_4", [1, 2, 3, 4])
        lead = selection.get("lead") or selection.get("lead_2", bring[:2] if bring else [1, 2])
        
        # 단일 값이면 리스트로 변환
        if not isinstance(bring, list):
            bring = [bring]
        if not isinstance(lead, list):
            lead = [lead]
        
        # 포켓몬 이름 → 슬롯 번호 매핑 생성
        name_to_slot = {}
        for i, mon in enumerate(my_team):
            species = mon.get('species', '').lower().replace(' ', '').replace('-', '')
            name_to_slot[species] = i + 1
            # 변형된 이름도 매핑 (ursalunabloodmoon 등)
            name_to_slot[species.replace('_', '')] = i + 1
        
        def to_slot(val):
            """값을 슬롯 번호로 변환"""
            if isinstance(val, int):
                return val
            if isinstance(val, float):
                return int(val)
            if isinstance(val, str):
                # 숫자 문자열이면 정수로 변환
                if val.isdigit():
                    return int(val)
                # 포켓몬 이름이면 슬롯 번호로 변환
                normalized = val.lower().replace(' ', '').replace('-', '').replace('_', '')
                if normalized in name_to_slot:
                    return name_to_slot[normalized]
                # 부분 매칭 시도
                for name, slot in name_to_slot.items():
                    if normalized in name or name in normalized:
                        return slot
            return None
        
        # bring과 lead를 슬롯 번호로 변환
        bring = [s for s in (to_slot(b) for b in bring) if s is not None]
        lead = [s for s in (to_slot(l) for l in lead) if s is not None]
        
        # 중복 제거
        bring = list(dict.fromkeys(bring))
        lead = list(dict.fromkeys(lead))
        
        # VGC 규칙: 4마리만 선택 (bring), 2마리 선발 (lead)
        bring = bring[:4]
        lead = lead[:2]
        
        # 유효하지 않은 경우 기본값
        if not bring or len(bring) < 4:
            # bring이 부족하면 나머지 슬롯으로 채움
            all_slots = [1, 2, 3, 4, 5, 6]
            for s in all_slots:
                if s not in bring and len(bring) < 4:
                    bring.append(s)
        if not lead:
            lead = bring[:2]
        
        # 분석 결과 추출
        opponent_analysis = result.get("opponent_analysis", {})
        my_strategy = result.get("my_strategy", {})
        reasoning = result.get("reasoning", "")  # 구 포맷 호환
        
        # =====================================================================
        # BattleState 업데이트 - 선택 정보 저장
        # =====================================================================
        if state:
            # 선택된 포켓몬 리스트 (species names)
            selected_species = []
            lead_species = []
            for slot in bring:
                if isinstance(slot, int) and 1 <= slot <= len(my_team):
                    species = my_team[slot - 1]['species']
                    selected_species.append(species)
                    if slot in lead:
                        lead_species.append(species)
            
            # BattleState에 선택 정보 반영
            self._update_battle_state_selections(state, selected_species, lead_species)
        
        print("\n" + "=" * 70)
        print("GPT TEAM ANALYSIS")
        print("=" * 70)
        
        # 상대 분석
        if opponent_analysis and isinstance(opponent_analysis, dict):
            print("【 OPPONENT ANALYSIS 】")
            print(f"  Archetype: {opponent_analysis.get('archetype', 'Unknown')}")
            key_threats = opponent_analysis.get('key_threats', [])
            predicted_leads = opponent_analysis.get('predicted_leads') or opponent_analysis.get('likely_leads', [])
            expected_strategy = opponent_analysis.get('expected_strategy', '')
            
            if isinstance(key_threats, list) and key_threats:
                print(f"  Key Threats: {', '.join(str(t) for t in key_threats)}")
            if isinstance(predicted_leads, list) and predicted_leads:
                print(f"  Predicted Leads: {', '.join(str(l) for l in predicted_leads)}")
            if expected_strategy:
                print(f"  Expected Strategy: {expected_strategy}")
        
        print("-" * 70)
        
        # 내 전략
        if my_strategy and isinstance(my_strategy, dict):
            print("【 MY STRATEGY 】")
            game_plan = my_strategy.get('game_plan', '')
            lead_reasoning = my_strategy.get('lead_reasoning', '')
            backup_plan = my_strategy.get('backup_plan', '')
            
            if game_plan:
                print(f"  Game Plan: {game_plan}")
            if lead_reasoning:
                print(f"  Lead Reasoning: {lead_reasoning}")
            if backup_plan:
                print(f"  Backup Plan: {backup_plan}")
        elif reasoning:
            # 구 포맷 호환
            print("【 STRATEGY 】")
            if isinstance(reasoning, dict):
                print(f"  {reasoning.get('game_plan', reasoning.get('why_these_leads', 'N/A'))}")
            else:
                print(f"  {reasoning}")
        
        print("-" * 70)
        print("【 SELECTION 】")
        
        # 선택된 포켓몬 출력
        bring_names = []
        for slot in bring:
            if isinstance(slot, int) and 1 <= slot <= len(my_team):
                mon = my_team[slot - 1]
                bring_names.append(mon['species'])
                lead_marker = " [LEAD]" if slot in lead else ""
                print(f"  Slot {slot}: {mon['species']}{lead_marker}")
        
        print("=" * 70)
        
        # Showdown 포맷: /team ABCD (선발이 앞에 오도록)
        # lead를 먼저, 나머지를 뒤에
        back = [s for s in bring if s not in lead]
        ordered = list(lead) + back
        
        team_str = "".join([str(s) for s in ordered])
        team_cmd = f"/team {team_str}"
        
        # Log the team selection order
        lead_names = [my_team[s-1]['species'] for s in lead]
        back_names = [my_team[s-1]['species'] for s in back]
        self.logger.info(f"[TEAM ORDER] Lead: {', '.join(lead_names)} | Bench: {', '.join(back_names)}")
        
        return team_cmd

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def _battle_finished_callback(self, battle: AbstractBattle):
        """Clean up after battle and save turn logs."""
        if battle.battle_tag in self._battle_response_ids:
            del self._battle_response_ids[battle.battle_tag]
        
        for key in list(self._strategy_progress.keys()):
            if battle.battle_tag in key:
                del self._strategy_progress[key]
        
        # Clear opponent info for this battle
        self._opponent_info.clear()
        self._my_pokemon_hp.clear()
        
        # =========================================================================
        # Save turn logs to file
        # =========================================================================
        self._save_battle_logs(battle)
        
        # Clean up BattleState
        self._cleanup_battle_state(battle.battle_tag)
        
        self.logger.info(f"Battle finished: {battle.battle_tag}")
        self.logger.info(f"Result: {'Won' if battle.won else 'Lost' if battle.lost else 'Tie'}")
        self.logger.info(f"Tokens: input={self._tokens['input']}, output={self._tokens['output']}, reasoning={self._tokens['reasoning']}")
    
    def _save_battle_logs(self, battle: AbstractBattle) -> None:
        """
        Save all turn logs to a file for post-battle analysis.
        
        File naming convention:
        {format}_{battle_number}_{player1}_vs_{player2}_{result}_{timestamp}.json
        
        Example: gen9vgc2025regh_159_PochampsPlayer1_vs_RandomPlayer_win_20251205_143052.json
        """
        if battle.battle_tag not in self._turn_logs:
            return
        
        logs = self._turn_logs[battle.battle_tag]
        if not logs:
            return
        
        # Create log directory
        log_dir = self.log_dir or "battle_log/analysis"
        os.makedirs(log_dir, exist_ok=True)
        
        # Parse battle info from battle_tag
        # Format: "battle-gen9vgc2025regh-159" -> format="gen9vgc2025regh", number="159"
        battle_tag = battle.battle_tag
        tag_parts = battle_tag.split("-")
        battle_format = tag_parts[1] if len(tag_parts) > 1 else "unknown"
        battle_number = tag_parts[2] if len(tag_parts) > 2 else "0"
        
        # Get player names
        my_name = self.username if hasattr(self, 'username') and self.username else "PochampsPlayer"
        # Opponent name from battle object if available
        opp_name = "Opponent"
        if hasattr(battle, 'opponent_username') and battle.opponent_username:
            opp_name = battle.opponent_username
        elif hasattr(battle, '_opponent_username') and battle._opponent_username:
            opp_name = battle._opponent_username
        
        # Sanitize names for filename
        my_name_safe = my_name.replace(" ", "").replace("/", "_")[:20]
        opp_name_safe = opp_name.replace(" ", "").replace("/", "_")[:20]
        
        # Result
        result = "win" if battle.won else "loss" if battle.lost else "tie"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create filename: format_number_player1_vs_player2_result_timestamp.json
        filename = f"{log_dir}/{battle_format}_{battle_number}_{my_name_safe}_vs_{opp_name_safe}_{result}_{timestamp}.json"
        
        # Build comprehensive log data
        log_data = {
            "meta": {
                "battle_tag": battle_tag,
                "format": battle_format,
                "battle_number": battle_number,
                "player1": my_name,
                "player2": opp_name,
                "result": result,
                "total_turns": len(logs),
                "timestamp": timestamp,
                "datetime": datetime.now().isoformat()
            },
            "summary": {
                "winner": my_name if battle.won else opp_name if battle.lost else "tie",
                "my_pokemon_remaining": sum(1 for p in battle.team.values() if not p.fainted),
                "opp_pokemon_remaining": sum(1 for p in battle.opponent_team.values() if not p.fainted),
                "strategies_used": self._summarize_strategies_used(logs)
            },
            "turns": [log.to_dict() for log in logs]
        }
        
        # Save to file (no console output)
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Battle analysis saved: {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save battle logs: {e}")
        
        # Clean up logs from memory
        del self._turn_logs[battle.battle_tag]
    
    def _summarize_strategies_used(self, logs: List[BattleTurnLog]) -> Dict[str, int]:
        """Summarize which strategies were selected across all turns."""
        strategy_counts = {}
        for log in logs:
            strategy = log.selected_strategy or "unknown"
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        return strategy_counts

    def __del__(self):
        """Cleanup."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
