"""
PochampsPlayer - Custom Pokemon Battle AI using OpenAI API.
Implements parallel strategy evaluation with function calling.
"""

import json
import os
import sys
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
    original_tera: Optional[str] = None  # Original tera type (if known)
    
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
        self.global_timeout = 25.0

        # OpenAI client & thread pool
        # For local OSS models, set base_url (e.g., gpt-oss Responses API server)
        if self.api_key:
            if self.base_url:
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
                print(f"[CONFIG] Using custom API endpoint: {self.base_url}")
            else:
                self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None
        
        # Log model configuration
        print(f"[CONFIG] Models - Fast: {self.fast_model}, Normal: {self.backend}, Deep: {self.deep_model}")
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
                print(f"[DEBUG] Loaded {len(self._response_cache)} cached responses from {self._cache_file}")
            except FileNotFoundError:
                self._response_cache = {}
                print(f"[DEBUG] No cache file found, starting fresh")
            except json.JSONDecodeError:
                self._response_cache = {}
                print(f"[DEBUG] Cache file corrupted, starting fresh")
    
    def _save_response_cache(self):
        """Save GPT responses to cache file."""
        if self._cache_file and self._response_cache:
            try:
                with open(self._cache_file, "w") as f:
                    json.dump(self._response_cache, f, indent=2)
                print(f"[DEBUG] Saved {len(self._response_cache)} responses to cache")
            except Exception as e:
                print(f"[DEBUG] Failed to save cache: {e}")
    
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
            print(f"[DEBUG] Using cached response for {cache_key}")
            return self._response_cache[cache_key]
        return None
    
    def _cache_response(self, cache_key: str, response: Dict):
        """Cache a GPT response."""
        if self.debug_mode:
            self._response_cache[cache_key] = response
            self._save_response_cache()
            print(f"[DEBUG] Cached response for {cache_key}")

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
            print(f"[STATE] Created new BattleState for {battle_tag}")
        
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
        
        print(f"[STATE] Initialized teams - My: {my_team_pokemon}, Opp: {opponent_pokemon}")
    
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
        
        print(f"[STATE] Selections - Bring: {selected_pokemon}, Lead: {lead_pokemon}")
    
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
            print(f"[STATE] Battle ended: {battle_tag}")
    
    def _normalize_name(self, name: str) -> str:
        """Normalize pokemon/move names for comparison."""
        if not name:
            return ""
        return name.lower().replace(" ", "").replace("-", "").replace("'", "")

    # =========================================================================
    # Parallel GPT Strategies
    # =========================================================================

    def call_parallel_strategies(
        self,
        battle_tag: str,
        battle_context: Dict,
        tools: Optional[List[Dict]] = None,
        timeout: float = None
    ) -> Dict[str, str]:
        """
        Call 3 strategies in parallel: fast, normal, deep.
        Each strategy builds its own prompt from battle_context.
        Returns responses that complete within timeout.
        """
        print(f"[INFO] Calling parallel strategies for battle {battle_tag}/{len(battle_context.get('turns', []))} turns")
        if not self.client:
            self.logger.error("OpenAI client not initialized.")
            return {}
        
        timeout = timeout or self.global_timeout
        
        # Submit all strategies
        futures = {
            self.executor.submit(
                call_fast_strategy,
                client=self.client,
                battle_tag=battle_tag,
                battle_context=battle_context,
                model=self.fast_model,  # Configurable (default: backend)
                temperature=0.3,
                max_tokens=150,
                battle_response_ids=self._battle_response_ids,
                token_tracker=self._tokens,
                tools=tools
            ): "fast",
            self.executor.submit(
                call_normal_strategy,
                client=self.client,
                battle_tag=battle_tag,
                battle_context=battle_context,
                model=self.backend,
                temperature=self.temperature,
                max_tokens=300,
                battle_response_ids=self._battle_response_ids,
                token_tracker=self._tokens,
                tool_executor=self.tool_executor,
                tools=tools
            ): "normal",
            self.executor.submit(
                call_deep_strategy,
                client=self.client,
                battle_tag=battle_tag,
                battle_context=battle_context,
                model=self.deep_model,  # Configurable (default: backend)
                temperature=self.temperature,
                max_tokens=500,
                battle_response_ids=self._battle_response_ids,
                token_tracker=self._tokens,
                tool_executor=self.tool_executor,
                progress_tracker=self._strategy_progress,
                tools=tools,
                reasoning_effort="high"
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

    def select_best_response(self, responses: Dict[str, str]) -> Optional[str]:
        """Select best response with priority: deep > normal > fast."""
        for strategy in ["deep", "normal", "fast"]:
            if strategy in responses and responses[strategy]:
                try:
                    parsed = json.loads(responses[strategy])
                    self.logger.info(f"Using '{strategy}' response")
                    
                    # Log the actual decision for debugging
                    print(f"\n[DECISION] Strategy '{strategy}' response:")
                    print(f"  {json.dumps(parsed, indent=2)}")
                    
                    return responses[strategy]
                except json.JSONDecodeError:
                    continue
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
        return ForfeitBattleOrder()
        if isinstance(battle, DoubleBattle):
            return self._choose_doubles_move(battle, state)
        # Singles is rarely used in VGC - fallback to random
        return self.choose_random_move(battle)

    def _choose_doubles_move(self, battle: DoubleBattle, state: Optional[BattleState] = None) -> BattleOrder:
        """
        Doubles battle logic - main VGC format.
        
        Flow:
        1. Update confirmed info from last message
        2. Build battle context for GPT (including BattleState)
        3. Call GPT with parallel strategies (fast/normal/deep)
        4. Log turn details for visibility
        """
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
        
        # Capture GPT context summary for logging
        turn_log.gpt_context_summary = self._summarize_gpt_context(battle_context)
        
        # =====================================================================
        # Step 3: GPT inference with 3 parallel strategies
        # =====================================================================
        responses = self.call_parallel_strategies(
            battle_tag=battle.battle_tag,
            battle_context=battle_context,
            tools=BATTLE_TOOLS,
            timeout=self.global_timeout
        )
        
        # Capture all strategy responses
        turn_log.gpt_responses = self._capture_strategy_responses(responses)
        
        best = self.select_best_response(responses)
        if best:
            # Capture selected strategy
            turn_log.selected_strategy = best.get("strategy", "unknown")
            
            orders = self._parse_doubles_decision(best, battle, turn_log)
            if orders:
                # Finalize and store turn log
                self._finalize_turn_log(battle.battle_tag, turn_log, orders)
                return orders
        
        # Fallback to random
        turn_log.final_decision = "FALLBACK: Random move (GPT failed)"
        turn_log.selected_strategy = "random_fallback"
        random_order = self.choose_random_doubles_move(battle)
        self._finalize_turn_log(battle.battle_tag, turn_log, random_order)
        return random_order
    
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
    
    def _summarize_gpt_context(self, context: str) -> str:
        """Create a readable summary of what GPT received."""
        # Extract key info from context for log readability
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
    
    def _capture_strategy_responses(self, responses: Dict[str, Dict]) -> Dict[str, Any]:
        """Capture all strategy responses for logging."""
        captured = {}
        
        for strategy_name, resp in responses.items():
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
            response = self.client.responses.create(
                model=self.fast_model,  # Use fast model for analysis (lightweight task)
                input=[
                    {"role": "system", "content": self._get_analysis_system_prompt()},
                    {"role": "user", "content": json.dumps(analysis_context, indent=2)}
                ],
                temperature=0.3,
                max_output_tokens=400,
                text={"format": {"type": "json_object"}}
            )
            
            # Track tokens
            if hasattr(response, 'usage') and response.usage:
                self._tokens['input'] += getattr(response.usage, 'input_tokens', 0)
                self._tokens['output'] += getattr(response.usage, 'output_tokens', 0)
            
            # Extract and apply inferences
            text = None
            if response.output:
                for item in response.output:
                    if item.type == "message":
                        for content in item.content:
                            if content.type == "output_text":
                                text = content.text
                                break
            
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
        
        context = {
            "turn": battle.turn,
            "my_active": [],
            "opponent_active": [],
            "my_team": [],
            "opponent_revealed": [],
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
                    "can_terastallize": battle.can_tera[i] if i < len(battle.can_tera) else False
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
                "active": p.active
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
                "item": {"value": p.item, "probability": 100.0} if p.item else None
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
        
        # Available switches
        available_switches = []
        try:
            if hasattr(battle, 'available_switches') and battle.available_switches:
                available_switches = [p.species for p in battle.available_switches]
        except Exception as e:
            self.logger.debug(f"Error getting available switches: {e}")
        context["available_switches"] = available_switches
        
        return context

    def _parse_doubles_decision(
        self, 
        response: str, 
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
        """
        validation_log = []  # Track validation actions
        
        try:
            decision = json.loads(response)
            orders: List[Optional[BattleOrder]] = [None, None]
            
            # Log raw decision from GPT
            if turn_log:
                turn_log.decision_validation = {
                    "raw_decision": decision,
                    "validation_actions": validation_log
                }
            
            for i, slot_key in enumerate(["slot1", "slot2"]):
                if slot_key not in decision:
                    continue
                
                slot_data = decision[slot_key]
                action_type = slot_data.get("action", "").lower()
                
                if action_type == "move":
                    move_id = slot_data.get("move", "").lower()
                    target = slot_data.get("target", 0)
                    tera = slot_data.get("terastallize", False)
                    
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
                    
                    for pokemon in battle.available_switches:
                        if pokemon_name in pokemon.species.lower() or pokemon.species.lower() in pokemon_name:
                            validation_log.append({
                                "slot": i + 1,
                                "action": "switch",
                                "pokemon": pokemon.species
                            })
                            orders[i] = BattleOrder(pokemon)
                            break
            
            # Update turn_log with validation
            if turn_log:
                turn_log.decision_validation["validation_actions"] = validation_log
            
            # Build DoubleBattleOrder
            if orders[0] and orders[1]:
                return DoubleBattleOrder(first_order=orders[0], second_order=orders[1])
            elif orders[0]:
                return DoubleBattleOrder(first_order=orders[0])
            elif orders[1]:
                return DoubleBattleOrder(second_order=orders[1])
                
        except Exception as e:
            self.logger.error(f"Parse doubles error: {e}")
            if turn_log:
                turn_log.decision_validation = {"error": str(e)}
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
        # Rule 1: Damaging moves should NEVER target ally (-1) in normal situations
        # =========================================================================
        if is_damaging and requested_target == -1:
            print(f"    ⚠️ WARNING: Damaging move '{move.id}' targeting ally! Fixing...")
            
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
            print(f"    ⚠️ WARNING: Target {requested_target} not in valid targets {valid_targets}")
            
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
                    print(f"    ⚠️ WARNING: Target slot {requested_target} is empty/fainted!")
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
        print(f"\n[DEBUG] self.client = {self.client}", flush=True)
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
        
        # Fallback: 기본 선발 (앞 4마리, 1-2번 선발)
        return "/team 1234"
    
    def _parse_my_team(self) -> List[Dict]:
        """내 팀 정보를 팀빌더에서 파싱."""
        team_info = []
        if not self._team:
            return team_info
        
        packed = self._team.yield_team()
        for i, mon_str in enumerate(packed.split("]"), 1):
            fields = mon_str.split("|")
            # 0:nickname, 1:species, 2:item, 3:ability, 4:moves, 5:nature, 6:evs
            species = fields[1] if len(fields) > 1 else ""
            item = fields[2] if len(fields) > 2 else ""
            ability = fields[3] if len(fields) > 3 else ""
            moves = fields[4].split(",") if len(fields) > 4 and fields[4] else []
            nature = fields[5] if len(fields) > 5 else ""
            evs_str = fields[6] if len(fields) > 6 else ""
            
            # Tera type 파싱
            tera_type = ""
            if len(fields) > 12:
                for f in fields[12:]:
                    if f and f not in ["G", "S", ""]:
                        tera_type = f
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
                "description": "Get VGC usage statistics for a Pokemon: common items, abilities, moves, EV spreads, teammates, and Tera types with usage percentages. Use for Stage 2 statistical analysis.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "generation": {"type": "integer", "description": "Pokemon generation (e.g., 9 for Gen 9)"},
                        "regulation": {"type": "string", "description": "Battle regulation (e.g., 'H' for Regulation Set H, just alphabet)"},
                        "gametype": {"type": "string", "description": "Game type single or double"},
                        "pokemon": {"type": "string", "description": "Pokemon name"}
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
            response = self.client.responses.create(
                model=self.backend,  # gpt-4o
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                tools=analysis_tools,
                temperature=0.5,
                max_output_tokens=2000
            )
            
            # Token tracking
            if hasattr(response, 'usage') and response.usage:
                self._tokens['input'] += getattr(response.usage, 'input_tokens', 0)
                self._tokens['output'] += getattr(response.usage, 'output_tokens', 0)
            
            # Function calls 처리 (with state for storing predictions)
            result = self._process_team_analysis_response(response, analysis_tools, my_team, opponent_team, state)
            
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
        
        return f"""You are an expert Pokemon battle analyst.

# BATTLE FORMAT
- Team size: {team_size} → Bring: {max_team_size}
- Type: {"Doubles" if is_doubles else "Singles"} (Lead: {lead_count})

# TOOL USAGE
You have access to tools for additional information. Use them if needed:
- If you're confident about opponent's common sets/strategy → skip tools
- If you need specific usage stats, type matchups, or speed tiers → call tools
- When calling tools, batch ALL calls together for efficiency

# OUTPUT FORMAT (JSON)

```json
{{
    "opponent_analysis": {{
        "archetype": "<team style>",
        "key_threats": ["pokemon1", "pokemon2"],
        "predicted_leads": ["pokemon1", "pokemon2"],
        "expected_strategy": "<1-2 sentences: what opponent likely wants to do>"
    }},
    "my_strategy": {{
        "game_plan": "<2-3 sentences: overall win condition and approach>",
        "lead_reasoning": "<why these leads counter opponent's likely play>",
        "backup_plan": "<what to do if prediction is wrong>"
    }},
    "selection": {{
        "bring": [slot1, slot2, ...],
        "lead": [slot1, ...]
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
            prompt += "\n"
        
        prompt += "---\n\n"
        prompt += "Analyze the matchup and make your team selection.\n"
        prompt += "Use tools only if you need additional information to make a confident decision."
        
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
            pokemon = args.get("pokemon", "").lower().replace(" ", "")
            # TODO: generation별 pokedex 분기 처리
            # 현재는 하드코딩된 샘플 데이터 반환
            pokemon_db = {
                "incineroar": {
                    "species": "Incineroar",
                    "types": ["Fire", "Dark"],
                    "baseStats": {"hp": 95, "atk": 115, "def": 90, "spa": 80, "spd": 90, "spe": 60},
                    "abilities": {"0": "Blaze", "H": "Intimidate"}
                },
                "rillaboom": {
                    "species": "Rillaboom",
                    "types": ["Grass"],
                    "baseStats": {"hp": 100, "atk": 125, "def": 90, "spa": 60, "spd": 70, "spe": 85},
                    "abilities": {"0": "Overgrow", "H": "Grassy Surge"}
                },
                "urshifu": {
                    "species": "Urshifu",
                    "types": ["Fighting", "Dark"],
                    "baseStats": {"hp": 100, "atk": 130, "def": 100, "spa": 63, "spd": 60, "spe": 97},
                    "abilities": {"0": "Unseen Fist"}
                },
                "urshifurapidstrike": {
                    "species": "Urshifu-Rapid-Strike",
                    "types": ["Fighting", "Water"],
                    "baseStats": {"hp": 100, "atk": 130, "def": 100, "spa": 63, "spd": 60, "spe": 97},
                    "abilities": {"0": "Unseen Fist"}
                },
                "miraidon": {
                    "species": "Miraidon",
                    "types": ["Electric", "Dragon"],
                    "baseStats": {"hp": 100, "atk": 85, "def": 100, "spa": 135, "spd": 115, "spe": 135},
                    "abilities": {"0": "Hadron Engine"}
                },
                "koraidon": {
                    "species": "Koraidon",
                    "types": ["Fighting", "Dragon"],
                    "baseStats": {"hp": 100, "atk": 135, "def": 115, "spa": 85, "spd": 100, "spe": 135},
                    "abilities": {"0": "Orichalcum Pulse"}
                },
                "fluttermane": {
                    "species": "Flutter Mane",
                    "types": ["Ghost", "Fairy"],
                    "baseStats": {"hp": 55, "atk": 55, "def": 55, "spa": 135, "spd": 135, "spe": 135},
                    "abilities": {"0": "Protosynthesis"}
                },
                "chienpao": {
                    "species": "Chien-Pao",
                    "types": ["Dark", "Ice"],
                    "baseStats": {"hp": 80, "atk": 120, "def": 80, "spa": 90, "spd": 65, "spe": 135},
                    "abilities": {"0": "Sword of Ruin"}
                },
                "landorus": {
                    "species": "Landorus",
                    "types": ["Ground", "Flying"],
                    "baseStats": {"hp": 89, "atk": 125, "def": 90, "spa": 115, "spd": 80, "spe": 101},
                    "abilities": {"0": "Sand Force", "H": "Sheer Force"}
                },
                "amoonguss": {
                    "species": "Amoonguss",
                    "types": ["Grass", "Poison"],
                    "baseStats": {"hp": 114, "atk": 85, "def": 70, "spa": 85, "spd": 80, "spe": 30},
                    "abilities": {"0": "Effect Spore", "H": "Regenerator"}
                }
            }
            
            if pokemon in pokemon_db:
                return pokemon_db[pokemon]
            else:
                return {
                    "species": pokemon,
                    "types": ["Unknown"],
                    "baseStats": {"hp": 80, "atk": 80, "def": 80, "spa": 80, "spd": 80, "spe": 80},
                    "abilities": {},
                    "note": "No detailed data available for this Pokemon"
                }
        
        elif tool_name == "get_move_info":
            generation = args.get("generation", 9)
            move = args.get("move", "").lower().replace(" ", "")
            # TODO: generation별 moves 분기 처리
            # 현재는 하드코딩된 샘플 데이터 반환
            move_db = {
                "fakeout": {
                    "name": "Fake Out",
                    "type": "Normal",
                    "category": "Physical",
                    "basePower": 40,
                    "accuracy": 100,
                    "priority": 3,
                    "target": "normal",
                    "desc": "Usually goes first. The target flinches. First turn out only."
                },
                "protect": {
                    "name": "Protect",
                    "type": "Normal",
                    "category": "Status",
                    "basePower": 0,
                    "accuracy": 100,
                    "priority": 4,
                    "target": "self",
                    "desc": "Prevents all moves from affecting the user this turn."
                },
                "closecombat": {
                    "name": "Close Combat",
                    "type": "Fighting",
                    "category": "Physical",
                    "basePower": 120,
                    "accuracy": 100,
                    "priority": 0,
                    "target": "normal",
                    "desc": "Lowers the user's Defense and Sp. Def by 1."
                },
                "surgingstrikes": {
                    "name": "Surging Strikes",
                    "type": "Water",
                    "category": "Physical",
                    "basePower": 25,
                    "accuracy": 100,
                    "priority": 0,
                    "target": "normal",
                    "desc": "Hits 3 times. Always results in a critical hit."
                },
                "grassyglide": {
                    "name": "Grassy Glide",
                    "type": "Grass",
                    "category": "Physical",
                    "basePower": 55,
                    "accuracy": 100,
                    "priority": 0,
                    "target": "normal",
                    "desc": "User on Grassy Terrain: +1 priority."
                },
                "flareblitz": {
                    "name": "Flare Blitz",
                    "type": "Fire",
                    "category": "Physical",
                    "basePower": 120,
                    "accuracy": 100,
                    "priority": 0,
                    "target": "normal",
                    "desc": "Has 33% recoil. 10% chance to burn."
                },
                "knockoff": {
                    "name": "Knock Off",
                    "type": "Dark",
                    "category": "Physical",
                    "basePower": 65,
                    "accuracy": 100,
                    "priority": 0,
                    "target": "normal",
                    "desc": "1.5x damage if foe holds an item. Removes item."
                },
                "partingshot": {
                    "name": "Parting Shot",
                    "type": "Dark",
                    "category": "Status",
                    "basePower": 0,
                    "accuracy": 100,
                    "priority": 0,
                    "target": "normal",
                    "desc": "Lowers target's Atk, Sp. Atk by 1. User switches."
                },
                "electrodrift": {
                    "name": "Electro Drift",
                    "type": "Electric",
                    "category": "Special",
                    "basePower": 100,
                    "accuracy": 100,
                    "priority": 0,
                    "target": "normal",
                    "desc": "Super effective hits deal 1.33x damage."
                },
                "dracometeo": {
                    "name": "Draco Meteor",
                    "type": "Dragon",
                    "category": "Special",
                    "basePower": 130,
                    "accuracy": 90,
                    "priority": 0,
                    "target": "normal",
                    "desc": "Lowers the user's Sp. Atk by 2."
                }
            }
            
            if move in move_db:
                return move_db[move]
            else:
                return {
                    "name": move,
                    "type": "Unknown",
                    "category": "Unknown",
                    "basePower": 0,
                    "accuracy": 100,
                    "priority": 0,
                    "target": "normal",
                    "desc": "No detailed data available for this move"
                }
        
        elif tool_name == "get_item_info":
            generation = args.get("generation", 9)
            item = args.get("item", "").lower().replace(" ", "")
            # TODO: generation별 item 데이터 분기 처리
            # 현재는 하드코딩된 샘플 데이터 반환
            item_db = {
                "choicescarf": {
                    "name": "Choice Scarf",
                    "desc": "Holder's Speed is 1.5x, but it can only select the first move it executes.",
                    "category": "speed_boost",
                    "common_users": ["Urshifu", "Landorus", "Flutter Mane"]
                },
                "focussash": {
                    "name": "Focus Sash",
                    "desc": "If holder's HP is full, survives any single hit with 1 HP. Single use.",
                    "category": "survival",
                    "common_users": ["Whimsicott", "Chi-Yu", "Froslass"]
                },
                "lifeorb": {
                    "name": "Life Orb",
                    "desc": "Holder's attacks do 1.3x damage, loses 1/10 max HP after attacking.",
                    "category": "damage_boost",
                    "common_users": ["Miraidon", "Koraidon", "Flutter Mane"]
                },
                "assaultvest": {
                    "name": "Assault Vest",
                    "desc": "Holder's Sp. Def is 1.5x, but can only use attacking moves.",
                    "category": "bulk",
                    "common_users": ["Rillaboom", "Incineroar", "Kingambit"]
                },
                "safetygoggles": {
                    "name": "Safety Goggles",
                    "desc": "Protects from weather damage and powder moves.",
                    "category": "utility",
                    "common_users": ["Incineroar", "Amoonguss", "Tornadus"]
                },
                "sitrusberry": {
                    "name": "Sitrus Berry",
                    "desc": "Restores 25% max HP when at 50% or less.",
                    "category": "recovery",
                    "common_users": ["Incineroar", "Amoonguss", "Dondozo"]
                },
                "clearamulet": {
                    "name": "Clear Amulet",
                    "desc": "Prevents other Pokemon from lowering the holder's stat stages.",
                    "category": "utility",
                    "common_users": ["Koraidon", "Arcanine", "Kingambit"]
                },
                "covertcloak": {
                    "name": "Covert Cloak",
                    "desc": "Protects holder from additional effects of moves.",
                    "category": "utility",
                    "common_users": ["Kingambit", "Tornadus", "Iron Hands"]
                },
                "choiceband": {
                    "name": "Choice Band",
                    "desc": "Holder's Attack is 1.5x, but it can only select the first move it executes.",
                    "category": "damage_boost",
                    "common_users": ["Rillaboom", "Chien-Pao", "Dragonite"]
                },
                "choicespecs": {
                    "name": "Choice Specs",
                    "desc": "Holder's Sp. Atk is 1.5x, but it can only select the first move it executes.",
                    "category": "damage_boost",
                    "common_users": ["Miraidon", "Flutter Mane", "Chi-Yu"]
                }
            }
            
            if item in item_db:
                return item_db[item]
            else:
                return {
                    "name": item,
                    "desc": "No detailed information available for this item.",
                    "category": "unknown",
                    "common_users": []
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
        return f"/team {team_str}"

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
