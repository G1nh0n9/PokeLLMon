"""
Strategy implementations for parallel GPT calls using OpenAI Responses API.
Contains fast, normal, and deep strategy logic.
"""

import json
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI
    from src.player.functions import ToolExecutor


# =============================================================================
# Token Tracking & Response Parsing Helpers
# =============================================================================

def _track_tokens(response, token_tracker: Dict[str, int]):
    """Track token usage from Responses API response."""
    if hasattr(response, 'usage') and response.usage:
        usage = response.usage
        token_tracker['input'] += getattr(usage, 'input_tokens', 0)
        token_tracker['output'] += getattr(usage, 'output_tokens', 0)
        # reasoning_tokens is inside output_tokens_details for reasoning models
        if hasattr(usage, 'output_tokens_details') and usage.output_tokens_details:
            token_tracker['reasoning'] += getattr(usage.output_tokens_details, 'reasoning_tokens', 0)


def _extract_text_content(response) -> Optional[str]:
    """Extract text content from Responses API response."""
    if not response.output:
        return None
    for item in response.output:
        if item.type == "message":
            for content in item.content:
                if content.type == "output_text":
                    return content.text
    return None


def _extract_function_calls(response) -> List[Dict]:
    """Extract function calls from Responses API response."""
    calls = []
    if not response.output:
        return calls
    for item in response.output:
        if item.type == "function_call":
            calls.append({
                "id": item.call_id,
                "name": item.name,
                "arguments": item.arguments
            })
    return calls


# =============================================================================
# Strategy Implementations
# =============================================================================

def call_fast_strategy(
    client: "OpenAI",
    battle_tag: str,
    battle_context: Dict,
    model: str,
    temperature: float,
    max_tokens: int,
    battle_response_ids: Dict[str, str],
    token_tracker: Dict[str, int],
    tools: Optional[List[Dict]] = None
) -> str:
    """
    Fast strategy: Quick response with simplified prompt.
    Focus on immediate tactical advantage without deep analysis.
    
    :param client: OpenAI client
    :param battle_tag: Battle identifier
    :param battle_context: Battle state dictionary
    :param model: Model to use (e.g., gpt-4o-mini)
    :param temperature: Sampling temperature
    :param max_tokens: Max tokens to generate
    :param battle_response_ids: Dict to track response IDs per battle
    :param token_tracker: Dict with 'input', 'output', 'reasoning' keys for tracking
    :param tools: Optional tools (not used in fast strategy)
    :return: JSON response string
    """
    try:
        # Build fast-specific prompt from context
        user_prompt = _build_fast_prompt(battle_context)
        
        fast_system_prompt = """You are a Pokemon VGC doubles AI. Make a quick tactical decision.
Prioritize: type advantage, immediate threats, protect predictions.

CRITICAL TARGET RULES:
=====================================
SPREAD MOVES (hypervoice, dazzlinggleam, heatwave, rockslide, earthquake, etc.):
- These moves hit ALL opponents - DO NOT specify a target!
- OMIT the "target" field or use 0

SINGLE TARGET MOVES:
- target: 1 = attack opponent's LEFT Pokemon
- target: 2 = attack opponent's RIGHT Pokemon

SELF MOVES (Protect, Swords Dance):
- target: 0 or omit target field

NEVER use -1 or -2 for attacking moves (those target allies/self)

SWITCH RULES:
- Both slots CANNOT switch to the same Pokemon
- You can only switch to Pokemon that are not already active

Respond with JSON only:
{
    "slot1": {"action": "move", "move": "<move_id>", "target": <1|2>},
    "slot2": {"action": "move", "move": "<move_id>", "target": <1|2>}
}

For spread moves (NO target):
{
    "slot1": {"action": "move", "move": "hypervoice"},
    "slot2": {"action": "move", "move": "earthquake"}
}

For switches:
{
    "slot1": {"action": "switch", "pokemon": "<species>"},
    "slot2": {"action": "move", "move": "<move_id>", "target": <1|2>}
}"""
        
        request_params = {
            "model": model,
            "input": [
                {"role": "system", "content": fast_system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "text": {"format": {"type": "json_object"}}
        }
        
        # NOTE: Each strategy call is independent - don't use previous_response_id
        # This prevents "No tool output found" errors when previous response had tool calls
        
        response = client.responses.create(**request_params)
        
        if hasattr(response, 'id') and battle_tag not in battle_response_ids:
            battle_response_ids[battle_tag] = response.id
        
        _track_tokens(response, token_tracker)
        
        text = _extract_text_content(response)
        return text if text else "{}"
    
    except Exception as e:
        raise RuntimeError(f"Fast strategy failed: {e}")


def call_normal_strategy(
    client: "OpenAI",
    battle_tag: str,
    battle_context: Dict,
    model: str,
    temperature: float,
    max_tokens: int,
    battle_response_ids: Dict[str, str],
    token_tracker: Dict[str, int],
    tool_executor: "ToolExecutor",
    tools: Optional[List[Dict]] = None
) -> str:
    """
    Normal strategy: Balanced approach with limited tool usage.
    
    :param client: OpenAI client
    :param battle_tag: Battle identifier
    :param battle_context: Battle state dictionary
    :param model: Model to use
    :param temperature: Sampling temperature
    :param max_tokens: Max tokens
    :param battle_response_ids: Response ID tracker
    :param token_tracker: Token usage tracker
    :param tool_executor: Tool executor for function calling
    :param tools: Tool definitions
    :return: JSON response string
    """
    try:
        # Build normal-specific prompts from context
        system_prompt = _build_normal_system_prompt()
        user_prompt = _build_normal_prompt(battle_context)
        
        input_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        request_params = {
            "model": model,
            "input": input_messages,
            "temperature": temperature,
            "max_output_tokens": max_tokens
        }
        
        # NOTE: Each strategy call is independent - don't use previous_response_id
        # This prevents "No tool output found" errors when previous response had tool calls
        
        if tools:
            request_params["tools"] = tools
        else:
            request_params["text"] = {"format": {"type": "json_object"}}
        
        response = client.responses.create(**request_params)
        
        # Always update response ID to latest
        if hasattr(response, 'id'):
            battle_response_ids[battle_tag] = response.id
            current_response_id = response.id
        else:
            current_response_id = None
        
        _track_tokens(response, token_tracker)
        
        # Check for function calls
        func_calls = _extract_function_calls(response)
        if func_calls:
            return _handle_tool_calls_single(
                client=client,
                battle_tag=battle_tag,
                func_calls=func_calls,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                battle_response_ids=battle_response_ids,
                token_tracker=token_tracker,
                tool_executor=tool_executor,
                current_response_id=current_response_id
            )
        
        text = _extract_text_content(response)
        return text if text else "{}"
    
    except Exception as e:
        raise RuntimeError(f"Normal strategy failed: {e}")


def call_deep_strategy(
    client: "OpenAI",
    battle_tag: str,
    battle_context: Dict,
    model: str,
    temperature: float,
    max_tokens: int,
    battle_response_ids: Dict[str, str],
    token_tracker: Dict[str, int],
    tool_executor: "ToolExecutor",
    progress_tracker: Dict[str, Dict],
    tools: Optional[List[Dict]] = None,
    reasoning_effort: str = "high",
    max_iterations: int = 5
) -> str:
    """
    Deep strategy: Comprehensive analysis with iterative refinement.
    Uses extended reasoning and extensive tool usage.
    
    :param client: OpenAI client
    :param battle_tag: Battle identifier
    :param battle_context: Battle state dictionary
    :param model: Model (e.g., gpt-5.1)
    :param temperature: Sampling temperature
    :param max_tokens: Max tokens
    :param battle_response_ids: Response ID tracker
    :param token_tracker: Token tracker
    :param tool_executor: Tool executor
    :param progress_tracker: Progress tracking dict
    :param tools: Tool definitions
    :param reasoning_effort: Reasoning effort level
    :param max_iterations: Max iteration count
    :return: JSON response string
    """
    try:
        progress_key = f"{battle_tag}_deep"
        _init_progress(progress_tracker, progress_key, max_iterations)
        
        # Build deep-specific prompts from context
        deep_system_prompt = _build_deep_system_prompt()
        user_prompt = _build_deep_prompt(battle_context)
        
        conversation = [
            {"role": "system", "content": deep_system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        for iteration in range(max_iterations):
            _update_progress(progress_tracker, progress_key, "iteration", iteration + 1)
            
            request_params = {
                "model": model,
                "input": conversation,
                "max_output_tokens": max_tokens
            }
            
            # gpt-5 and o-series don't support temperature
            if not (model.startswith("o") or "gpt-5" in model):
                request_params["temperature"] = temperature
            
            # Only add reasoning parameter for models that support it (o-series, gpt-5)
            if model.startswith("o") or "gpt-5" in model:
                request_params["reasoning"] = {"effort": reasoning_effort}
            
            # NOTE: Each strategy call is independent - don't use previous_response_id
            # This prevents "No tool output found" errors when previous response had tool calls
            
            if tools:
                request_params["tools"] = tools
            
            response = client.responses.create(**request_params)
            
            # Always update response ID to latest - needed for tool call continuity
            current_response_id = None
            if hasattr(response, 'id'):
                battle_response_ids[battle_tag] = response.id
                current_response_id = response.id
            
            _track_tokens(response, token_tracker)
            if hasattr(response, 'usage') and response.usage:
                progress_tracker[progress_key]["tokens_used"] += getattr(response.usage, 'output_tokens', 0)
            
            # Check for function calls
            func_calls = _extract_function_calls(response)
            if func_calls:
                # Build function outputs and continue with previous_response_id
                function_outputs = []
                for fc in func_calls:
                    func_args = json.loads(fc["arguments"])
                    result = tool_executor.execute(fc["name"], func_args)
                    _add_tool_result(progress_tracker, progress_key, fc["name"], func_args, result)
                    function_outputs.append({
                        "type": "function_call_output",
                        "call_id": fc["id"],
                        "output": json.dumps(result)
                    })
                # Update conversation to be function outputs for next iteration
                conversation = function_outputs
                continue
            
            # Check for text response
            text = _extract_text_content(response)
            if text:
                try:
                    decision = json.loads(text)
                    if "action" in decision and "name" in decision:
                        _update_progress(progress_tracker, progress_key, "status", "completed")
                        _update_progress(progress_tracker, progress_key, "decision", decision)
                        return text
                except json.JSONDecodeError:
                    pass
                
                _add_reasoning(progress_tracker, progress_key, text)
                conversation.append({"role": "assistant", "content": text})
                conversation.append({"role": "user", "content": "Provide final decision in JSON format."})
                continue
            
            break
        
        _update_progress(progress_tracker, progress_key, "status", "max_iterations")
        return "{}"
    
    except Exception as e:
        raise RuntimeError(f"Deep strategy failed: {e}")


# =============================================================================
# Helper Functions
# =============================================================================

def _handle_tool_calls_single(
    client: "OpenAI",
    battle_tag: str,
    func_calls: List[Dict],
    model: str,
    temperature: float,
    max_tokens: int,
    battle_response_ids: Dict[str, str],
    token_tracker: Dict[str, int],
    tool_executor: "ToolExecutor",
    current_response_id: str = None
) -> str:
    """Handle single round of tool calls for normal strategy."""
    # Responses API uses function_call_output items, not role: function
    function_results = []
    
    for fc in func_calls:
        func_args = json.loads(fc["arguments"])
        result = tool_executor.execute(fc["name"], func_args)
        # Responses API format for function output
        function_results.append({
            "type": "function_call_output",
            "call_id": fc["id"],
            "output": json.dumps(result)
        })
    
    request_params = {
        "model": model,
        "input": function_results,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "text": {"format": {"type": "text"}}  # Use text format after tool calls
    }
    
    # Must use the response ID that generated the tool calls
    if current_response_id:
        request_params["previous_response_id"] = current_response_id
    elif battle_tag in battle_response_ids:
        request_params["previous_response_id"] = battle_response_ids[battle_tag]
    
    response = client.responses.create(**request_params)
    
    if hasattr(response, 'id'):
        battle_response_ids[battle_tag] = response.id
    
    _track_tokens(response, token_tracker)
    
    text = _extract_text_content(response)
    return text if text else "{}"


def _init_progress(tracker: Dict, key: str, max_iter: int):
    """Initialize progress tracking."""
    tracker[key] = {
        "status": "running",
        "iteration": 0,
        "max_iterations": max_iter,
        "tokens_used": 0,
        "tool_calls": [],
        "reasoning": [],
        "decision": None
    }


def _update_progress(tracker: Dict, key: str, field: str, value):
    """Update progress field."""
    if key in tracker:
        tracker[key][field] = value


def _add_tool_result(tracker: Dict, key: str, func: str, args: Dict, result: Dict):
    """Add tool result to progress."""
    if key in tracker:
        tracker[key]["tool_calls"].append({"function": func, "args": args, "result": result})


def _add_reasoning(tracker: Dict, key: str, content: str):
    """Add reasoning to progress."""
    if key in tracker:
        tracker[key]["reasoning"].append(content)


# =============================================================================
# Prompt Building Functions
# =============================================================================

def _build_fast_prompt(context: Dict) -> str:
    """Build simplified prompt for fast strategy - minimal info for quick decisions."""
    lines = [f"Turn {context.get('turn', '?')}"]
    
    # CRITICAL: Check force_switch situation
    force_switch = context.get("force_switch", [False, False])
    if force_switch[0] or force_switch[1]:
        lines.append("[FORCE SWITCH]")
        if force_switch[0]:
            lines.append("Slot1: MUST SWITCH (no moves)")
        if force_switch[1]:
            lines.append("Slot2: MUST SWITCH (no moves)")
    
    # My active Pokemon - simplified (species, HP, moves only)
    for poke in context.get("my_active", []):
        slot_idx = poke['slot'] - 1
        # Skip moves display if this slot must switch
        if force_switch[slot_idx]:
            continue
        moves = [m["id"] for m in poke.get("moves", [])]
        tera = " [TERA]" if poke.get("can_terastallize") else ""
        lines.append(f"Slot{poke['slot']}: {poke['species']} ({poke['hp_percent']}%){tera} - {', '.join(moves)}")
    
    # Opponent active - simplified (species, HP only)
    for poke in context.get("opponent_active", []):
        types = "/".join(poke.get("types", [])) if poke.get("types") else "?"
        lines.append(f"Opp{poke['slot']}: {poke['species']} [{types}] ({poke['hp_percent']}%)")
    
    # Available switches - simplified per-slot format
    switches = context.get("available_switches", [])
    if switches:
        for slot_idx, slot_switches in enumerate(switches):
            if isinstance(slot_switches, list) and slot_switches:
                lines.append(f"Sw{slot_idx + 1}: {', '.join(slot_switches)}")
    
    return "\n".join(lines)


def _build_normal_system_prompt() -> str:
    """Build system prompt for normal strategy."""
    return """You are a Pokemon VGC doubles battle AI.
Analyze the battle state and choose optimal actions for both Pokemon.

CRITICAL TARGET RULES:
=====================================
SPREAD MOVES (hit all opponents at once):
- hypervoice, dazzlinggleam, heatwave, rockslide, earthquake, muddy water, etc.
- These moves CANNOT have a target! OMIT the "target" field or use 0!
- Server error "You can't choose a target for X" means you wrongly specified target

SINGLE TARGET MOVES:
- target: 1 = opponent's LEFT slot (position 1)
- target: 2 = opponent's RIGHT slot (position 2)

SELF/ALLY MOVES:
- 0 = self-targeting (Protect, Swords Dance, Calm Mind)
- NEVER target your ally (-1) with damaging moves! That attacks your own Pokemon!

SWITCH RULES (IMPORTANT):
- Both slots CANNOT switch to the same Pokemon (each Pokemon can only be in one slot)
- You can only switch to Pokemon that are NOT already active on the field
- If you want to switch, pick different Pokemon for each slot

Use function calling to:
- calculate_damage: Check if you can KO or deal significant damage
- check_speed_order: Determine who moves first
- get_tera_matchup: Check tera type advantages

RESPONSE FORMAT (JSON):
{
    "slot1": {"action": "move", "move": "<move_id>", "target": <1|2>},
    "slot2": {"action": "move", "move": "<move_id>", "target": <1|2>}
}

For spread moves (NO target needed):
{
    "slot1": {"action": "move", "move": "hypervoice"},
    "slot2": {"action": "move", "move": "heatwave"}
}

For switches:
{
    "slot1": {"action": "switch", "pokemon": "<species>"},
    "slot2": {"action": "move", "move": "<move_id>", "target": <1|2>}
}

Optional: "terastallize": true"""


def _build_normal_prompt(context: Dict) -> str:
    """Build detailed prompt for normal strategy."""
    lines = [f"=== Turn {context.get('turn', '?')} ==="]
    
    # CRITICAL: Check force_switch situation
    force_switch = context.get("force_switch", [False, False])
    if force_switch[0] or force_switch[1]:
        lines.append("")
        lines.append("*** FORCE SWITCH REQUIRED ***")
        if force_switch[0]:
            lines.append("  Slot 1: MUST SWITCH (fainted/forced out, NO moves available)")
        if force_switch[1]:
            lines.append("  Slot 2: MUST SWITCH (fainted/forced out, NO moves available)")
        lines.append("")
    
    # Field conditions
    field = context.get("field", {})
    conditions = []
    if field.get("weather"): conditions.append(f"Weather: {field['weather']}")
    if field.get("terrain"): conditions.append(f"Terrain: {field['terrain']}")
    if field.get("trick_room"): conditions.append("TRICK ROOM")
    if field.get("my_tailwind"): conditions.append("Your Tailwind")
    if field.get("opp_tailwind"): conditions.append("Opp Tailwind")
    if conditions:
        lines.append(" | ".join(conditions))
    
    lines.append("")
    lines.append("YOUR POKEMON:")
    for poke in context.get("my_active", []):
        types = "/".join(poke.get("types", [])) if poke.get("types") else "?"
        lines.append(f"  Slot {poke['slot']}: {poke['species']} [{types}] HP:{poke['hp_percent']}%")
        if poke.get("status"): lines.append(f"    Status: {poke['status']}")
        lines.append(f"    Ability: {poke.get('ability') or '?'} | Item: {poke.get('item') or '?'}")
        if poke.get("terastallized"):
            lines.append(f"    [TERA ACTIVE] -> {poke.get('tera_type', '?')}")
        elif poke.get("can_terastallize"):
            lines.append(f"    [CAN TERA] -> {poke.get('tera_type') or '?'}")
        lines.append("    Moves:")
        for move in poke.get("moves", []):
            targets = move.get("valid_targets", [0])
            lines.append(f"      - {move['id']} ({move.get('type','?')}/{move.get('category','?')}) BP:{move.get('base_power','-')} targets:{targets}")
    
    lines.append("")
    lines.append("OPPONENT POKEMON:")
    for poke in context.get("opponent_active", []):
        types = "/".join(poke.get("types", [])) if poke.get("types") else "?"
        lines.append(f"  Slot {poke['slot']}: {poke['species']} [{types}] HP:{poke['hp_percent']}%")
        if poke.get("status"): lines.append(f"    Status: {poke['status']}")
        if poke.get("terastallized"):
            lines.append(f"    [TERA ACTIVE] -> {poke.get('tera_type', '?')}")
        known_moves = poke.get("known_moves", [])
        if known_moves:
            lines.append(f"    Known moves: {', '.join(known_moves)}")
        # Ability/Item - handle dict format
        ability = poke.get("ability")
        item = poke.get("item")
        ability_str = ability.get('value', '?') if isinstance(ability, dict) else (ability or '?')
        item_str = item.get('value', '?') if isinstance(item, dict) else (item or '?')
        lines.append(f"    Ability: {ability_str} | Item: {item_str}")
    
    # Bench info - now per-slot format
    lines.append("")
    lines.append("AVAILABLE SWITCHES:")
    switches = context.get("available_switches", [])
    if switches:
        for slot_idx, slot_switches in enumerate(switches):
            if isinstance(slot_switches, list):
                if slot_switches:
                    lines.append(f"  Slot {slot_idx + 1}: {', '.join(slot_switches)}")
                else:
                    lines.append(f"  Slot {slot_idx + 1}: (none)")
            else:
                # Fallback to old format
                lines.append(f"  - {slot_switches}")
    else:
        lines.append("  (none)")
    
    lines.append("")
    lines.append("Choose actions for both Pokemon.")
    
    return "\n".join(lines)


def _build_deep_system_prompt() -> str:
    """Build system prompt for deep strategy."""
    return """You are an expert Pokemon VGC doubles battle AI performing deep analysis.

CRITICAL: TARGET RULES (MUST FOLLOW)
=====================================
SPREAD MOVES (hit all opponents at once):
- hypervoice, dazzlinggleam, heatwave, rockslide, earthquake, etc.
- These moves CANNOT have a target! OMIT the "target" field or use 0!
- Server error "You can't choose a target for X" means you wrongly specified target

SINGLE TARGET MOVES:
- target: 1 = opponent's LEFT Pokemon (slot 1)
- target: 2 = opponent's RIGHT Pokemon (slot 2)

NEVER USE target: -1 FOR ATTACKING MOVES!
-1 means "ally" and will attack your own partner Pokemon!
-2 means "self" - only for moves like Protect, Swords Dance

SWITCH RULES (MUST FOLLOW)
=====================================
- Both slots CANNOT switch to the same Pokemon
- You can only switch to bench Pokemon (not already active)
- If both slots need to switch, choose DIFFERENT Pokemon for each

For switches:
{
    "slot1": {"action": "switch", "pokemon": "<species1>"},
    "slot2": {"action": "switch", "pokemon": "<species2>"}
}

ANALYSIS FRAMEWORK:
1. Speed Tier Analysis: Who moves first? Consider Tailwind, paralysis, abilities
2. Damage Calculation: Can we KO? Can they KO us? Use function calling
3. Win Condition: What's our path to victory? Preserve key Pokemon
4. Prediction: What will opponent likely do? Protect patterns, switch patterns
5. Risk Assessment: Downside of each option if opponent does X/Y/Z

Use function calling:
- calculate_damage: Verify KO thresholds
- check_speed_order: Confirm speed order with field effects
- get_tera_matchup: Check type changes from Terastallization

RESPONSE FORMAT (JSON):
{
    "slot1": {"action": "move", "move": "<move_id>", "target": <1|2|0>},
    "slot2": {"action": "move", "move": "<move_id>", "target": <1|2|0>}
}

Optional: "terastallize": true for Terastallizing"""


def _build_deep_prompt(context: Dict) -> str:
    """Build comprehensive prompt for deep strategy - ALL available info."""
    lines = [f"=== TURN {context.get('turn', '?')} - DEEP ANALYSIS ==="]
    
    # CRITICAL: Check force_switch situation FIRST
    force_switch = context.get("force_switch", [False, False])
    if force_switch[0] or force_switch[1]:
        lines.append("")
        lines.append("*** FORCE SWITCH REQUIRED ***")
        if force_switch[0]:
            lines.append("  SLOT 1: MUST SWITCH (fainted/forced out)")
            lines.append("    - Cannot use moves, must select a Pokemon to switch in")
        if force_switch[1]:
            lines.append("  SLOT 2: MUST SWITCH (fainted/forced out)")
            lines.append("    - Cannot use moves, must select a Pokemon to switch in")
        lines.append("")
    
    # Field conditions - complete info
    field = context.get("field", {})
    lines.append("")
    lines.append("FIELD STATE:")
    lines.append(f"  Weather: {field.get('weather') or 'None'}")
    lines.append(f"  Terrain: {field.get('terrain') or 'None'}")
    if field.get('trick_room'):
        lines.append(f"  [TRICK ROOM] slower moves first")
    if field.get('my_tailwind'):
        lines.append(f"  [YOUR TAILWIND] 2x speed")
    if field.get('opp_tailwind'):
        lines.append(f"  [OPP TAILWIND] 2x speed")
    speed_mode = context.get('speed_mode', 'normal')
    lines.append(f"  Speed Mode: {speed_mode}")
    
    # My team full status
    lines.append("")
    lines.append("=== YOUR TEAM ===")
    for poke in context.get("my_active", []):
        types = "/".join(poke.get("types", [])) if poke.get("types") else "unknown"
        lines.append(f"")
        lines.append(f"[ACTIVE Slot {poke['slot']}] {poke['species']} ({types})")
        lines.append(f"  HP: {poke['hp_percent']}% | Status: {poke.get('status') or 'healthy'}")
        lines.append(f"  Ability: {poke.get('ability') or 'unknown'} | Item: {poke.get('item') or 'unknown'}")
        
        # Terastallize info - always show status
        if poke.get("terastallized"):
            lines.append(f"  [TERA ACTIVE] -> {poke.get('tera_type', 'unknown')}")
        elif poke.get("can_terastallize"):
            lines.append(f"  [CAN TERA] -> {poke.get('tera_type') or 'unknown type'}")
        else:
            lines.append(f"  [NO TERA] already used or unavailable")
        
        # Turns on field tracking
        if poke.get("turns_on_field"):
            lines.append(f"  Turns on field: {poke['turns_on_field']}")
        
        lines.append(f"  AVAILABLE MOVES:")
        for move in poke.get("moves", []):
            bp = move.get('base_power', 0) or '-'
            targets = move.get('valid_targets', [0])
            lines.append(f"    - {move['id']}: {move.get('type','?')}/{move.get('category','?')} BP={bp} targets={targets}")
    
    # Bench Pokemon with detailed info
    lines.append("")
    lines.append("BENCH:")
    for p in context.get("my_team", []):
        if not p.get("active") and not p.get("fainted"):
            status = f" ({p.get('status')})" if p.get('status') else ""
            selected = " [SEL]" if p.get("selected") else ""
            was_lead = " [LEAD]" if p.get("is_lead") else ""
            tera = f" Tera:{p['tera_type']}" if p.get("tera_type") else ""
            lines.append(f"  - {p['species']} ({p['hp_percent']}%){status}{tera}{selected}{was_lead}")
    
    # Fainted Pokemon
    fainted = [p for p in context.get("my_team", []) if p.get("fainted")]
    if fainted:
        lines.append("")
        lines.append("FAINTED:")
        for p in fainted:
            lines.append(f"  - {p['species']} (KO)")
    
    # Opponent info - complete with types, tera, item, ability
    lines.append("")
    lines.append("=== OPPONENT ===")
    for poke in context.get("opponent_active", []):
        types = "/".join(poke.get("types", [])) if poke.get("types") else "unknown"
        lines.append(f"")
        lines.append(f"[ACTIVE Slot {poke['slot']}] {poke['species']} ({types})")
        lines.append(f"  HP: {poke['hp_percent']}% | Status: {poke.get('status') or 'healthy'}")
        
        # Tera status
        if poke.get("terastallized"):
            lines.append(f"  [TERA ACTIVE] -> {poke.get('tera_type', 'unknown')}")
        elif poke.get("tera_type"):
            lines.append(f"  Tera Type (if used): {poke.get('tera_type')}")
        
        # Ability - show probability if inferred
        ability = poke.get("ability")
        if ability:
            if isinstance(ability, dict):
                lines.append(f"  Ability: {ability.get('value', 'unknown')} ({ability.get('probability', 0):.0f}% confidence)")
            else:
                lines.append(f"  Ability: {ability}")
        else:
            lines.append(f"  Ability: unknown")
        
        # Item - show probability if inferred
        item = poke.get("item")
        if item:
            if isinstance(item, dict):
                lines.append(f"  Item: {item.get('value', 'unknown')} ({item.get('probability', 0):.0f}% confidence)")
            else:
                lines.append(f"  Item: {item}")
        else:
            lines.append(f"  Item: unknown")
        
        # Known moves
        known = poke.get("known_moves", [])
        if known:
            lines.append(f"  Known moves: {', '.join(known)}")
        else:
            lines.append(f"  Known moves: none revealed yet")
        
        # Predicted moves from state tracking
        predicted = poke.get("predicted_moves", [])
        if predicted:
            lines.append(f"  Predicted moves (by usage):")
            for m in predicted[:5]:
                conf = m.get('confidence', 0)
                used = m.get('used_count', 0)
                lines.append(f"    - {m.get('move', '?')} (used {used}x, {conf:.0f}% conf)")
        
        # Speed info (critical for VGC)
        speed = poke.get("speed")
        if speed:
            lines.append(f"  Speed: {speed}")
        eff_speed = poke.get("effective_speed")
        if eff_speed:
            lines.append(f"  Effective Speed (current): {eff_speed}")
        
        # Stat boosts
        boosts = poke.get("stat_boosts")
        if boosts:
            boost_str = ", ".join([f"{k}:{'+' if v > 0 else ''}{v}" for k, v in boosts.items()])
            lines.append(f"  Stat boosts: {boost_str}")
        
        # Item/Ability effects observed
        if poke.get("item_effect_observed"):
            lines.append(f"  Item effect seen: {poke['item_effect_observed']}")
        if poke.get("ability_effect_observed"):
            lines.append(f"  Ability effect seen: {poke['ability_effect_observed']}")
        
        # Turns on field
        if poke.get("turns_on_field"):
            lines.append(f"  Turns on field: {poke['turns_on_field']}")
    
    # Opponent bench with full info
    lines.append("")
    lines.append("OPPONENT BENCH (revealed):")
    for p in context.get("opponent_revealed", []):
        if not p.get("active") and not p.get("fainted"):
            types = "/".join(p.get("types", [])) if p.get("types") else "?"
            status = f" [{p.get('status')}]" if p.get("status") else ""
            was_lead = " [WAS LEAD]" if p.get("was_lead") else ""
            lines.append(f"  - {p['species']} ({types}) {p['hp_percent']}%{status}{was_lead}")
            
            # Item/Ability
            item = p.get("item")
            ability = p.get("ability")
            if item or ability:
                item_str = item.get('value', '?') if isinstance(item, dict) else (item or '?')
                ability_str = ability.get('value', '?') if isinstance(ability, dict) else (ability or '?')
                lines.append(f"      Item: {item_str} | Ability: {ability_str}")
            
            # Known/predicted moves
            moves = p.get("known_moves", [])
            predicted = p.get("predicted_moves", [])
            if moves:
                lines.append(f"      Known: {', '.join(moves)}")
            if predicted:
                pred_names = [m.get('move', '?') for m in predicted[:4]]
                lines.append(f"      Predicted: {', '.join(pred_names)}")
            
            # Speed info
            speed = p.get("speed")
            if speed:
                lines.append(f"      Speed: {speed}")
    
    # Opponent fainted
    opp_fainted = [p for p in context.get("opponent_revealed", []) if p.get("fainted")]
    if opp_fainted:
        lines.append("")
        lines.append("OPPONENT FAINTED:")
        for p in opp_fainted:
            lines.append(f"  - {p['species']} (KO)")
    
    # Available switches - detailed for deep analysis
    # CRITICAL: Show which slot needs switch and what options are available
    lines.append("")
    lines.append("=== AVAILABLE SWITCHES ===")
    if force_switch[0] or force_switch[1]:
        lines.append("(Remember: slots with FORCE SWITCH must use action='switch')")
    
    switches_detailed = context.get("available_switches_detailed", [])
    switches_simple = context.get("available_switches", [])
    
    if switches_detailed:
        for slot_idx, slot_switches in enumerate(switches_detailed):
            slot_label = f"Slot {slot_idx + 1}"
            if force_switch[slot_idx]:
                slot_label += " [MUST SWITCH]"
            lines.append(f"{slot_label}:")
            if slot_switches:
                for sw in slot_switches:
                    types = "/".join(sw.get("types", [])) if sw.get("types") else "?"
                    status = f" [{sw.get('status')}]" if sw.get("status") else ""
                    lines.append(f"  - {sw['species']} ({types}) {sw['hp_percent']}%{status}")
                    lines.append(f"      Ability: {sw.get('ability') or '?'} | Item: {sw.get('item') or '?'}")
                    moves = sw.get("moves", [])
                    if moves:
                        lines.append(f"      Moves: {', '.join(moves)}")
            else:
                lines.append(f"  (no switches available for this slot)")
    elif switches_simple:
        # Fallback to simple format
        for slot_idx, slot_switches in enumerate(switches_simple):
            if isinstance(slot_switches, list):
                lines.append(f"For Slot {slot_idx + 1}: {', '.join(slot_switches) if slot_switches else 'none'}")
            else:
                lines.append(f"Available: {', '.join(switches_simple)}")
                break
    else:
        lines.append("  No switches available")
    
    # Strategy context from BattleState
    strategy = context.get("strategy")
    if strategy:
        lines.append("")
        lines.append("=== STRATEGY CONTEXT ===")
        if strategy.get("win_condition"):
            lines.append(f"  Win Condition: {strategy['win_condition']}")
        if strategy.get("threats"):
            threats = strategy['threats']
            if isinstance(threats, list):
                lines.append(f"  Key Threats: {', '.join(threats)}")
            else:
                lines.append(f"  Key Threats: {threats}")
        if strategy.get("tera_plan"):
            lines.append(f"  Tera Plan: {strategy['tera_plan']}")
    
    # Recent turn history
    turn_history = context.get("turn_history", [])
    if turn_history:
        lines.append("")
        lines.append("=== RECENT TURN HISTORY (last 5) ===")
        for turn in turn_history[-5:]:
            lines.append(f"  Turn {turn.get('turn', '?')}: {turn.get('summary', 'no summary')}")
    
    lines.append("")
    lines.append("Analyze thoroughly and choose optimal actions for both Pokemon.")
    lines.append("Consider type matchups, speed, potential switches, and win condition.")
    
    return "\n".join(lines)
