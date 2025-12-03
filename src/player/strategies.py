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

RESPONSE FORMAT:
{
    "slot1": {"action": "move", "move": "<move_id>", "target": <1|2|-1|-2|0>},
    "slot2": {"action": "move", "move": "<move_id>", "target": <1|2|-1|-2|0>}
}
Targets: 1=opp left, 2=opp right, -1=ally, -2=self, 0=spread/self"""
        
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
        
        if battle_tag in battle_response_ids:
            request_params["previous_response_id"] = battle_response_ids[battle_tag]
        
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
        
        if battle_tag in battle_response_ids:
            request_params["previous_response_id"] = battle_response_ids[battle_tag]
        
        if tools:
            request_params["tools"] = tools
        else:
            request_params["text"] = {"format": {"type": "json_object"}}
        
        response = client.responses.create(**request_params)
        
        if hasattr(response, 'id') and battle_tag not in battle_response_ids:
            battle_response_ids[battle_tag] = response.id
        
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
                tool_executor=tool_executor
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
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "reasoning": {"effort": reasoning_effort}
            }
            
            if battle_tag in battle_response_ids:
                request_params["previous_response_id"] = battle_response_ids[battle_tag]
            
            if tools:
                request_params["tools"] = tools
            
            response = client.responses.create(**request_params)
            
            if hasattr(response, 'id') and battle_tag not in battle_response_ids:
                battle_response_ids[battle_tag] = response.id
            
            _track_tokens(response, token_tracker)
            if hasattr(response, 'usage') and response.usage:
                progress_tracker[progress_key]["tokens_used"] += getattr(response.usage, 'output_tokens', 0)
            
            # Check for function calls
            func_calls = _extract_function_calls(response)
            if func_calls:
                for fc in func_calls:
                    func_args = json.loads(fc["arguments"])
                    result = tool_executor.execute(fc["name"], func_args)
                    _add_tool_result(progress_tracker, progress_key, fc["name"], func_args, result)
                    conversation.append({
                        "role": "function",
                        "call_id": fc["id"],
                        "output": json.dumps(result)
                    })
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
    tool_executor: "ToolExecutor"
) -> str:
    """Handle single round of tool calls for normal strategy."""
    function_results = []
    
    for fc in func_calls:
        func_args = json.loads(fc["arguments"])
        result = tool_executor.execute(fc["name"], func_args)
        function_results.append({
            "role": "function",
            "call_id": fc["id"],
            "output": json.dumps(result)
        })
    
    request_params = {
        "model": model,
        "input": function_results,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "text": {"format": {"type": "json_object"}}
    }
    
    if battle_tag in battle_response_ids:
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
    """Build simplified prompt for fast strategy."""
    lines = [f"Turn {context.get('turn', '?')}"]
    
    # My active Pokemon - simplified
    for poke in context.get("my_active", []):
        moves = [m["id"] for m in poke.get("moves", [])]
        lines.append(f"Slot{poke['slot']}: {poke['species']} ({poke['hp_percent']}%) - {', '.join(moves)}")
    
    # Opponent active - simplified
    for poke in context.get("opponent_active", []):
        lines.append(f"Opp{poke['slot']}: {poke['species']} ({poke['hp_percent']}%)")
    
    # Available switches
    switches = context.get("available_switches", [])
    if switches:
        lines.append(f"Switches: {', '.join(switches)}")
    
    return "\n".join(lines)


def _build_normal_system_prompt() -> str:
    """Build system prompt for normal strategy."""
    return """You are a Pokemon VGC doubles battle AI.
Analyze the battle state and choose optimal actions for both Pokemon.

Use function calling to:
- calculate_damage: Check if you can KO or deal significant damage
- evaluate_speed_tie: Determine who moves first
- calculate_matchup_outcome: Evaluate overall matchup

RESPONSE FORMAT (JSON):
{
    "slot1": {"action": "move", "move": "<move_id>", "target": <target_int>},
    "slot2": {"action": "switch", "pokemon": "<species>"}
}

Target indices: 1=opponent left, 2=opponent right, -1=ally, -2=self, 0=no target
Optional: "terastallize": true"""


def _build_normal_prompt(context: Dict) -> str:
    """Build detailed prompt for normal strategy."""
    lines = [f"=== Turn {context.get('turn', '?')} ==="]
    
    # Field conditions
    field = context.get("field", {})
    if field.get("weather") or field.get("terrain"):
        conditions = []
        if field.get("weather"): conditions.append(f"Weather: {field['weather']}")
        if field.get("terrain"): conditions.append(f"Terrain: {field['terrain']}")
        lines.append(" | ".join(conditions))
    
    lines.append("")
    lines.append("YOUR POKEMON:")
    for poke in context.get("my_active", []):
        types = "/".join(poke.get("types", []))
        lines.append(f"  Slot {poke['slot']}: {poke['species']} [{types}] HP:{poke['hp_percent']}%")
        if poke.get("status"): lines.append(f"    Status: {poke['status']}")
        lines.append(f"    Ability: {poke.get('ability', '?')} | Item: {poke.get('item', '?')}")
        if poke.get("can_terastallize"):
            lines.append(f"    Can Tera -> {poke.get('tera_type', '?')}")
        lines.append("    Moves:")
        for move in poke.get("moves", []):
            targets = move.get("valid_targets", [0])
            lines.append(f"      - {move['id']} ({move.get('type','?')}/{move.get('category','?')}) BP:{move.get('base_power','-')} targets:{targets}")
    
    lines.append("")
    lines.append("OPPONENT POKEMON:")
    for poke in context.get("opponent_active", []):
        lines.append(f"  Slot {poke['slot']}: {poke['species']} HP:{poke['hp_percent']}%")
        if poke.get("status"): lines.append(f"    Status: {poke['status']}")
        known_moves = poke.get("known_moves", [])
        if known_moves:
            lines.append(f"    Known moves: {', '.join(known_moves)}")
        lines.append(f"    Ability: {poke.get('ability', '?')} | Item: {poke.get('item', '?')}")
    
    # Bench info
    lines.append("")
    lines.append("AVAILABLE SWITCHES:")
    for species in context.get("available_switches", []):
        lines.append(f"  - {species}")
    
    lines.append("")
    lines.append("Choose actions for both Pokemon.")
    
    return "\n".join(lines)


def _build_deep_system_prompt() -> str:
    """Build system prompt for deep strategy."""
    return """You are an expert Pokemon VGC doubles battle AI performing deep analysis.

ANALYSIS FRAMEWORK:
1. Speed Tier Analysis: Who moves first? Consider Tailwind, paralysis, abilities
2. Damage Calculation: Can we KO? Can they KO us? Use function calling
3. Win Condition: What's our path to victory? Preserve key Pokemon
4. Prediction: What will opponent likely do? Protect patterns, switch patterns
5. Risk Assessment: Downside of each option if opponent does X/Y/Z

Use function calling extensively:
- calculate_damage: Verify KO thresholds
- evaluate_speed_tie: Confirm speed order
- calculate_matchup_outcome: Overall matchup score
- calculate_win_probability: Win chance analysis

RESPONSE FORMAT (JSON):
{
    "slot1": {"action": "move", "move": "<move_id>", "target": <target_int>},
    "slot2": {"action": "move", "move": "<move_id>", "target": <target_int>}
}

Target indices: 1=opponent left, 2=opponent right, -1=ally, -2=self, 0=no target
Optional flags: "terastallize": true"""


def _build_deep_prompt(context: Dict) -> str:
    """Build comprehensive prompt for deep strategy."""
    lines = [f"=== TURN {context.get('turn', '?')} - DEEP ANALYSIS ==="]
    
    # Field
    field = context.get("field", {})
    lines.append("")
    lines.append("FIELD STATE:")
    lines.append(f"  Weather: {field.get('weather') or 'None'}")
    lines.append(f"  Terrain: {field.get('terrain') or 'None'}")
    
    # My team full status
    lines.append("")
    lines.append("=== YOUR TEAM ===")
    for poke in context.get("my_active", []):
        types = "/".join(poke.get("types", []))
        lines.append(f"")
        lines.append(f"[ACTIVE Slot {poke['slot']}] {poke['species']} ({types})")
        lines.append(f"  HP: {poke['hp_percent']}% | Status: {poke.get('status') or 'healthy'}")
        lines.append(f"  Ability: {poke.get('ability', 'unknown')} | Item: {poke.get('item', 'unknown')}")
        if poke.get("can_terastallize"):
            lines.append(f"  ★ Can Terastallize -> {poke.get('tera_type', 'unknown')}")
        lines.append(f"  AVAILABLE MOVES:")
        for move in poke.get("moves", []):
            bp = move.get('base_power', 0) or '-'
            targets = move.get('valid_targets', [0])
            lines.append(f"    • {move['id']}: {move.get('type','?')}/{move.get('category','?')} BP={bp} targets={targets}")
    
    lines.append("")
    lines.append("BENCH:")
    for p in context.get("my_team", []):
        if not p.get("active") and not p.get("fainted"):
            lines.append(f"  - {p['species']} ({p['hp_percent']}%)")
    
    # Opponent info
    lines.append("")
    lines.append("=== OPPONENT ===")
    for poke in context.get("opponent_active", []):
        lines.append(f"")
        lines.append(f"[ACTIVE Slot {poke['slot']}] {poke['species']}")
        lines.append(f"  HP: {poke['hp_percent']}% | Status: {poke.get('status') or 'healthy'}")
        lines.append(f"  Ability: {poke.get('ability', 'unknown')} | Item: {poke.get('item', 'unknown')}")
        known = poke.get("known_moves", [])
        if known:
            lines.append(f"  Known moves: {', '.join(known)}")
    
    lines.append("")
    lines.append("OPPONENT BENCH (revealed):")
    for p in context.get("opponent_revealed", []):
        if not p.get("active") and not p.get("fainted"):
            status = f"({p['hp_percent']}%)"
            moves = p.get("known_moves", [])
            move_info = f" knows: {', '.join(moves)}" if moves else ""
            lines.append(f"  - {p['species']} {status}{move_info}")
    
    # Available actions summary
    lines.append("")
    lines.append("AVAILABLE SWITCHES: " + ", ".join(context.get("available_switches", [])))
    
    lines.append("")
    lines.append("Analyze thoroughly and choose optimal actions for both Pokemon.")
    
    return "\n".join(lines)
