"""
Strategy implementations for parallel GPT calls using OpenAI Responses API.
Contains fast, normal, and deep strategy logic.
"""

import json
import time
from typing import Dict, List, Optional, TYPE_CHECKING
from .test_field import responses

if TYPE_CHECKING:
    from openai import OpenAI
    from src.player.functions import ToolExecutor

logger = __import__('logging').getLogger(__name__)
# =============================================================================
# Retry Helper for Temporary Server Errors
# =============================================================================

def _retry_with_exponential_backoff(func, max_retries: int = 5, base_delay: float = 1.0):
    """
    Retry a function with exponential backoff for temporary server errors.
    
    :param func: Function to retry (should return response)
    :param max_retries: Maximum number of retry attempts
    :param base_delay: Initial delay in seconds (doubles each retry)
    :return: Response from successful call
    :raises: Last exception if all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e
            error_str = str(e)
            logger.error(f"API call failed on attempt {attempt + 1}: {error_str}")
            # Check if it's a temporary server error we should retry
            is_500_error = 'Error code: 500' in error_str or 'Internal Server Error' in error_str
            is_429_error = 'Error code: 429' in error_str or 'Too Many Requests' in error_str
            
            if not (is_500_error or is_429_error):
                # Not a retryable error, fail immediately
                raise
            
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Temporary server error (attempt {attempt + 1}/{max_retries}): {error_str}")
                logger.warning(f"Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"All {max_retries} retry attempts failed")
                raise
    
    # Should never reach here, but just in case
    raise last_exception

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
                # Handle both "output_text" and "text" content types
                if content.type in ("output_text", "text"):
                    return content.text
                # Also check for 'text' attribute directly
                if hasattr(content, 'text') and content.text:
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

    # Fast path: single-response request, no tools/function-calling
    try:
        system_prompt = (
    "You are a Pokemon VGC doubles AI for Pokemon Showdown style battles.\n"
    "You will receive a single user message containing a JSON payload with:\n"
    "- battle_context (my_active, opp_active, available_switches, force_switch, field, etc.)\n"
    "- optionally: battle_state, team_preview_cache, opponent_inferred, context_summary\n"
    "\n"
    "Your task is to choose EXACTLY ONE action for each of my active Pokemon: slot1 and slot2.\n"
    "\n"
    "OUTPUT FORMAT (STRICT):\n"
    "Return ONE JSON object only, with exactly two top-level keys: \"slot1\" and \"slot2\".\n"
    "\n"
    "Each slot MUST be one of the following shapes:\n"
    "- Move action:\n"
    "  {\"action\": \"move\", \"move\": \"<move_id>\", \"target\": <int_optional>}\n"
    "- Switch action:\n"
    "  {\"action\": \"switch\", \"switch_to\": \"<switch_target_id>\"}\n"
    "\n"
    "TARGET RULES:\n"
    "- 1 = left opponent, 2 = right opponent.\n"
    "- -1 = ally (partner).\n"
    "- For self-target, spread, or targetless moves, OMIT the \"target\" field entirely OR set it to 0.\n"
    "- Never use any string like \"omit\" as a target value; targets MUST be integers if present.\n"
    "\n"
    "ALLY TARGETING RULES (VERY IMPORTANT):\n"
    "- You MUST NOT select damaging moves that directly target the ally (target = -1).\n"
    "- Only the following are allowed on the ally (target = -1):\n"
    "  * Clearly non-damaging support/status moves (e.g., Helping Hand, \n"
    "    Ally Switch, heals, defensive buffs, etc.), when such moves are listed as usable.\n"
    "  * Situations where the move inherently hits multiple Pokemon (spread moves with target omitted\n"
    "    or target = 0), and the ally being hit is unavoidable by game mechanics.\n"
    "- If a move is single-target and damaging, its target MUST be 1 or 2 (opponents), NOT -1.\n"
    "- When in doubt, prefer targeting opponents or using a safe support/self move over\n"
    "  damaging your own ally.\n"
    "\n"
    "LEGALITY RULES:\n"
    "- Use ONLY moves listed in battle_context.my_active[slot_index].moves.\n"
    "- If battle_context.force_switch[slot_index] is true OR the Pokemon has no usable moves,\n"
    "  you MUST choose a switch action for that slot.\n"
    "- Switch targets (\"switch_to\") MUST be chosen from battle_context.available_switches for that slot.\n"
    "- Do NOT invent Pokemon, moves, items, or targets that are not present in the input payload.\n"
    "- Do NOT mix move and switch fields (e.g., never return action=\"switch\" with a \"move\" key).\n"
    "\n"
    "OUTPUT CONSTRAINTS:\n"
    "- Return ONLY the JSON object with slot1 and slot2.\n"
    "- No explanations, no comments, no markdown, no extra text outside the JSON.\n"
)

        # Minimal context serialization
        user_prompt = json.dumps(battle_context, ensure_ascii=False)

        request_params = {
            "model": model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "text": {"format": {"type": "json_object"}},
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        print(
            f"[FAST] API call: model={model}, temp={temperature}, max_tokens={max_tokens}, response_format=json_object"
        )
        
        # Retry with exponential backoff for temporary server errors
        response = _retry_with_exponential_backoff(
            lambda: client.responses.create(**request_params)
        )

        logger.warning(f"[FAST] Response received: {response}")

        # Track response id for this battle (even though we don't follow up)
        if hasattr(response, "id"):
            battle_response_ids[battle_tag] = response.id

        _track_tokens(response, token_tracker)

        text = _extract_text_content(response)
        if text:
            return text

        return "{}"

    except Exception as e:
        print(
            f"[FAST] Error with params: model={model}, temp={temperature}, max_tokens={max_tokens}"
        )
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
        system_prompt = (
    "You are a Pokemon VGC doubles AI for Pokemon Showdown style battles.\n"
    "You have access to function-calling tools (usage stats, type matchups, speed tiers, damage calc, cached info, etc.).\n"
    "\n"
    "PROCESS FOR THIS RESPONSE (FIRST TURN STEP):\n"
    "1) First, carefully read the AVAILABLE TOOLS schema.\n"
    "2) Then, read the user message JSON payload, which contains:\n"
    "   - battle_context (my_active, opp_active, available_switches, force_switch, field, etc.)\n"
    "   - optionally: battle_state, team_preview_cache, opponent_inferred, context_summary\n"
    "3) Decide whether the information in the payload is ALREADY sufficient to select strong, legal actions\n"
    "   for slot1 and slot2.\n"
    "\n"
    "DECISION LOGIC FOR THIS RESPONSE:\n"
    "- If the provided information is SUFFICIENT to choose good actions:\n"
    "  • DO NOT CALL ANY TOOLS.\n"
    "  • DIRECTLY output the FINAL decision as a single JSON object with slot1 and slot2 actions\n"
    "    (following the action/target rules described below).\n"
    "\n"
    "- If the provided information is NOT sufficient (you need usage stats, matchups, speed, damage, etc.):\n"
    "  • In THIS FIRST RESPONSE, you MUST ONLY issue TOOL CALLS.\n"
    "  • DO NOT output any move/switch decisions yet.\n"
    "  • You are allowed to call MULTIPLE tools in parallel, but ONLY IN THIS FIRST STEP.\n"
    "  • Plan ahead and issue ALL needed tool calls at once (e.g., usage stats for multiple opponent Pokemon,\n"
    "    several matchup or damage checks, etc.).\n"
    "  • After you receive tool outputs in a FOLLOW-UP request, you will then output the final JSON actions.\n"
    "\n"
    "ACTION FORMAT (WHEN YOU DECIDE DIRECTLY WITHOUT TOOLS):\n"
    "- If you decide that no tools are needed and you can choose actions now, your response MUST be\n"
    "  ONE JSON object only, with exactly two top-level keys: \"slot1\" and \"slot2\".\n"
    "\n"
    "Each slot MUST be one of the following shapes:\n"
    "- Move action:\n"
    "  {\"action\": \"move\", \"move\": \"<move_id>\", \"target\": <int_optional>}\n"
    "- Switch action:\n"
    "  {\"action\": \"switch\", \"switch_to\": \"<switch_target_id>\"}\n"
    "\n"
    "TARGET RULES:\n"
    "- 1 = left opponent, 2 = right opponent.\n"
    "- -1 = ally (partner).\n"
    "- For self-target, spread, or targetless moves, OMIT the \"target\" field entirely OR set it to 0.\n"
    "- Never use any string (e.g., \"omit\") as a target value; targets MUST be integers if present.\n"
    "\n"
    "ALLY TARGETING RULES (VERY IMPORTANT):\n"
    "- You MUST NOT select damaging moves that directly target the ally (target = -1).\n"
    "- The ONLY allowed ally targets (target = -1) are:\n"
    "  * Clearly non-damaging support/status moves (e.g., Helping Hand, Follow Me/Rage Powder,\n"
    "    Ally Switch, healing, defensive buffs, etc.), when such moves are listed as usable.\n"
    "  * Situations where a move inherently hits multiple Pokemon (spread moves with target omitted\n"
    "    or target = 0), and the ally being hit is unavoidable by game mechanics.\n"
    "- If a move is single-target and damaging, its target MUST be 1 or 2 (opponents), NOT -1.\n"
    "- When in doubt, prefer targeting opponents or using a safe support/self move over\n"
    "  damaging your own ally.\n"
    "\n"
    "LEGALITY RULES:\n"
    "- Use ONLY moves listed in battle_context.my_active[slot_index].moves.\n"
    "- If battle_context.force_switch[slot_index] is true OR the Pokemon has no usable moves,\n"
    "  you MUST choose a switch action for that slot.\n"
    "- Switch targets (\"switch_to\") MUST be chosen from battle_context.available_switches for that slot.\n"
    "- Do NOT invent Pokemon, moves, items, or targets that are not present in the input payload.\n"
    "- Do NOT mix move and switch fields (e.g., never return action=\"switch\" with a \"move\" key).\n"
    "\n"
    "TERASTALLIZATION RULES (CRITICAL):\n"
    "- You can only Terastallize ONCE per battle.\n"
    "- Check battle_context.tera_available and each Pokemon's can_terastallize flag.\n"
    "- If tera_available is false OR can_terastallize is false for a Pokemon,\n"
    "  you MUST set terastallize=false for that slot.\n"
    "- Never set terastallize=true if the Pokemon is already terastallized.\n"
    "\n"
    "OUTPUT CONSTRAINTS FOR THIS FIRST STEP:\n"
    "- CASE A (no tools needed):\n"
    "  • Return ONLY the final JSON object with slot1 and slot2.\n"
    "- CASE B (tools needed):\n"
    "  • Return ONLY tool calls (no moves/switch decisions, no extra text).\n"
    "- In both cases, do NOT include natural language explanations, comments, or markdown.\n"
)

        user_prompt = json.dumps(battle_context, ensure_ascii=False)

        initial_params = {
            "model": model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        if tools:
            initial_params["tools"] = tools

        print(
            f"[NORMAL] API call: model={model}, temp={temperature}, max_tokens={max_tokens}, tools={'yes' if tools else 'no'}"
        )
        
        # Retry with exponential backoff for temporary server errors
        response = _retry_with_exponential_backoff(
            lambda: client.responses.create(**initial_params)
        )

        # Track response id and tokens
        if hasattr(response, "id"):
            battle_response_ids[battle_tag] = response.id
        _track_tokens(response, token_tracker)

        function_calls = _extract_function_calls(response)
        if not function_calls:
            text = _extract_text_content(response)
            if text:
                return text
            return "{}"

        # Log function call usage
        tool_names = [fc.get("name", "unknown") for fc in function_calls]
        tool_counts = {}
        for name in tool_names:
            tool_counts[name] = tool_counts.get(name, 0) + 1
        logger.warning(f"[NORMAL] Function calls: {len(function_calls)} total - {tool_counts}")

        # Execute each tool once and build tool outputs
        tool_results = []
        for fc in function_calls:
            try:
                args = json.loads(fc.get("arguments", "{}")) if isinstance(fc.get("arguments"), str) else fc.get("arguments", {})
            except json.JSONDecodeError:
                args = {}

            try:
                tool_output = tool_executor.execute(fc.get("name"), args or {})
            except Exception as e:
                tool_output = {"error": str(e)}

            tool_results.append({
                "type": "function_call_output",
                "call_id": fc.get("id"),
                "output": json.dumps(tool_output, ensure_ascii=False)
            })
        
        normal_stage2_system_prompt = (
    "You are a Pokemon VGC doubles AI for Pokemon Showdown style battles.\n"
    "You have ALREADY called tools in a previous step for this turn, and you now receive:\n"
    "- The original battle_context and related payload, AND\n"
    "- The outputs of all tools you requested.\n"
    "\n"
    "For THIS RESPONSE, you MUST NOT request any more tools.\n"
    "You must use the available information (battle_context, battle_state, team_preview_cache,\n"
    "opponent_inferred, tool results, etc.) to choose EXACTLY ONE action for each active Pokemon:\n"
    "slot1 and slot2.\n"
    "\n"
    "OUTPUT FORMAT (STRICT):\n"
    "Return ONE JSON object only, with exactly two top-level keys: \"slot1\" and \"slot2\".\n"
    "\n"
    "Each slot MUST be one of the following shapes:\n"
    "- Move action:\n"
    "  {\"action\": \"move\", \"move\": \"<move_id>\", \"target\": <int_optional>}\n"
    "- Switch action:\n"
    "  {\"action\": \"switch\", \"switch_to\": \"<switch_target_id>\"}\n"
    "\n"
    "TARGET RULES:\n"
    "- 1 = left opponent, 2 = right opponent.\n"
    "- -1 = ally (partner).\n"
    "- For self-target, spread, or targetless moves, OMIT the \"target\" field entirely OR set it to 0.\n"
    "- Never use any string (e.g., \"omit\") as a target value; targets MUST be integers if present.\n"
    "\n"
    "ALLY TARGETING RULES (VERY IMPORTANT):\n"
    "- You MUST NOT select damaging moves that directly target the ally (target = -1).\n"
    "- The ONLY allowed ally targets (target = -1) are:\n"
    "  * Clearly non-damaging support/status moves (e.g., Helping Hand, Follow Me/Rage Powder,\n"
    "    Ally Switch, healing, defensive buffs, etc.), when such moves are listed as usable.\n"
    "  * Situations where a move inherently hits multiple Pokemon (spread moves with target omitted\n"
    "    or target = 0), and the ally being hit is unavoidable by game mechanics.\n"
    "- If a move is single-target and damaging, its target MUST be 1 or 2 (opponents), NOT -1.\n"
    "- When in doubt, prefer targeting opponents or using a safe support/self move over\n"
    "  damaging your own ally.\n"
    "\n"
    "LEGALITY RULES:\n"
    "- Use ONLY moves listed in battle_context.my_active[slot_index].moves.\n"
    "- If battle_context.force_switch[slot_index] is true OR the Pokemon has no usable moves,\n"
    "  you MUST choose a switch action for that slot.\n"
    "- Switch targets (\"switch_to\") MUST be chosen from battle_context.available_switches for that slot.\n"
    "- Do NOT invent Pokemon, moves, items, or targets that are not present in the input payload.\n"
    "- Do NOT mix move and switch fields (e.g., never return action=\"switch\" with a \"move\" key).\n"
    "\n"
    "OUTPUT CONSTRAINTS:\n"
    "- Return ONLY the final JSON object with slot1 and slot2.\n"
    "- No natural language explanations, no comments, no markdown, no other keys.\n"
)


        followup_input = [
            # Stage 2: 도구 사용 금지 + 최종 JSON만 요구
            {
                "role": "system",
                "content": normal_stage2_system_prompt,
            },
            *tool_results,
        ]

        # Stage 2에서는 tools를 API 레벨에서 제거하여 모델이 도구를 못 쓰게 강제
        followup_params = {
            "model": model,
            "previous_response_id": response.id,
            "input": followup_input,
            "text": {"format": {"type": "json_object"}},
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        # tools를 의도적으로 제거 - stage2에서는 도구 사용 불가

        print(
            f"[NORMAL] Follow-up: model={model}, prev_id={response.id}, tool_results={len(tool_results)}"
        )
        
        # Retry with exponential backoff for temporary server errors
        followup_response = _retry_with_exponential_backoff(
            lambda: client.responses.create(**followup_params)
        )

        # Track latest response id and tokens
        if hasattr(followup_response, "id"):
            battle_response_ids[battle_tag] = followup_response.id
        _track_tokens(followup_response, token_tracker)

        text = _extract_text_content(followup_response)
        if text:
            return text

        return "{}"

    except Exception as e:
        print(
            f"[NORMAL] Error with params: model={model}, temp={temperature}, max_tokens={max_tokens}"
        )
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
    reasoning_effort: str = "medium",
    max_iterations: Optional[int] = None
) -> str:
    """
    Deep strategy: Comprehensive analysis with iterative refinement.
    Uses extended reasoning and extensive tool usage.
    
    IMPORTANT: Tool call flow for Responses API:
    1. Initial request -> may return function_call items
    2. Execute tools locally
    3. Send tool results with previous_response_id to continue conversation
    4. Repeat until text response is received
    
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
        system_prompt = (
    "You are a high-level Pokemon VGC doubles strategist.\n"
    "Your role is to perform DEEP search-style planning: not only evaluate the current turn,\n"
    "but also mentally branch over the next 1–2 turns to estimate which present actions\n"
    "lead to the best overall position.\n"
    "\n"
    "You are allowed to call tools MULTIPLE TIMES in an iterative process.\n"
    "Use tools to refine your evaluation (usage stats, type/tera matchups, speed order,\n"
    "damage ranges, cached preview info, etc.), prune bad branches, and focus on the\n"
    "most promising action sequences.\n"
    "\n"
    "==================== INPUT CONTEXT ====================\n"
    "You will receive a JSON payload (as the user message) containing:\n"
    "- battle_context: current turn state (my_active, opp_active, available_switches,\n"
    "  force_switch, field, turn, etc.).\n"
    "- optionally: battle_state, team_preview_cache, opponent_inferred, context_summary.\n"
    "\n"
    "Treat this as the single source of truth, together with any tool outputs.\n"
    "Do NOT invent Pokemon, moves, items, or state that are not present in the payload or tools.\n"
    "\n"
    "==================== DEEP REASONING & ITERATIVE TOOL USE ====================\n"
    "Use your internal reasoning to simulate and compare multiple candidate lines,\n"
    "including this turn and likely follow-up turns. For each candidate, consider:\n"
    "- Offensive pressure (KOs, 2HKOs, chip, speed control).\n"
    "- Defensive stability (survival, positioning, board control).\n"
    "- Tempo/tools already available (Trick Room, Tailwind, speed tiers, tera options, etc.).\n"
    "\n"
    "You will proceed in ITERATIONS:\n"
    "- In each iteration, you receive the current context and (after the first iteration)\n"
    "  any tool results from previous calls.\n"
    "- You MUST decide whether more tools are needed to evaluate and prune candidate lines,\n"
    "  or whether you already have enough information to choose final actions.\n"
    "\n"
    "TOOL CALLING RULES (ITERATIVE):\n"
    "- As long as you still need information, you MAY call tools again in a new iteration.\n"
    "- In any response that includes tool calls, you MUST:\n"
    "  1) Issue ONLY the tool calls you currently need (one or many in parallel), AND\n"
    "  2) Include a brief natural-language explanation describing:\n"
    "     - Which branches or scenarios you are investigating next, and\n"
    "     - For each tool call, what you intend to learn and how you plan to interpret\n"
    "       its result for your decision.\n"
    "- Keep this explanation concise and focused; it is for logging/debugging, not for the user.\n"
    "- When a response contains tool calls, you MUST NOT output any slot1/slot2 actions.\n"
    "- Reuse previous tool results when possible and avoid redundant calls.\n"
    "\n"
    "STOPPING CRITERION:\n"
    "- Once you are reasonably confident about which present actions are best (after evaluating\n"
    "  and pruning enough branches), STOP calling tools.\n"
    "- When you stop calling tools, your next response MUST be the FINAL ACTION JSON only.\n"
    "\n"
    "==================== FINAL ACTION FORMAT ====================\n"
    "When you finish your deep evaluation and stop using tools, you MUST return a single JSON\n"
    "object with exactly two top-level keys: \"slot1\" and \"slot2\".\n"
    "\n"
    "Each slot MUST be one of the following shapes:\n"
    "- Move action:\n"
    "  {\"action\": \"move\", \"move\": \"<move_id>\", \"target\": <int_optional>}\n"
    "- Switch action:\n"
    "  {\"action\": \"switch\", \"switch_to\": \"<switch_target_id>\"}\n"
    "\n"
    "TARGET RULES:\n"
    "- 1 = left opponent, 2 = right opponent.\n"
    "- -1 = ally (partner).\n"
    "- For self-target, spread, or targetless moves, OMIT the \"target\" field entirely\n"
    "  OR set it to 0.\n"
    "- Never use any string (e.g., \"omit\") as a target value; targets MUST be integers\n"
    "  if present.\n"
    "\n"
    "ALLY TARGETING RULES (CRITICAL):\n"
    "- You MUST NOT select damaging moves that directly target the ally (target = -1).\n"
    "- The ONLY allowed ally targets (target = -1) are:\n"
    "  • Clearly non-damaging support/status moves (e.g., Helping Hand, Follow Me/Rage Powder,\n"
    "    Ally Switch, healing, defensive buffs, etc.) when such moves are usable.\n"
    "  • Situations where a move inherently hits multiple Pokemon (spread moves with target\n"
    "    omitted or target = 0), and the ally being hit is unavoidable by game mechanics.\n"
    "- If a move is single-target and damaging, its target MUST be 1 or 2 (opponents), NOT -1.\n"
    "- When in doubt, prefer targeting opponents or using a safe support/self move over\n"
    "  damaging your own ally.\n"
    "\n"
"LEGALITY RULES:\n"
    "- Use ONLY moves listed in battle_context.my_active[slot_index].moves.\n"
    "- If battle_context.force_switch[slot_index] is true OR the Pokemon has no usable moves,\n"
    "  you MUST choose a switch action for that slot.\n"
    "- Switch targets (\"switch_to\") MUST be chosen from battle_context.available_switches\n"
    "  for that slot.\n"
    "- Do NOT invent Pokemon, moves, items, or targets that are not present in the input\n"
    "  payload or tool outputs.\n"
    "- Do NOT mix move and switch fields (e.g., never return action=\"switch\" with a \"move\" key).\n"
    "\n"
    "TERASTALLIZATION RULES (CRITICAL):\n"
    "- You can only Terastallize ONCE per battle.\n"
    "- Check battle_context.tera_available and each Pokemon's can_terastallize flag.\n"
    "- If tera_available is false OR can_terastallize is false for a Pokemon,\n"
    "  you MUST set terastallize=false for that slot.\n"
    "- Never set terastallize=true if the Pokemon is already terastallized.\n"
    "\n"
    "==================== OUTPUT CONSTRAINTS ===================="
    "- INTERMEDIATE RESPONSES (while still exploring):\n"
    "  • MAY contain: tool calls + a brief natural-language explanation of why/how they are used.\n"
    "  • MUST NOT contain any slot1/slot2 actions.\n"
    "- FINAL RESPONSE (after finishing tool usage):\n"
    "  • MUST be ONLY the JSON object with slot1 and slot2.\n"
    "  • No explanations, no comments, no markdown, no other keys.\n"
)


        user_prompt = json.dumps(battle_context, ensure_ascii=False)

        params = {
            "model": model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "reasoning": {"effort": reasoning_effort},
        }

        # max_tokens <= 0이면 제한 없이 요청, 양수이면 지정
        if max_tokens > 0:
            params["max_output_tokens"] = max_tokens

        if tools:
            params["tools"] = tools

        print(
            f"[DEEP] API call: model={model}, temp={temperature}, max_tokens={max_tokens}, tools={'yes' if tools else 'no'}, effort={reasoning_effort}"
        )
        
        # Retry with exponential backoff for temporary server errors
        response = _retry_with_exponential_backoff(
            lambda: client.responses.create(**params)
        )

        # Track response id and tokens
        if hasattr(response, "id"):
            battle_response_ids[battle_tag] = response.id
        _track_tokens(response, token_tracker)

        # If max_iterations is None, run until model stops calling tools, but keep a hard safety cap
        safety_cap = max_iterations if isinstance(max_iterations, int) else 20
        iteration = 0

        while iteration < safety_cap:
            iteration += 1
            function_calls = _extract_function_calls(response)

            # No tool calls -> try to return content
            if not function_calls:
                text = _extract_text_content(response)
                if text:
                    return text
                return "{}"

            # Log function call usage
            tool_names = [fc.get("name", "unknown") for fc in function_calls]
            tool_counts = {}
            for name in tool_names:
                tool_counts[name] = tool_counts.get(name, 0) + 1
            logger.warning(f"[DEEP] Iteration {iteration}/{safety_cap} - Function calls: {len(function_calls)} total - {tool_counts}")

            # Execute tool calls in sequence (could be parallelized externally if needed)
            tool_results = []
            for fc in function_calls:
                try:
                    args = json.loads(fc.get("arguments", "{}")) if isinstance(fc.get("arguments"), str) else fc.get("arguments", {})
                except json.JSONDecodeError:
                    args = {}

                try:
                    tool_output = tool_executor.execute(fc.get("name"), args or {})
                except Exception as e:
                    tool_output = {"error": str(e)}

                tool_results.append({
                    "type": "function_call_output",
                    "call_id": fc.get("id"),
                    "output": json.dumps(tool_output, ensure_ascii=False)
                })

            print(
                f"[DEEP] Follow-up {iteration}/{safety_cap}: model={model}, prev_id={response.id}, tool_results={len(tool_results)}"
            )

            # Deep follow-up: response_format 지정 안 함 (중간 iteration에서 자유로운 tool + text 허용)
            followup_params = {
                "model": model,
                "previous_response_id": response.id,
                "input": tool_results,
            }

            if max_tokens > 0:
                followup_params["max_output_tokens"] = max_tokens

            if tools:
                followup_params["tools"] = tools

            # Retry with exponential backoff for temporary server errors
            response = _retry_with_exponential_backoff(
                lambda: client.responses.create(**followup_params)
            )

            # Track response id and tokens
            if hasattr(response, "id"):
                battle_response_ids[battle_tag] = response.id
            _track_tokens(response, token_tracker)

        # Safety cap reached: if 여전히 function_call만 있으면 강제 마무리 요청
        final_calls = _extract_function_calls(response)
        if final_calls:
            final_params = {
                "model": model,
                "previous_response_id": response.id,
                "input": [
                    {
                        "role": "user",
                        "content": "Stop calling tools. Return final JSON for slot1/slot2 actions now."
                    }
                ],
                "text": {"format": {"type": "json_object"}},
            }

            if max_tokens > 0:
                final_params["max_output_tokens"] = max_tokens

            print(f"[DEEP] Safety cap reached; forcing final response (prev_id={response.id})")
            
            # Retry with exponential backoff for temporary server errors
            response = _retry_with_exponential_backoff(
                lambda: client.responses.create(**final_params)
            )

            if hasattr(response, "id"):
                battle_response_ids[battle_tag] = response.id
            _track_tokens(response, token_tracker)

        # Return best effort text
        text = _extract_text_content(response)
        if text:
            return text
        return "{}"

    except Exception as e:
        print(
            f"[DEEP] Error with params: model={model}, temp={temperature}, max_tokens={max_tokens}"
        )
        raise RuntimeError(f"Deep strategy failed: {e}")
