import json
import os
import sys
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Any, TypedDict
import logging
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder

from openai import OpenAI

from src.data.gen_data import GenData
from src.environment.abstract_battle import AbstractBattle
from src.environment.double_battle import DoubleBattle
from src.player.player import Player, BattleOrder
from src.player.battle_order import DefaultBattleOrder, DoubleBattleOrder, ForfeitBattleOrder


# =====================================================================
# 1. 코어 상태 dataclass들
# =====================================================================

@dataclass
class PokemonState:
    """확률/추론용 최소 단위 포켓몬 상태."""
    name: str                    # Showdown identifier (e.g., "Flutter Mane")
    side: str                    # "me" or "opponent"
    slot: Optional[int] = None   # doubles: 1 / 2, 필드에 없으면 None

    # 직접 관측 가능한 정보
    hp_percent: Optional[float] = None
    status: Optional[str] = None
    fainted: bool = False
    terastallized: bool = False

    # 관측 데이터
    seen_moves: List[str] = field(default_factory=list)
    item: Optional[str] = None
    ability: Optional[str] = None

    # 확률/추론 힌트 (구현 확장 포인트)
    speed_tier_hint: Optional[str] = None
    ev_profile_hint: Optional[str] = None

    def record_move(self, move_id: str) -> None:
        if move_id not in self.seen_moves:
            self.seen_moves.append(move_id)

    def to_gpt_dict(self) -> Dict[str, Any]:
        """GPT 컨텍스트용 compact dict."""
        return {
            "name": self.name,
            "side": self.side,
            "slot": self.slot,
            "hp_percent": self.hp_percent,
            "status": self.status,
            "fainted": self.fainted,
            "terastallized": self.terastallized,
            "seen_moves": self.seen_moves,
            "item": self.item,
            "ability": self.ability,
            "speed_tier_hint": self.speed_tier_hint,
            "ev_profile_hint": self.ev_profile_hint,
        }


@dataclass
class TeamState:
    """한쪽 팀 전체 상태."""
    side: str                    # "me" or "opponent"
    pokemons: Dict[str, PokemonState] = field(default_factory=dict)
    tera_used: bool = False

    def get_or_create(self, name: str) -> PokemonState:
        if name not in self.pokemons:
            self.pokemons[name] = PokemonState(name=name, side=self.side)
        return self.pokemons[name]

    def active_pokemon(self) -> List[PokemonState]:
        return [p for p in self.pokemons.values() if p.slot is not None and not p.fainted]

    def to_gpt_dict(self) -> Dict[str, Any]:
        return {
            "side": self.side,
            "tera_used": self.tera_used,
            "pokemons": [p.to_gpt_dict() for p in self.pokemons.values()],
        }


@dataclass
class BattleState:
    """양 팀 + 필드 정보까지 포함하는 배틀 전역 상태."""
    battle_tag: str
    battle_format: str

    my_team: TeamState = field(default_factory=lambda: TeamState(side="me"))
    opp_team: TeamState = field(default_factory=lambda: TeamState(side="opponent"))

    # 필드/사이드 조건 (필요할 때 확장)
    weather: Optional[str] = None
    terrain: Optional[str] = None
    trick_room: bool = False
    turn: int = 0

    # team preview에서 수집한 정보 캐시 (battle phase에서 재사용)
    team_preview_cache: Optional[Any] = None

    # 디버깅용 로그
    turn_logs: List[Dict[str, Any]] = field(default_factory=list)

    def to_gpt_dict(self) -> Dict[str, Any]:
        return {
            "battle_tag": self.battle_tag,
            "format": self.battle_format,
            "turn": self.turn,
            "field": {
                "weather": self.weather,
                "terrain": self.terrain,
                "trick_room": self.trick_room,
            },
            "my_team": self.my_team.to_gpt_dict(),
            "opponent_team": self.opp_team.to_gpt_dict(),
        }


# =====================================================================
# 2. GPT 전략 클라이언트 (Responses API 래퍼)
# =====================================================================

class GPTStrategyClient:
    """이 봇 전용 OpenAI Responses API thin wrapper + StrategyEngine 통합."""

    def __init__(
        self,
        backend_model: str,
        api_key: str,
        base_url: Optional[str] = None,
        fast_model: Optional[str] = None,
        deep_model: Optional[str] = None,
        temperature: float = 0.7,
        log_dir: Optional[str] = None,
        use_strategy_engine: bool = True,
        global_timeout: float = 25.0,
    ) -> None:
        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)
        self.backend_model = backend_model
        self.fast_model = fast_model or backend_model
        self.deep_model = deep_model or backend_model
        self.temperature = temperature
        self.log_dir = log_dir
        self.use_strategy_engine = use_strategy_engine

        # 토큰 사용량 누적
        self.token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        
        # StrategyEngine 초기화 (function calling + parallel strategies)
        if self.use_strategy_engine:
            from src.player.strategy_engine import StrategyEngine
            self.strategy_engine = StrategyEngine(
                client=self.client,
                fast_model=self.fast_model,
                backend_model=self.backend_model,
                deep_model=self.deep_model,
                temperature=self.temperature,
                logger=logging.getLogger(__name__),
                global_timeout=global_timeout,
            )
        else:
            self.strategy_engine = None

    # ------------------------------------------------------------------
    # 공용 GPT 호출 래퍼
    # ------------------------------------------------------------------
    def _call_gpt_json(
        self,
        *,
        messages: List[Dict[str, str]],
        max_output_tokens: int,
        temperature: Optional[float] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Responses API JSON 호출 공통 래퍼.

        - messages: Responses API에 넘길 role/content 리스트
        - max_output_tokens: 출력 토큰 제한
        - temperature: 없으면 self.temperature 사용
        - response_format: "json_object" 등
        - meta: battle_tag, turn, task 타입 등 로깅용 메타데이터

        반환:
        - JSON 파싱된 dict (실패 시 {}).
        """
        use_temp = self.temperature if temperature is None else temperature

        try:
            resp = self.client.responses.create(
                model=self.backend_model,
                input=messages,
                temperature=use_temp,
                max_output_tokens=max_output_tokens,
            )
        except Exception as e:
            # 호출 실패 로그
            self._log_gpt_call(
                success=False,
                meta=meta,
                messages=messages,
                raw_text=None,
                parsed=None,
                parsed_ok=False,
                usage=None,
                error=str(e),
            )
            return {}

        # 텍스트 추출
        raw_text = ""
        try:
            raw_text = resp.output[0].content[0].text.strip()
        except Exception:
            raw_text = repr(getattr(resp, "output", None))

        # usage 추출 및 누적
        usage = getattr(resp, "usage", None)
        try:
            if usage:
                in_t = getattr(usage, "input_tokens", None)
                out_t = getattr(usage, "output_tokens", None)
                tot_t = getattr(usage, "total_tokens", None)
                if in_t:
                    self.token_usage["input_tokens"] += in_t
                if out_t:
                    self.token_usage["output_tokens"] += out_t
                if tot_t:
                    self.token_usage["total_tokens"] += tot_t
        except Exception:
            usage = None

        # JSON 파싱
        parsed_ok = False
        try:
            parsed = json.loads(raw_text)
            parsed_ok = True
        except json.JSONDecodeError:
            parsed = None

        # 호출 로그 남기기
        self._log_gpt_call(
            success=True,
            meta=meta,
            messages=messages,
            raw_text=raw_text,
            parsed=parsed,
            parsed_ok=parsed_ok,
            usage=usage,
            error=None,
        )

        return parsed or {}

    def _log_gpt_call(
        self,
        *,
        success: bool,
        meta: Optional[Dict[str, Any]],
        messages: List[Dict[str, str]],
        raw_text: Optional[str],
        parsed: Optional[Dict[str, Any]],
        parsed_ok: Optional[bool],
        usage: Any,
        error: Optional[str],
    ) -> None:
        """
        GPT 호출 1회에 대한 JSONL 로그.
        """
        if not self.log_dir:
            return

        try:
            os.makedirs(self.log_dir, exist_ok=True)
            date_str = datetime.utcnow().strftime("%Y%m%d")
            path = os.path.join(self.log_dir, f"gpt_calls_{date_str}.jsonl")

            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "model": self.backend_model,
                "success": success,
                "meta": meta or {},
                "request": messages,
                "response_raw": raw_text,
                "parsed_ok": parsed_ok if parsed_ok is not None else parsed is not None,
                "usage": {
                    "input_tokens": getattr(usage, "input_tokens", None) if usage else None,
                    "output_tokens": getattr(usage, "output_tokens", None) if usage else None,
                    "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
                },
                "error": error,
            }

            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        except Exception:
            logging.getLogger(__name__).exception("Failed to write GPT call log.")

    # ------------------------------------------------------------------
    # 전략 결정 메서드들
    # ------------------------------------------------------------------
    def choose_team(
        self,
        state: BattleState,
        battle: AbstractBattle,
        my_team_data: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        GPT에게 팀프리뷰 순서 문자열을 요청한다.
        Function calling으로 상대 팀 분석 함수를 호출할 수 있게 함.
        """
        # 내 팀 정보 구성 (호출자가 제공한 데이터 우선, 없으면 battle에서 추출)
        if not my_team_data:
            my_team_data = self._build_team_data_from_battle(battle)
        
        # 상대 팀 정보 구성
        opp_team_data = []
        opp_species_list = []
        if hasattr(battle, 'opponent_team') and battle.opponent_team:
            for species, pokemon in battle.opponent_team.items():
                opp_team_data.append({
                    "species": pokemon.species,
                    "types": [t.name for t in pokemon.types if t],
                })
                opp_species_list.append(pokemon.species)
        
        prompt_content = f"""You are a Pokemon VGC coach selecting which 4 Pokemon to bring and their lead order.

MY TEAM (6 Pokemon):
{json.dumps(my_team_data, indent=2, ensure_ascii=False)}

OPPONENT TEAM (6 Pokemon):
{json.dumps(opp_team_data, indent=2, ensure_ascii=False)}

TASK: Select 4 Pokemon from my team and choose the order (lead 2, then 2 in back).
- Consider type matchups, speed control, and win conditions
- Lead with Pokemon that can pressure opponent's likely leads
- Keep good switch-ins in the back

OUTPUT FORMAT (JSON):
{{
    "reasoning": "Why these Pokemon and this order? (brief explanation)",
    "my_strategy": "Overall game plan (e.g., speed control + offensive pressure)",
    "opponent_strategy": "Expected opponent strategy (e.g., weather team, TR team)",
    "selection": "1234"
}}

Example: "1234" means bring Pokemon 1,2,3,4 with 1+2 as leads.

Respond in JSON format ONLY:"""

        # Tools for team preview analysis
        tools = [
            {
                "type": "function",
                "name": "analyze_opponent_team",
                "description": "Analyze opponent team to identify likely strategies and roles",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "team": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of opponent Pokemon species"
                        }
                    },
                    "required": ["team"]
                }
            }
        ]

        messages = [{"role": "user", "content": prompt_content}]
        
        # Loop for function calling
        max_iterations = 5
        iteration = 0
        final_result = {}
        
        while iteration < max_iterations:
            iteration += 1
            
            try:
                resp = self.client.responses.create(
                    model=self.backend_model,
                    input=messages,
                    tools=tools,
                    temperature=0.5,
                    max_output_tokens=1024,
                )
            except Exception as e:
                logging.getLogger(__name__).warning(f"[{state.battle_tag}] Function calling failed: {e}")
                break
            
            if not resp or not resp.output:
                break
            
            # resp.output is directly a list of block items (not nested)
            # Each item has .type, and depending on type, different attributes:
            # - type='function_call': has .name, .arguments, .call_id
            # - type='text': has .text
            function_called = False
            
            for block in resp.output:
                # Check for function call
                if hasattr(block, 'type') and block.type == 'function_call':
                    function_called = True
                    func_name = block.name if hasattr(block, 'name') else ""
                    func_args = block.arguments if hasattr(block, 'arguments') else {}
                    
                    logging.getLogger(__name__).info(f"[{state.battle_tag}] GPT called function: {func_name} with args: {func_args}")
                    
                    # Execute function
                    try:
                        func_result = self._execute_team_preview_function(func_name, func_args)
                        logging.getLogger(__name__).info(f"[{state.battle_tag}] Function result: {func_result}")
                    except Exception as e:
                        logging.getLogger(__name__).warning(f"[{state.battle_tag}] Function execution failed: {e}")
                        func_result = {"error": str(e)}
                    
                    # Add to message history for next iteration
                    messages.append({
                        "role": "assistant",
                        "content": json.dumps({
                            "type": "function_call",
                            "name": func_name,
                            "arguments": func_args
                        })
                    })
                    messages.append({
                        "role": "user",
                        "content": json.dumps(func_result)
                    })
                
                # Check for text response with JSON
                elif hasattr(block, 'type') and block.type == 'text':
                    raw_text = block.text.strip() if hasattr(block, 'text') else ""
                    if raw_text:
                        try:
                            final_result = json.loads(raw_text)
                            logging.getLogger(__name__).info(f"[{state.battle_tag}] GPT final response: {final_result}")
                            # Success - break loop
                            iteration = max_iterations
                        except json.JSONDecodeError:
                            logging.getLogger(__name__).debug(f"[{state.battle_tag}] Response not JSON, continuing loop")
                            pass
            
            # If no function was called and no JSON found, break to avoid infinite loop
            if not function_called and not final_result:
                break
        
        selection = final_result.get("selection", "")
        order = "".join(ch for ch in selection if ch.isdigit())
        
        if len(order) != 4:
            self.logger.warning(f"[{state.battle_tag}] Invalid team selection: '{order}'")
            return ""
        
        return order

    def _execute_team_preview_function(self, func_name: str, func_args: Dict[str, Any]) -> Dict[str, Any]:
        """팀 프리뷰 단계 함수 실행."""
        if func_name == "analyze_opponent_team":
            team = func_args.get("team", [])
            return self._analyze_opponent_team_impl(team)
        else:
            return {"error": f"Unknown function: {func_name}"}
    
    def _analyze_opponent_team_impl(self, team_species: List[str]) -> Dict[str, Any]:
        """상대 팀 분석: 스탯/타입으로 전략 추론."""
        from src.data.gen_data import GenData
        
        try:
            gen_data = GenData.from_gen(9)
        except Exception:
            gen_data = None
        
        analysis = {
            "team_roles": [],
            "type_coverage": {},
            "strategy_hints": [],
        }
        
        for species in team_species:
            species_lower = species.lower()
            
            if gen_data:
                dex_entry = gen_data.pokedex.get(species_lower, {})
                base_stats = dex_entry.get("baseStats", {})
                types = dex_entry.get("types", [])
            else:
                base_stats = {}
                types = []
            
            atk = base_stats.get("atk", 0)
            spa = base_stats.get("spa", 0)
            hp = base_stats.get("hp", 0)
            def_ = base_stats.get("def", 0)
            spd = base_stats.get("spd", 0)
            spe = base_stats.get("spe", 0)
            
            # 역할 추론
            role = "utility"
            if atk > spa and atk >= 100:
                role = "physical_attacker"
            elif spa > atk and spa >= 100:
                role = "special_attacker"
            elif hp >= 100 and (def_ >= 100 or spd >= 100):
                role = "tank"
            
            if spe >= 110:
                role = f"{role}_fast"
            
            analysis["team_roles"].append({
                "species": species,
                "types": types,
                "role": role,
                "base_atk": atk,
                "base_spa": spa,
                "base_hp": hp,
                "base_def": def_,
                "base_spd": spd,
                "base_spe": spe,
            })
            
            # 타입 커버리지
            for t in types:
                analysis["type_coverage"][t] = analysis["type_coverage"].get(t, 0) + 1
        
        # 팀 전체 전략 힌트
        roles_str = ",".join(r["role"] for r in analysis["team_roles"])
        if "physical_attacker" in roles_str and "special_attacker" in roles_str:
            analysis["strategy_hints"].append("Mixed offensive team")
        if "tank" in roles_str:
            analysis["strategy_hints"].append("Has defensive backbone")
        if any("fast" in r["role"] for r in analysis["team_roles"]):
            analysis["strategy_hints"].append("Likely fast-paced team")
        
        return analysis

    def _build_team_data_from_battle(self, battle: AbstractBattle) -> List[Dict[str, Any]]:
        data: List[Dict[str, Any]] = []
        if hasattr(battle, "team") and battle.team:
            for i, (_species, pokemon) in enumerate(battle.team.items(), start=1):
                moves = [m.id for m in pokemon.moves.values()] if pokemon.moves else []
                data.append({
                    "index": i,
                    "species": pokemon.species,
                    "types": [t.name for t in pokemon.types if t],
                    "ability": pokemon.ability,
                    "item": pokemon.item,
                    "moves": moves,
                })
        return data

    def choose_move(
        self,
        state: BattleState,
        battle: DoubleBattle,
        legal_actions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        GPT에게 현재 턴 액션 선택을 요청한다.
        
        use_strategy_engine=True이면:
            - function calling + parallel strategies (fast/normal/deep) 사용
            - 정보 수집 → 최종 의사결정 파이프라인
        
        use_strategy_engine=False이면:
            - 기존 단일 JSON 호출 방식 (fallback)
        
        반환값은 slot별로 액션을 표현하는 JSON dict.
        실제 BattleOrder로 바꾸는 건 Player에서 처리.
        """
        # 1. team preview 캐시를 전략 엔진과 공유
        if self.strategy_engine:
            self.strategy_engine.set_team_preview_cache(
                getattr(state, 'team_preview_cache', None)
            )
        
        # 2. battle_context 구성 (기존 pochamps_player와 동일한 포맷)
        showdown_view = {
            "turn": battle.turn,
            "active": {
                "me": [
                    mon.species if mon else None for mon in battle.active_pokemon
                ],
                "opponent": [
                    mon.species if mon else None for mon in battle.opponent_active_pokemon
                ],
            },
            "weather": str(battle.weather) if battle.weather else None,
            "fields": [str(f) for f in (battle.fields or [])],
        }
        
        battle_context = {
            "battle_tag": state.battle_tag,
            "turn": state.turn,
            "format": state.battle_format,
            "battle_state": state.to_gpt_dict(),
            "showdown": showdown_view,
            "legal_actions": legal_actions,
        }
        
        # 3. StrategyEngine 사용 여부에 따라 분기
        if self.use_strategy_engine and self.strategy_engine:
            return self._choose_move_with_strategies(battle_context, state)
        else:
            return self._choose_move_simple(battle_context, state)
    
    def _choose_move_with_strategies(
        self,
        battle_context: Dict,
        state: BattleState,
    ) -> Dict[str, Any]:
        """
        function calling + parallel strategies를 사용하여 move 선택.
        
        예전 pochamps_player의 파이프라인:
        1. fast/normal/deep 전략 병렬 실행 (function calling으로 정보 수집)
        2. 결과 합성 또는 best 선택
        3. JSON 포맷 변환 (slot1/slot2 → slot_1/slot_2)
        """
        from src.player.functions import BATTLE_TOOLS
        
        # 1. 병렬 전략 실행
        responses = self.strategy_engine.run_strategies(
            battle_tag=state.battle_tag,
            battle_context=battle_context,
            tools=BATTLE_TOOLS,
        )
        
        # 2. 최선의 응답 선택/합성
        best = self.strategy_engine.select_best_response(
            responses=responses,
            battle_context=battle_context,
        )
        
        if not best:
            logging.getLogger(__name__).warning(
                f"[{state.battle_tag}] No valid strategy response, using fallback"
            )
            return {}
        
        # 3. 포맷 변환: 예전 구조 (slot1/slot2) → 새 구조 (slot_1/slot_2)
        raw = best.get("content", {})
        
        result = {
            "reasoning": raw.get("reasoning", ""),
            "slot_1": self._convert_slot_format(raw.get("slot1", {})),
            "slot_2": self._convert_slot_format(raw.get("slot2", {})),
            "terastallize": raw.get("terastallize", False),
            "forfeit": raw.get("forfeit", False),
        }
        
        return result
    
    def _convert_slot_format(self, old_slot: Dict) -> Dict:
        """
        예전 포맷 {"action": "move", "move": "move_name", "target": 1}
        → 새 포맷 {"action": "move", "move_id": "move_name", "target": "opponent_1"}
        """
        if not old_slot:
            return {"action": "move", "move_id": "", "target": "auto"}
        
        action = old_slot.get("action", "move")
        move = old_slot.get("move", "")
        target = old_slot.get("target", "auto")
        
        # target 정수 → 문자열 변환
        if isinstance(target, int):
            target_map = {
                1: "opponent_1",
                2: "opponent_2",
                -1: "ally",
                0: "auto",
            }
            target = target_map.get(target, "auto")
        
        return {
            "action": action,
            "move_id": move,
            "target": target,
        }
    
    def _choose_move_simple(
        self,
        battle_context: Dict,
        state: BattleState,
    ) -> Dict[str, Any]:
        """
        단일 JSON 호출 방식 (기존 fallback).
        function calling 없이 바로 JSON 응답만 받음.
        """
        payload = {
            "task": "choose_move",
            "battle_state": battle_context["battle_state"],
            "showdown": battle_context["showdown"],
            "legal_actions": battle_context["legal_actions"],
        }

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a competitive Pokemon VGC coach. "
                    "Given the observation JSON (battle_state, showdown, legal_actions), "
                    "choose strong, safe plays.\n\n"
                    "Respond ONLY in JSON with the following format:\n"
                    "{\n"
                    '  "reasoning": "Brief explanation of your strategy",\n'
                    '  "slot_1": {"action": "move", "move_id": "<move_id>", "target": "opponent_1|opponent_2|ally|auto"},\n'
                    '  "slot_2": {"action": "move", "move_id": "<move_id>", "target": "opponent_1|opponent_2|ally|auto"},\n'
                    '  "terastallize": false,\n'
                    '  "forfeit": false\n'
                    "}\n\n"
                    "Target options:\n"
                    "- opponent_1: Left opponent slot\n"
                    "- opponent_2: Right opponent slot\n"
                    "- ally: Your other active Pokemon\n"
                    "- auto: Automatic target selection\n"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ]

        result = self._call_gpt_json(
            messages=messages,
            max_output_tokens=256,
            meta={
                "task": "choose_move",
                "battle_tag": state.battle_tag,
                "turn": state.turn,
            },
        )

        return result


# =====================================================================
# 3. 메인 Player 구현
# =====================================================================

class PochampsPlayer(Player):
    """
    새로 정리한 버전의 플레이어 클래스.

    책임:
    - 배틀마다 BattleState를 하나씩 유지
    - 가능한 한 모든 “계산 가능한 추론”은 BattleState 계층에서 처리
    - GPTStrategyClient는
        * teampreview,
        * 매 턴 choose_move
      두 가지만 담당
    """

    def __init__(
        self,
        battle_format: str = "gen9vgc2025regi",
        backend_model: str = "gpt-5.1",
        api_key: str = "",
        base_url: Optional[str] = None,
        fast_model: Optional[str] = None,
        deep_model: Optional[str] = None,
        temperature: float = 0.7,
        log_dir: Optional[str] = None,
        use_strategy_engine: bool = True,
        global_timeout: float = 25.0,
        **kwargs: Any,
    ) -> None:
        """
        :param battle_format: Showdown format ID.
        :param backend_model: teampreview/턴결정에 사용할 모델.
        :param api_key: OpenAI (또는 호환) API 키.
        :param base_url: self-hosted / proxy endpoint 있을 경우.
        :param fast_model: 빠른 전략에 사용할 모델 (None이면 backend_model).
        :param deep_model: 깊은 전략에 사용할 모델 (None이면 backend_model).
        :param temperature: GPT 샘플링 temperature.
        :param log_dir: 턴별 로그 저장 디렉토리 (JSONL).
        :param use_strategy_engine: True면 function calling + parallel strategies 사용.
        :param global_timeout: 병렬 전략 실행 타임아웃.
        :param kwargs: Player.__init__ 에 그대로 전달.
        """
        # Filter out parameters that are not valid for Player.__init__
        # main.py might pass backend, fast_model, deep_model, etc.
        invalid_keys = {'backend', 'fast_model', 'deep_model', 'save_replays', 'open_team_sheets', 'debug_mode'}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in invalid_keys}
        
        super().__init__(battle_format=battle_format, **filtered_kwargs)

        # Note: self.logger is inherited from Player as a property (returns ps_client.logger)
        self.battle_format = battle_format
        self.log_dir = log_dir

        self.strategy_client = GPTStrategyClient(
            backend_model=backend_model,
            api_key=api_key,
            base_url=base_url,
            fast_model=fast_model,
            deep_model=deep_model,
            temperature=temperature,
            log_dir=log_dir,
            use_strategy_engine=use_strategy_engine,
            global_timeout=global_timeout,
        )

        # battle_tag / battle_id 기준으로 BattleState 관리
        self._states: Dict[str, BattleState] = {}

        # main.py에서 update_team으로 전달된 로컬 팀 원본 문자열을 캐싱 (battle.team 비어 있을 때 사용)
        self._local_team_raw: Optional[str] = None

    # -----------------------------------------------------------------
    # 내부 헬퍼
    # -----------------------------------------------------------------

    def _battle_key(self, battle: AbstractBattle) -> str:
        # poke-env는 보통 battle.battle_tag 또는 battle.battle_id를 제공
        return getattr(battle, "battle_tag", None) or getattr(battle, "battle_id", str(id(battle)))

    def _get_state(self, battle: AbstractBattle) -> BattleState:
        key = self._battle_key(battle)
        if key not in self._states:
            self._states[key] = BattleState(
                battle_tag=key,
                battle_format=self.battle_format,
            )
        return self._states[key]

    # Player.update_team은 _team만 설정하므로, 로컬 캐시도 함께 저장해 team preview에서 사용한다.
    def update_team(self, team: Any):
        """
        main.py에서 나중에 update_team(team2)로 넣어주는 팀 정보를
        최대한 원본 형태 그대로 캐싱해서,
        battle.team이 비어 있을 때 team preview에 활용한다.
        """
        raw: Optional[str] = None

        # 1) 문자열이면 그대로 packed 팀 문자열로 간주
        if isinstance(team, str):
            raw = team

        # 2) Teambuilder/객체인 경우 yield_team()이 있으면 이용
        elif hasattr(team, "yield_team"):
            try:
                packed = team.yield_team()
                if isinstance(packed, str):
                    raw = packed
            except Exception:
                raw = None

        # 3) 그래도 못 얻었으면 마지막 수단 (디버깅용)
        if raw is None:
            try:
                raw = str(team)
            except Exception:
                raw = None

        self._local_team_raw = raw
        # poke-env Player 쪽에도 그대로 전달해서 self._team 설정
        super().update_team(team)


    def _sync_from_battle(self, battle: DoubleBattle, state: BattleState) -> None:
        """
        poke-env battle 객체에서 BattleState로 동기화.

        이 함수에 가능한 한 많은 “결정론/확률 기반 추론”을 몰아넣으면 됨.
        지금은 HP/상태/필드 정도만 동기화하는 최소 버전.
        """
        state.turn = battle.turn

        # 우리 쪽 active
        for i, mon in enumerate(battle.active_pokemon, start=1):
            if mon is None:
                continue
            p_state = state.my_team.get_or_create(mon.species)
            p_state.slot = i
            p_state.hp_percent = round(mon.current_hp_fraction * 100, 1) if mon.current_hp_fraction else 0
            p_state.status = str(mon.status) if mon.status else None
            p_state.fainted = bool(mon.fainted)
            p_state.terastallized = bool(getattr(mon, "terastallized", False))
            
            # 사용한 기술 기록
            if mon.moves:
                for move_id in mon.moves.keys():
                    p_state.record_move(move_id)

        # 상대 active
        for i, mon in enumerate(battle.opponent_active_pokemon, start=1):
            if mon is None:
                continue
            p_state = state.opp_team.get_or_create(mon.species)
            p_state.slot = i
            p_state.hp_percent = round(mon.current_hp_fraction * 100, 1) if mon.current_hp_fraction else 0
            p_state.status = str(mon.status) if mon.status else None
            p_state.fainted = bool(mon.fainted)
            p_state.terastallized = bool(getattr(mon, "terastallized", False))
            
            # 관측된 정보
            if mon.ability:
                p_state.ability = mon.ability
            if mon.item:
                p_state.item = mon.item
            if mon.moves:
                for move_id in mon.moves.keys():
                    p_state.record_move(move_id)

        # 필드 정보
        state.weather = str(battle.weather) if battle.weather else None
        # terrain은 별도 속성으로, fields는 전체 필드 효과
        state.terrain = (
            str(getattr(battle, "terrain", None)) if getattr(battle, "terrain", None) else None
        )
        # Trick Room 여부
        state.trick_room = any("trickroom" in str(f).lower() for f in (battle.fields or []))

    def _build_legal_actions(self, battle: DoubleBattle) -> Dict[str, Any]:
        """
        GPT가 볼 수 있는 legal action 뷰를 만든다.

        나중에 여기서 각 기술의 대략적인 기대값/확률 등을 같이 넣어줄 수 있음.
        """
        actions: Dict[str, Any] = {}

        # Force switch 여부
        force_switch = getattr(battle, 'force_switch', [False, False])
        if not isinstance(force_switch, list):
            force_switch = [False, False]
        actions["force_switch"] = force_switch

        # 각 슬롯별 가능한 기술
        moves_per_slot: List[List[Dict[str, Any]]] = []
        for i, mon in enumerate(battle.active_pokemon):
            slot_moves: List[Dict[str, Any]] = []
            if mon is None or mon.fainted:
                moves_per_slot.append(slot_moves)
                continue

            # available_moves from battle
            if i < len(battle.available_moves):
                for move in battle.available_moves[i]:
                    slot_moves.append({
                        "id": move.id,
                        "base_power": move.base_power,
                        "type": move.type.name if move.type else None,
                        "category": move.category.name if hasattr(move, 'category') and move.category else None,
                        "priority": move.priority,
                    })
            moves_per_slot.append(slot_moves)

        actions["moves"] = moves_per_slot
        
        # 스위치 가능한 포켓몬
        available_switches = []
        if hasattr(battle, 'available_switches') and battle.available_switches:
            for slot_switches in battle.available_switches:
                slot_list = []
                if slot_switches:
                    for p in slot_switches:
                        slot_list.append({
                            "species": p.species,
                            "hp_percent": round(p.current_hp_fraction * 100, 1)
                        })
                available_switches.append(slot_list)
        actions["available_switches"] = available_switches
        
        return actions

    # -----------------------------------------------------------------
    # Teampreview
    # -----------------------------------------------------------------

    def teampreview(self, battle: AbstractBattle) -> str:
        """
        teampreview가 있는 포맷에서 리드/벤치 순서를 결정.

        - BattleState를 초기화/획득
        - GPT에 한 번 질의해서 order string을 받음
        - 실패 시 상위 Player.teampreview로 fallback
        """
        state = self._get_state(battle)
        my_team_data = self._resolve_my_team_data(battle)

        if not my_team_data:
            self.logger.info(
                f"[{state.battle_tag}] no team info (battle.team empty and no local roster); using default heuristic order"
            )
            return self._fallback_teampreview(battle)

        # 캐시된 내 팀 정보를 BattleState에도 반영해 이후 턴 컨텍스트에 활용
        self._update_state_with_my_team(state, my_team_data)

        try:
            order = self.strategy_client.choose_team(state, battle, my_team_data=my_team_data)
            if order:
                self.logger.info(f"[{state.battle_tag}] GPT teampreview order: {order}")
                return order
        except Exception as exc:
            self.logger.warning(
                f"[{state.battle_tag}] GPT teampreview failed: {exc}", exc_info=True
            )

        # fallback – 기본 heuristic
        return self._fallback_teampreview(battle)

    # -----------------------------------------------------------------
    # 턴별 액션 선택
    # -----------------------------------------------------------------

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """
        poke-env가 매 턴 호출하는 엔트리 포인트.

        예외 케이스 우선 처리:
        1. Force switch (기절 등으로 교체 필수)
        2. Trapped (교체 불가)
        3. 사용 가능한 행동이 없을 때
        
        그 외: GPT 질의 → BattleOrder 변환
        """
        state = self._get_state(battle)

        if not isinstance(battle, DoubleBattle):
            # 싱글/기타 포맷은 우선 랜덤/단순 heuristic으로 처리
            return self.choose_random_move(battle)

        self._sync_from_battle(battle, state)
        
        # =====================================================================
        # CASE 1: Force Switch - 포켓몬이 기절하거나 강제 교체 상황
        # =====================================================================
        force_switch = getattr(battle, 'force_switch', [False, False])
        if not isinstance(force_switch, list):
            force_switch = [False, False]
        
        if any(force_switch):
            return self._handle_force_switch(battle, force_switch)
        
        # =====================================================================
        # CASE 2: Trapped - 특정 포켓몬이 교체 불가 상태 (예: Shadow Tag)
        # =====================================================================
        if self._has_trapped_pokemon(battle):
            return self._handle_trapped_pokemon(battle)
        
        # =====================================================================
        # CASE 3: 사용 가능한 행동이 없을 때 (극히 드문 경우)
        # =====================================================================
        if not self._has_available_actions(battle):
            self.logger.warning(f"[{state.battle_tag}] No available actions! Using default order.")
            return DefaultBattleOrder()
        
        # =====================================================================
        # NORMAL CASE: GPT 질의
        # =====================================================================
        legal_actions = self._build_legal_actions(battle)
        
        try:
            decision = self.strategy_client.choose_move(state, battle, legal_actions)
            order = self._parse_decision_to_order(battle, decision)
            
            if order is None:
                self.logger.warning(
                    f"[{state.battle_tag}] Failed to parse GPT decision, using fallback. decision={decision}"
                )
                return self._fallback_random(battle)
            
            if self.log_dir:
                self._log_turn(state, battle, legal_actions, decision)
            
            return order
            
        except Exception as exc:
            self.logger.warning(
                f"[{state.battle_tag}] GPT choose_move failed: {exc}", exc_info=True
            )
            return self._fallback_random(battle)

    # -----------------------------------------------------------------
    # Exception case handlers
    # -----------------------------------------------------------------

    def _handle_force_switch(
        self, battle: DoubleBattle, force_switch: List[bool]
    ) -> BattleOrder:
        """
        강제 교체 처리 (포켓몬 기절 등).
        GPT에게 어떤 포켓몬으로 교체할지 질의.
        
        force_switch = [True, False] 같은 형태로, 각 슬롯별로 교체가 필요한지 표시.
        """
        from src.player.battle_order import DoubleBattleOrder
        
        state = self._get_state(battle)
        
        # ====================================================================
        # 1. 현재 상황 정리 - 어떤 슬롯에 교체가 필요한지 명확히 파악
        # ====================================================================
        switch_situation = []
        for slot_idx in range(2):  # Doubles = 2 slots
            slot_number = slot_idx + 1  # 1-indexed for display
            needs_switch = force_switch[slot_idx] if slot_idx < len(force_switch) else False
            
            current_mon = None
            if slot_idx < len(battle.active_pokemon):
                current_mon = battle.active_pokemon[slot_idx]
            
            available_switches_for_this_slot = []
            if slot_idx < len(battle.available_switches):
                available_switches_for_this_slot = battle.available_switches[slot_idx]
            
            switch_situation.append({
                "slot": slot_number,
                "needs_switch": needs_switch,
                "current_pokemon": current_mon.species if current_mon else None,
                "fainted": current_mon.fainted if current_mon else True,
                "available_switches": [
                    {
                        "species": p.species,
                        "hp_percent": round(p.current_hp_fraction * 100, 1),
                        "types": [t.name for t in p.types if t],
                    }
                    for p in available_switches_for_this_slot
                ]
            })
        
        # ====================================================================
        # 2. GPT에게 질의 - 어떤 포켓몬으로 교체할지 결정
        # ====================================================================
        prompt = f"""FORCE SWITCH SITUATION:

{json.dumps(switch_situation, indent=2, ensure_ascii=False)}

OPPONENT ACTIVE:
{json.dumps([
    {{
        "slot": i+1,
        "species": mon.species,
        "hp_percent": round(mon.current_hp_fraction * 100, 1),
        "types": [t.name for t in mon.types if t]
    }}
    for i, mon in enumerate(battle.opponent_active_pokemon) if mon
], indent=2, ensure_ascii=False)}

FIELD:
- Weather: {battle.weather}
- Terrain: {battle.fields}

TASK: Choose which Pokemon to switch in for each slot that needs_switch=true.
- Consider type matchups, HP, and current field conditions
- ONLY choose from the available_switches list for each slot
- Do NOT choose Pokemon that are already active

OUTPUT FORMAT (JSON):
{{
    "reasoning": "Brief explanation of switch choices",
    "slot_1": {{"action": "switch", "pokemon": "species_name"}} or {{"action": "move", "move": "move_id"}},
    "slot_2": {{"action": "switch", "pokemon": "species_name"}} or {{"action": "move", "move": "move_id"}}
}}

If a slot doesn't need switching, you can use "move" action (will be handled automatically).
Respond in JSON ONLY:"""

        messages = [{"role": "user", "content": prompt}]
        
        decision = self.strategy_client._call_gpt_json(
            messages=messages,
            max_output_tokens=256,
            temperature=0.3,  # Low temp for consistent decisions
            meta={
                "task": "force_switch",
                "battle_tag": state.battle_tag,
                "turn": battle.turn,
            },
        )
        
        if not decision:
            self.logger.warning(f"GPT force switch failed, using fallback")
            decision = None
        
        # ====================================================================
        # 3. GPT 결정을 BattleOrder로 변환
        # ====================================================================
        orders = []
        
        for slot_idx in range(2):
            slot_key = f"slot_{slot_idx + 1}"
            needs_switch = force_switch[slot_idx] if slot_idx < len(force_switch) else False
            
            if needs_switch:
                # 이 슬롯은 반드시 교체해야 함
                available_switches_for_this_slot = []
                if slot_idx < len(battle.available_switches):
                    available_switches_for_this_slot = battle.available_switches[slot_idx]
                
                if not available_switches_for_this_slot:
                    # 교체할 포켓몬이 없으면 default
                    orders.append(DefaultBattleOrder())
                    self.logger.error(f"Slot {slot_idx+1} needs switch but no Pokemon available!")
                    continue
                
                # GPT 결정 파싱
                chosen_pokemon = None
                if decision and slot_key in decision:
                    slot_decision = decision[slot_key]
                    if slot_decision.get("action") == "switch":
                        target_species = slot_decision.get("pokemon", "").lower().replace("-", "").replace(" ", "")
                        
                        # available_switches에서 해당 포켓몬 찾기
                        for switch_option in available_switches_for_this_slot:
                            option_species = switch_option.species.lower().replace("-", "").replace(" ", "")
                            if option_species == target_species:
                                chosen_pokemon = switch_option
                                break
                
                # GPT가 선택 못했거나 잘못 선택한 경우 fallback
                if chosen_pokemon is None:
                    # HP가 가장 높은 포켓몬으로 교체
                    chosen_pokemon = max(available_switches_for_this_slot, 
                                        key=lambda p: p.current_hp_fraction)
                    self.logger.warning(f"Slot {slot_idx+1}: GPT failed, using HP-based fallback -> {chosen_pokemon.species}")
                else:
                    self.logger.info(f"Slot {slot_idx+1}: Switching to {chosen_pokemon.species}")
                
                orders.append(BattleOrder(chosen_pokemon))
                
            else:
                # 이 슬롯은 교체 불필요 - 정상 행동 가능
                current_mon = None
                if slot_idx < len(battle.active_pokemon):
                    current_mon = battle.active_pokemon[slot_idx]
                
                if current_mon and not current_mon.fainted:
                    # 가능한 기술 중 첫 번째 선택 (간단한 fallback)
                    available_moves_for_this_slot = []
                    if slot_idx < len(battle.available_moves):
                        available_moves_for_this_slot = battle.available_moves[slot_idx]
                    
                    if available_moves_for_this_slot:
                        move = available_moves_for_this_slot[0]
                        orders.append(BattleOrder(move))
                        self.logger.info(f"Slot {slot_idx+1}: Using move {move.id}")
                    else:
                        orders.append(DefaultBattleOrder())
                else:
                    orders.append(DefaultBattleOrder())
        
        # ====================================================================
        # 4. 두 슬롯의 order를 합쳐서 반환
        # ====================================================================
        if len(orders) == 2:
            return DoubleBattleOrder(orders[0], orders[1])
        elif len(orders) == 1:
            return orders[0]
        else:
            return DefaultBattleOrder()
    
    def _has_trapped_pokemon(self, battle: DoubleBattle) -> bool:
        """
        교체 불가 상태인 포켓몬이 있는지 확인.
        
        예: Shadow Tag, Arena Trap, Mean Look 등
        """
        # 1) trapped 플래그 체크
        for mon in battle.active_pokemon:
            if mon and not mon.fainted:
                if getattr(mon, 'trapped', False):
                    return True
        
        # 2) 모든 슬롯의 available_switches가 비어 있으면 trap 또는 마지막 포켓몬 상태
        if battle.available_switches:
            all_empty = all((not slot) for slot in battle.available_switches)
            return all_empty
        
        return False
    
    def _handle_trapped_pokemon(self, battle: DoubleBattle) -> BattleOrder:
        """
        Trapped 상태 처리 - 교체 불가이므로 기술만 사용.
        
        간단한 heuristic: 첫 번째 사용 가능한 기술 선택
        """
        from src.player.battle_order import DoubleBattleOrder
        
        orders = []
        
        for slot_idx, mon in enumerate(battle.active_pokemon):
            if mon and not mon.fainted:
                if slot_idx < len(battle.available_moves) and battle.available_moves[slot_idx]:
                    # 첫 번째 기술 사용
                    move = battle.available_moves[slot_idx][0]
                    orders.append(BattleOrder(move))
                else:
                    orders.append(DefaultBattleOrder())
            else:
                orders.append(DefaultBattleOrder())
        
        if len(orders) == 2:
            return DoubleBattleOrder(orders[0], orders[1])
        elif len(orders) == 1:
            return orders[0]
        else:
            return DefaultBattleOrder()
    
    def _has_available_actions(self, battle: DoubleBattle) -> bool:
        """
        사용 가능한 행동이 있는지 확인.
        
        기술이나 교체가 하나라도 가능하면 True.
        """
        # 기술 확인
        if battle.available_moves:
            for slot_moves in battle.available_moves:
                if slot_moves:
                    return True
        
        # 교체 확인
        if battle.available_switches:
            for slot_switches in battle.available_switches:
                if slot_switches:
                    return True
        
        return False

    # -----------------------------------------------------------------
    # Helper methods
    # -----------------------------------------------------------------
    
    def _convert_target_str(self, target_str: str) -> int:
        """
        GPT target string → poke-env target 인덱스.
        
        DoubleBattle target positions:
        - -2, -1: 자신의 포켓몬들
        - 1, 2: 상대 포켓몬들
        - EMPTY_TARGET_POSITION: 자동 타겟팅
        """
        if not target_str or target_str == "auto":
            return DoubleBattle.EMPTY_TARGET_POSITION
        
        target_str = target_str.lower().strip()
        
        if target_str == "opponent_1":
            return 1
        elif target_str == "opponent_2":
            return 2
        elif target_str == "ally":
            return -1
        else:
            # 알 수 없는 target → auto
            return DoubleBattle.EMPTY_TARGET_POSITION
    
    def _fallback_random(self, battle: DoubleBattle) -> BattleOrder:
        """
        GPT 실패 시 사용할 랜덤 행동 fallback.
        
        Player 기본 제공 choose_random_move를 사용하되,
        프로젝트에 더 나은 랜덤 전략이 있으면 그걸 사용.
        """
        # choose_random_move는 Player 기본 메서드로 제공됨
        return self.choose_random_move(battle)

    def _fallback_teampreview(self, battle: AbstractBattle) -> str:
        """teampreview 기본 순서 생성 (battle.team 비어도 조용히 처리)."""
        team_size = len(getattr(battle, "team", []) or [])
        if team_size <= 0:
            team_size = 6  # showdown defaults

        members = list(range(1, team_size + 1))
        random.shuffle(members)

        if getattr(battle, "max_team_size", None):
            members = members[: battle.max_team_size]

        return "/team " + "".join(str(c) for c in members)

    def _build_team_data_from_battle(self, battle: AbstractBattle) -> List[Dict[str, Any]]:
        data: List[Dict[str, Any]] = []
        if hasattr(battle, "team") and battle.team:
            for i, (_species, pokemon) in enumerate(battle.team.items(), start=1):
                moves = [m.id for m in pokemon.moves.values()] if pokemon.moves else []
                data.append({
                    "index": i,
                    "species": pokemon.species,
                    "types": [t.name for t in pokemon.types if t],
                    "ability": pokemon.ability,
                    "item": pokemon.item,
                    "moves": moves,
                })
        return data

    def _resolve_my_team_data(self, battle: AbstractBattle) -> List[Dict[str, Any]]:
        """Prefer battle.team; fall back to the locally cached teambuilder roster."""
        self.logger.info(f"[{self._battle_key(battle)}] _resolve_my_team_data: checking battle.team")
        data = self._build_team_data_from_battle(battle)
        if data:
            self.logger.info(f"[{self._battle_key(battle)}] _resolve_my_team_data: got {len(data)} mons from battle.team")
            return data
        self.logger.info(f"[{self._battle_key(battle)}] _resolve_my_team_data: battle.team empty, trying teambuilder roster")
        roster = self._get_teambuilder_roster()
        if roster:
            self.logger.info(f"[{self._battle_key(battle)}] _resolve_my_team_data: got {len(roster)} mons from teambuilder roster")
        else:
            self.logger.warning(f"[{self._battle_key(battle)}] _resolve_my_team_data: teambuilder roster also empty!")
        return roster

    def _update_state_with_my_team(
        self, state: BattleState, roster: List[Dict[str, Any]]
    ) -> None:
        """Hydrate BattleState.my_team with cached roster details."""
        for mon in roster:
            species = mon.get("species")
            if not species:
                continue

            p_state = state.my_team.get_or_create(species)
            if mon.get("item"):
                p_state.item = mon.get("item")
            if mon.get("ability"):
                p_state.ability = mon.get("ability")
            for mv in mon.get("moves") or []:
                p_state.record_move(mv)

    def _get_teambuilder_roster(self) -> List[Dict[str, Any]]:
        """현재 플레이어가 알고 있는 팀 정보를 최대한 활용해서 roster로 변환.

        우선순위:
        1) self._team 이 문자열인 경우 → 그대로 packed 팀 문자열로 사용
        2) self._team 에 yield_team() 이 있는 경우 → yield_team()
        3) _local_team_raw (update_team 에서 캐싱) 사용
        """
        packed: Optional[str] = None

        tb = getattr(self, "_team", None)
        self.logger.info(
            f"[teambuilder] self._team = {tb} "
            f"(type: {type(tb).__name__ if tb is not None else 'None'})"
        )

        # 1) self._team 이 이미 packed 문자열인 경우
        if isinstance(tb, str) and tb.strip():
            packed = tb
            self.logger.info(
                f"[teambuilder] using packed team directly from self._team "
                f"(len={len(packed)})"
            )

        # 2) yield_team() 이 있는 객체인 경우
        elif tb is not None and hasattr(tb, "yield_team"):
            try:
                candidate = tb.yield_team()
                if isinstance(candidate, str) and candidate.strip():
                    packed = candidate
                    self.logger.info(
                        f"[teambuilder] got packed from self._team.yield_team() "
                        f"(len={len(packed)})"
                    )
            except Exception as e:
                self.logger.warning(
                    f"[teambuilder] self._team.yield_team() failed: {e}"
                )

        # 3) 그래도 없으면 update_team 에서 저장해둔 _local_team_raw 사용
        if not packed and self._local_team_raw:
            if isinstance(self._local_team_raw, str) and self._local_team_raw.strip():
                packed = self._local_team_raw
                self.logger.info(
                    f"[teambuilder] using packed team from _local_team_raw "
                    f"(len={len(packed)})"
                )

        if not packed:
            self.logger.warning(
                "[teambuilder] no packed team available "
                "(self._team and _local_team_raw both empty or invalid)"
            )
            return []

        self.logger.info(f"[teambuilder] packed content (first 300 chars): {repr(packed[:300])}")
        roster = self._parse_packed_team(packed)
        self.logger.info(f"[teambuilder] parsed {len(roster)} mons from packed team")
        if not roster:
            self.logger.warning(f"[teambuilder] EMPTY ROSTER! packed={repr(packed)}")
        return roster

    def _parse_packed_team(self, packed: str) -> List[Dict[str, Any]]:
        roster: List[Dict[str, Any]] = []
        try:
            self.logger.info(f"[parse_packed] input: {repr(packed[:150])}")
            mons = packed.split("]")
            self.logger.info(f"[parse_packed] split by ']' gives {len(mons)} parts")
            for i, mon in enumerate(mons, start=1):
                if not mon:
                    self.logger.debug(f"[parse_packed] part {i} is empty, skipping")
                    continue
                parts = mon.split("|")
                self.logger.debug(f"[parse_packed] part {i}: split by '|' gives {len(parts)} fields, parts={parts[:6]}")
                
                # packed format: species|nickname|item|ability|moves|nature|evs|ivs|gender|shiny|level|happiness|pokeball|teratype
                # parts[0] = species
                # parts[1] = nickname (usually empty)
                # parts[2] = item
                # parts[3] = ability
                # parts[4] = moves (comma-separated)
                # parts[5] = nature
                # parts[6] = evs (comma-separated: hp,atk,def,spa,spd,spe)
                # parts[7] = ivs (comma-separated)
                # parts[8] = gender
                # parts[9] = shiny
                # parts[10] = level
                # parts[11] = happiness
                # parts[12] = pokeball
                # parts[13] = teratype
                
                species = parts[0].strip() if parts and parts[0] else ""
                nickname = parts[1].strip() if len(parts) > 1 else ""
                item = parts[2].strip() if len(parts) > 2 else ""
                ability = parts[3].strip() if len(parts) > 3 else ""
                moves_field = parts[4] if len(parts) > 4 else ""
                moves = [m.strip() for m in moves_field.split(",") if m.strip()]
                nature = parts[5].strip() if len(parts) > 5 else ""
                evs_field = parts[6] if len(parts) > 6 else ""
                evs_list = [e.strip() for e in evs_field.split(",")]
                # Parse EVs to dict
                ev_names = ["hp", "atk", "def", "spa", "spd", "spe"]
                evs_dict = {}
                for idx, ev_val in enumerate(evs_list[:6]):
                    if ev_val:
                        try:
                            evs_dict[ev_names[idx]] = int(ev_val)
                        except ValueError:
                            pass
                teratype = parts[13].strip() if len(parts) > 13 else ""
                
                self.logger.debug(
                    f"[parse_packed] part {i}: species={species}, item={item}, ability={ability}, "
                    f"moves={moves}, nature={nature}, evs={evs_dict}, teratype={teratype}"
                )
                if species:
                    roster.append({
                        "index": i,
                        "species": species,
                        "types": [],
                        "ability": ability if ability else None,
                        "item": item if item else None,
                        "moves": moves,
                        "nature": nature if nature else None,
                        "evs": evs_dict if evs_dict else None,
                        "teratype": teratype if teratype else None,
                    })
        except Exception as e:
            self.logger.exception(f"[parse_packed] exception: {e}")
            return []

        return roster

    def _build_team_data_from_teambuilder(self) -> List[Dict[str, Any]]:
        return self._get_teambuilder_roster()
    
    # -----------------------------------------------------------------
    # Decision parsing / logging
    # -----------------------------------------------------------------

    def _parse_decision_to_order(
        self, battle: DoubleBattle, decision: Dict[str, Any]
    ) -> Optional[BattleOrder]:
        """
        GPT JSON 응답을 BattleOrder로 변환.

        기대 JSON 예시:
        {
          "reasoning": "...",
          "slot_1": {"action": "move", "move_id": "earthpower", "target": "opponent_1"},
          "slot_2": {"action": "move", "move_id": "protect", "target": "self"},
          "terastallize": false,
          "forfeit": false
        }
        """
        # 완전 비어있으면 바로 실패
        if not isinstance(decision, dict) or not decision:
            self.logger.warning("GPT decision is empty or not a dict")
            return None

        # 기권 먼저 처리
        if decision.get("forfeit"):
            self.logger.info("GPT requested forfeit")
            return ForfeitBattleOrder()

        # Tera 여부
        tera_flag = bool(decision.get("terastallize", False))
        tera_slot = int(decision.get("terastallize_slot", 1)) if "terastallize_slot" in decision else 1
        # terastallize_slot은 1-indexed (slot_1, slot_2)

        orders_per_slot: List[Optional[BattleOrder]] = []

        # 더블 기준 2 슬롯
        for slot_idx in range(2):
            slot_key = f"slot_{slot_idx + 1}"
            cmd = decision.get(slot_key, {}) or {}

            # 해당 슬롯에 active 포켓몬이 없으면 더미
            if slot_idx >= len(battle.active_pokemon):
                orders_per_slot.append(DefaultBattleOrder())
                continue

            mon = battle.active_pokemon[slot_idx]
            if mon is None or mon.fainted:
                orders_per_slot.append(DefaultBattleOrder())
                continue

            action = (cmd.get("action") or "move").lower()

            # -------------------------------
            # 1) 일반 기술 선택
            # -------------------------------
            if action == "move":
                move_id = (cmd.get("move_id") or "").strip().lower()
                if not move_id:
                    self.logger.warning(f"Slot {slot_idx+1}: missing move_id in GPT decision")
                    return None

                # active 포켓몬의 moves에서 move_id 찾기
                move_obj = None
                for m in mon.moves.values():
                    if m.id.lower() == move_id:
                        move_obj = m
                        break

                if move_obj is None:
                    self.logger.warning(
                        f"Slot {slot_idx+1}: GPT chose unknown move_id '{move_id}' for {mon.species}"
                    )
                    # ---- Fallback: 해당 슬롯의 available_moves 중 첫 번째 기술 사용 ----
                    legal_moves = []
                    if slot_idx < len(battle.available_moves):
                        legal_moves = battle.available_moves[slot_idx] or []

                    if legal_moves:
                        fallback_move = legal_moves[0]
                        self.logger.warning(
                            f"Slot {slot_idx+1}: falling back to first legal move '{fallback_move.id}'"
                        )
                        order = BattleOrder(
                            order=fallback_move,
                            move_target=DoubleBattle.EMPTY_TARGET_POSITION,
                            terastallize=(tera_flag and (tera_slot == (slot_idx + 1))),
                        )
                        orders_per_slot.append(order)
                        continue  # 다음 슬롯 처리
                    else:
                        # 이 슬롯에 쓸 수 있는 기술 자체가 없으면 전체 파싱 실패
                        self.logger.error(
                            f"Slot {slot_idx+1}: no legal moves available for fallback"
                        )
                        return None

                # target 문자열 → poke-env target 인덱스
                target_str = (cmd.get("target") or "auto").lower()
                move_target = self._convert_target_str(target_str)

                # BattleOrder 생성
                order = BattleOrder(
                    order=move_obj,
                    move_target=move_target,
                    terastallize=(tera_flag and (tera_slot == (slot_idx + 1))),
                )
                orders_per_slot.append(order)

            # -------------------------------
            # 2) 교체 (나중에 일반 턴에서도 허용하고 싶으면 구현)
            # -------------------------------
            elif action == "switch":
                # TODO: 일반 턴에서의 자발적 교체를 지원하려면
                # - cmd["switch_to"] 등을 정의하고
                # - battle.available_switches[slot_idx]에서 매칭해서 BattleOrder 생성
                # 지금은 지원 안 하므로 실패 처리
                self.logger.warning(f"Slot {slot_idx+1}: switch action not yet implemented in parser")
                return None

            else:
                self.logger.warning(f"Slot {slot_idx+1}: unknown action '{action}' in GPT decision")
                return None

        # 두 슬롯 오더를 DoubleBattleOrder로 묶어서 반환
        if len(orders_per_slot) == 2:
            return DoubleBattleOrder(orders_per_slot[0], orders_per_slot[1])
        elif len(orders_per_slot) == 1:
            return orders_per_slot[0]
        else:
            return None

    def _log_turn(
        self,
        state: BattleState,
        battle: DoubleBattle,
        legal_actions: Dict[str, Any],
        decision: Dict[str, Any],
    ) -> None:
        """
        오프라인 분석/디버깅용 턴 로그.
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "turn": battle.turn,
            "battle_tag": state.battle_tag,
            "battle_state": state.to_gpt_dict(),
            "legal_actions": legal_actions,
            "decision": decision,
        }
        state.turn_logs.append(log_entry)

        if not self.log_dir:
            return

        try:
            import os

            os.makedirs(self.log_dir, exist_ok=True)
            path = os.path.join(self.log_dir, f"{state.battle_tag}.jsonl")
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception:
            # 로그 때문에 봇이 죽으면 안 되므로 예외는 삼킨다
            self.logger.exception("Failed to write turn log.")