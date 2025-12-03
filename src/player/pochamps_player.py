"""
PochampsPlayer - Custom Pokemon Battle AI using OpenAI API.
Implements parallel strategy evaluation with function calling.
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any

from openai import OpenAI

from src.data.gen_data import GenData
from src.environment.abstract_battle import AbstractBattle
from src.environment.double_battle import DoubleBattle
from src.player.player import Player, BattleOrder
from src.player.battle_order import DoubleBattleOrder
from src.player.functions import BATTLE_TOOLS, ToolExecutor
from src.player.strategies import call_fast_strategy, call_normal_strategy, call_deep_strategy


# =============================================================================
# Probabilistic Info Structures
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
    - Fast: Quick tactical response (gpt-4o-mini)
    - Normal: Balanced with tool usage (backend model)
    - Deep: Iterative analysis (gpt-5.1 with reasoning)
    """

    def __init__(
        self,
        battle_format: str = "gen9vgc2025regi",
        api_key: str = "",
        backend: str = "gpt-4o",
        temperature: float = 0.8,
        log_dir: Optional[str] = None,
        team=None,
        save_replays=None,
        account_configuration=None,
        server_configuration=None
    ):
        super().__init__(
            battle_format=battle_format,
            team=team,
            save_replays=save_replays,
            account_configuration=account_configuration,
            server_configuration=server_configuration
        )

        # API Configuration
        self.backend = backend
        self.api_key = api_key
        self.temperature = temperature
        self.log_dir = log_dir
        self.global_timeout = 25.0

        # OpenAI client & thread pool
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.executor = ThreadPoolExecutor(max_workers=3)

        # Game data
        self.gen = GenData.from_format(battle_format)
        self._load_game_data()

        # Token tracking (Responses API uses input_tokens, output_tokens)
        self._tokens = {"input": 0, "output": 0, "reasoning": 0}
        
        # Battle context tracking
        self._battle_response_ids: Dict[str, str] = {}
        self._strategy_progress: Dict[str, Dict] = {}
        
        # Opponent info tracking (species -> PokemonInferredInfo)
        self._opponent_info: Dict[str, PokemonInferredInfo] = {}
        # My Pokemon HP tracking
        self._my_pokemon_hp: Dict[str, float] = {}
        
        # Tool executor
        self.tool_executor = ToolExecutor(game_data={
            "moves": self.gen9_moves,
            "pokedex": self.gen9_pokedex,
            "items": self.item_effect,
            "abilities": self.ability_effect
        })

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
        except FileNotFoundError as e:
            self.logger.warning(f"Could not load game data: {e}")
            self.gen9_moves = {}
            self.ability_effect = {}
            self.item_effect = {}
            self.gen9_pokedex = {}

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
                model="gpt-4o-mini",
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
                model="gpt-5.1",
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
                    json.loads(responses[strategy])
                    self.logger.info(f"Using '{strategy}' response")
                    return responses[strategy]
                except json.JSONDecodeError:
                    continue
        return None

    # =========================================================================
    # Battle Logic
    # =========================================================================

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """Main decision function - routes to doubles or singles."""
        if isinstance(battle, DoubleBattle):
            return self._choose_doubles_move(battle)
        # Singles is rarely used in VGC - fallback to random
        return self.choose_random_move(battle)

    def _choose_doubles_move(self, battle: DoubleBattle) -> BattleOrder:
        """
        Doubles battle logic - main VGC format.
        
        Flow:
        1. Update confirmed info from last message
        2. Build battle context for GPT
        3. Call GPT with parallel strategies (fast/normal/deep)
        """
        # =====================================================================
        # Step 1: Update confirmed information from last battle message
        # =====================================================================
        self._update_battle_info(battle)
        
        # =====================================================================
        # Step 2: Build battle context for GPT
        # =====================================================================
        battle_context = self._build_doubles_context(battle)
        
        # =====================================================================
        # Step 3: GPT inference with 3 parallel strategies
        # =====================================================================
        responses = self.call_parallel_strategies(
            battle_tag=battle.battle_tag,
            battle_context=battle_context,
            tools=BATTLE_TOOLS,
            timeout=self.global_timeout
        )
        
        best = self.select_best_response(responses)
        if best:
            orders = self._parse_doubles_decision(best, battle)
            if orders:
                return orders
        
        return self.choose_random_doubles_move(battle)

    def _update_battle_info(self, battle: DoubleBattle) -> None:
        """
        Update opponent information using GPT-based inference.
        
        Process:
        1. Extract turn events (damage, moves, abilities, items)
        2. Calculate expected damage and compare with actual
        3. Use GPT to infer EVs, items, abilities from discrepancies
        4. Update confirmed info database
        """
        # Step 1: Extract direct observations from battle state
        self._extract_direct_observations(battle)
        
        # Step 2: Extract turn events for analysis
        turn_events = self._extract_turn_events(battle)
        if not turn_events:
            return
        
        # Step 3: Analyze with GPT for deeper inference
        if self.client and battle.turn > 1:
            self._analyze_turn_with_gpt(battle, turn_events)

    def _get_or_create_pokemon_info(self, species: str) -> PokemonInferredInfo:
        """Get or create PokemonInferredInfo for a species."""
        if species not in self._opponent_info:
            self._opponent_info[species] = PokemonInferredInfo(species=species)
        return self._opponent_info[species]

    def _extract_direct_observations(self, battle: DoubleBattle) -> None:
        """Extract directly observable information from battle state."""
        for pokemon in battle.opponent_team.values():
            info = self._get_or_create_pokemon_info(pokemon.species)
            
            # Confirmed ability (if revealed) - 100% probability
            if pokemon.ability:
                info.set_observed("ability", pokemon.ability, battle.turn, 
                                  "Ability revealed in battle")
            
            # Confirmed item (if revealed or consumed) - 100% probability
            if pokemon.item:
                info.set_observed("item", pokemon.item, battle.turn,
                                  "Item revealed/consumed in battle")
            
            # Confirmed moves (from usage) - 100% probability
            if pokemon.moves:
                for move_id in pokemon.moves.keys():
                    info.add_move(move_id, battle.turn)
            
            # Confirmed tera type (if terastallized) - 100% probability
            if pokemon.terastallized and pokemon.tera_type:
                info.set_observed("tera_type", pokemon.tera_type.name if hasattr(pokemon.tera_type, 'name') else str(pokemon.tera_type),
                                  battle.turn, "Pokemon terastallized")
            
            # Track HP changes for damage analysis
            current_hp = round(pokemon.current_hp_fraction * 100, 1)
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
                model="gpt-4o-mini",
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

    def _build_doubles_context(self, battle: DoubleBattle) -> Dict:
        """Build doubles battle context for GPT."""
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
        
        # My active Pokemon (up to 2)
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
                
                context["my_active"].append({
                    "slot": i + 1,
                    "species": pokemon.species,
                    "hp_percent": round(pokemon.current_hp_fraction * 100, 1),
                    "status": str(pokemon.status) if pokemon.status else None,
                    "moves": moves_info,
                    "item": pokemon.item,
                    "ability": pokemon.ability,
                    "types": [t.name for t in pokemon.types if t],
                    "tera_type": pokemon.tera_type.name if pokemon.tera_type else None,
                    "can_terastallize": battle.can_tera[i] if i < len(battle.can_tera) else False
                })
        
        # Opponent active Pokemon (up to 2) - use PokemonInferredInfo
        for i, pokemon in enumerate(battle.opponent_active_pokemon):
            if pokemon:
                # Get probabilistic info for this Pokemon
                inferred_info = self._opponent_info.get(pokemon.species)
                
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
                
                # Add inferred info from PokemonInferredInfo
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
                    
                    # EV spread estimates
                    if inferred_info.ev_spreads:
                        opp_data["ev_estimates"] = [e.to_dict() for e in inferred_info.ev_spreads[:2]]
                    
                    # Nature
                    if inferred_info.nature:
                        opp_data["nature"] = inferred_info.nature.to_dict()
                    
                    # Speed tier observations
                    if inferred_info.speed_observations:
                        opp_data["speed_info"] = inferred_info.speed_observations[-3:]  # Last 3
                
                context["opponent_active"].append(opp_data)
        
        # My full team
        for p in battle.team.values():
            context["my_team"].append({
                "species": p.species,
                "hp_percent": round(p.current_hp_fraction * 100, 1),
                "fainted": p.fainted,
                "active": p.active
            })
        
        # Opponent revealed team - use PokemonInferredInfo
        for p in battle.opponent_team.values():
            inferred_info = self._opponent_info.get(p.species)
            
            opp_data = {
                "species": p.species,
                "hp_percent": round(p.current_hp_fraction * 100, 1),
                "fainted": p.fainted,
                "active": p.active,
                "known_moves": list(p.moves.keys()) if p.moves else [],
                "item": {"value": p.item, "probability": 100.0} if p.item else None
            }
            
            # Add inferred info
            if inferred_info:
                if not opp_data["item"] and inferred_info.item:
                    opp_data["item"] = inferred_info.item.to_dict()
                if inferred_info.ability:
                    opp_data["ability"] = inferred_info.ability.to_dict()
                if inferred_info.ev_spreads:
                    opp_data["ev_estimates"] = [e.to_dict() for e in inferred_info.ev_spreads[:2]]
            
            context["opponent_revealed"].append(opp_data)
        
        # Available switches
        context["available_switches"] = [p.species for p in battle.available_switches]
        
        return context

    def _parse_doubles_decision(self, response: str, battle: DoubleBattle) -> Optional[BattleOrder]:
        """Parse GPT doubles decision to DoubleBattleOrder."""
        try:
            decision = json.loads(response)
            orders: List[Optional[BattleOrder]] = [None, None]
            
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
                                orders[i] = BattleOrder(
                                    move, 
                                    move_target=int(target) if target else 0,
                                    terastallize=tera
                                )
                                break
                
                elif action_type == "switch":
                    pokemon_name = slot_data.get("pokemon", "").lower()
                    
                    for pokemon in battle.available_switches:
                        if pokemon_name in pokemon.species.lower() or pokemon.species.lower() in pokemon_name:
                            orders[i] = BattleOrder(pokemon)
                            break
            
            # Build DoubleBattleOrder
            if orders[0] and orders[1]:
                return DoubleBattleOrder(first_order=orders[0], second_order=orders[1])
            elif orders[0]:
                return DoubleBattleOrder(first_order=orders[0])
            elif orders[1]:
                return DoubleBattleOrder(second_order=orders[1])
                
        except Exception as e:
            self.logger.error(f"Parse doubles error: {e}")
        return None

    def teampreview(self, battle: AbstractBattle) -> str:
        """Handle team preview - 상대 팀 확인 후 기권."""
        self.logger.info("=" * 60)
        self.logger.info("TEAM PREVIEW")
        self.logger.info("=" * 60)
        
        # 내 팀 출력
        self.logger.info("【 MY TEAM 】")
        for i, (species, mon) in enumerate(battle.team.items(), 1):
            types_str = "/".join([t.name for t in mon.types if t])
            ability_str = mon.ability if mon.ability else "Unknown"
            item_str = mon.item if mon.item else "Unknown"
            self.logger.info(f"  {i}. {species:<20} | Type: {types_str:<20} | Ability: {ability_str:<15} | Item: {item_str}")
        
        self.logger.info("-" * 60)
        
        # 상대 팀 출력
        self.logger.info("【 OPPONENT TEAM 】")
        for species, mon in battle.opponent_team.items():
            types_str = "/".join([t.name for t in mon.types if t])
            base_stats = mon.base_stats
            stats_str = f"HP:{base_stats['hp']} Atk:{base_stats['atk']} Def:{base_stats['def']} SpA:{base_stats['spa']} SpD:{base_stats['spd']} Spe:{base_stats['spe']}"
            self.logger.info(f"  - {species:<20} | Type: {types_str:<20} | {stats_str}")
        
        self.logger.info("=" * 60)
        self.logger.info("Forfeiting battle for testing...")
        self.logger.info("=" * 60)
        
        # 기권
        return "/forfeit"

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def _battle_finished_callback(self, battle: AbstractBattle):
        """Clean up after battle."""
        if battle.battle_tag in self._battle_response_ids:
            del self._battle_response_ids[battle.battle_tag]
        
        for key in list(self._strategy_progress.keys()):
            if battle.battle_tag in key:
                del self._strategy_progress[key]
        
        # Clear opponent info for this battle
        self._opponent_info.clear()
        self._my_pokemon_hp.clear()
        
        self.logger.info(f"Battle finished: {battle.battle_tag}")
        self.logger.info(f"Result: {'Won' if battle.won else 'Lost' if battle.lost else 'Tie'}")
        self.logger.info(f"Tokens: input={self._tokens['input']}, output={self._tokens['output']}, reasoning={self._tokens['reasoning']}")

    def __del__(self):
        """Cleanup."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
