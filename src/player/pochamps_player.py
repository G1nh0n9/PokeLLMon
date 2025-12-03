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
from src.player.battle_order import DoubleBattleOrder, ForfeitBattleOrder
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
        backend: str = "gpt-4o-2024-08-06",
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
        
        # Opponent info tracking (species -> PokemonInferredInfo)
        self._opponent_info: Dict[str, PokemonInferredInfo] = {}
        # My Pokemon HP tracking
        self._my_pokemon_hp: Dict[str, float] = {}
        # Token tracking (Responses API uses input_tokens, output_tokens)
        self._tokens = {"input": 0, "output": 0, "reasoning": 0}
        # Battle context tracking
        self._battle_response_ids: Dict[str, str] = {}
        self._strategy_progress: Dict[str, Dict] = {}
        
        # Debug mode: cache GPT responses to avoid API costs
        self.debug_mode = debug_mode
        self._response_cache: Dict[str, Dict] = {}  # cache key -> response
        self._cache_file = "debug_gpt_cache.json" if debug_mode else None

        # API Configuration
        self.backend = backend
        self.api_key = api_key
        self.temperature = temperature
        self.log_dir = log_dir
        self.global_timeout = 25.0

        # OpenAI client & thread pool
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
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

        # Tool executor
        self.tool_executor = ToolExecutor(game_data={
            "moves": self.gen9_moves,
            "pokedex": self.gen9_pokedex,
            "items": self.item_effect,
            "abilities": self.ability_effect
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
        self.logger.warning("Battle %s - Turn %d", battle.battle_tag, battle.turn)
        # Forfeit for testing
        return ForfeitBattleOrder()
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
            if pokemon.terastallized and pokemon._terastallized_type:
                info.set_observed("tera_type", pokemon._terastallized_type.name if hasattr(pokemon._terastallized_type, 'name') else str(pokemon._terastallized_type),
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
                    "tera_type": pokemon._terastallized_type.name if pokemon._terastallized_type else None,
                    "terastallized": pokemon.terastallized,
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
        available_switches = []
        try:
            if hasattr(battle, 'available_switches') and battle.available_switches:
                available_switches = [p.species for p in battle.available_switches]
        except Exception as e:
            self.logger.debug(f"Error getting available switches: {e}")
        context["available_switches"] = available_switches
        
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

    async def teampreview(self, battle: AbstractBattle) -> str:
        """
        Handle team preview - GPT 기반 상대 팀 분석 및 선발 선택 (비동기).
        
        Flow:
        1. 내 팀 & 상대 팀 정보 수집
        2. GPT로 상대 팀 전략 분석 (function calling으로 데이터 조회)
        3. GPT가 최적의 4마리 선발 + 선발 2마리 결정
        
        Note: GPT API 호출은 asyncio.to_thread()로 별도 스레드에서 실행하여
              이벤트 루프를 블로킹하지 않음.
        """
        import asyncio
        
        # 팀 정보 파싱
        my_team_info = self._parse_my_team()
        opponent_team_info = self._parse_opponent_team(battle)
        
        # 콘솔 출력
        self._print_team_preview(my_team_info, opponent_team_info)
        
        # GPT 분석 및 선발 선택 (비동기 - 별도 스레드에서 실행)
        print(f"\n[DEBUG] self.client = {self.client}", flush=True)
        if self.client:
            try:
                # GPT 호출을 별도 스레드에서 실행하여 이벤트 루프 블로킹 방지
                selection = await asyncio.to_thread(
                    self._analyze_and_select_team,
                    my_team_info,
                    opponent_team_info,
                    battle
                )
                if selection:
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
        battle: AbstractBattle
    ) -> Optional[str]:
        """
        GPT로 상대 팀 분석 및 선발 선택.
        
        Function Calling으로:
        - get_pokemon_info: 포켓몬 상세 정보 조회
        - get_move_info: 기술 정보 조회
        - get_type_matchup: 타입 상성 조회
        - get_common_sets: 일반적인 세팅 조회 (usage stats 기반)
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
        system_prompt = self._get_team_analysis_system_prompt()
        
        # 유저 프롬프트: 양 팀 정보
        user_prompt = self._build_team_analysis_prompt(my_team, opponent_team)
        
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
            return self._format_team_selection(cached, my_team)
        
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
            
            # Function calls 처리
            result = self._process_team_analysis_response(response, analysis_tools, my_team, opponent_team)
            
            if result:
                # Cache the result for debug mode
                self._cache_response(cache_key, result)
                return self._format_team_selection(result, my_team)
            
        except Exception as e:
            self.logger.error(f"Team analysis error: {e}")
        
        return None
    
    def _get_team_analysis_system_prompt(self) -> str:
        """팀 분석용 시스템 프롬프트 - 병렬 처리 최적화."""
        return """You are an expert Pokemon VGC analyst. You have 60 SECONDS to analyze and respond.

# CRITICAL: PARALLEL TOOL CALLS

**Call ALL tools you need in ONE batch.** Do NOT call 5 tools, wait, then call more.
- Need info on 6 Pokemon? Call get_pokemon_info 6 times IN PARALLEL
- Need usage stats for all 6? Call get_usage_stats 6 times IN PARALLEL  
- Need type matchups? Call get_type_matchup for ALL relevant matchups AT ONCE

WRONG: Call 5 tools → wait → call 5 more → wait → call 5 more
RIGHT: Call ALL 15+ tools in ONE request → get all data → make decision

# ANALYSIS APPROACH

1. **Information Gathering (ONE parallel batch)**
   - get_pokemon_info for ALL 6 opponent Pokemon
   - get_usage_stats for ALL 6 opponent Pokemon
   - get_type_matchup for key offensive matchups
   - DO THIS ALL IN ONE TOOL CALL BATCH

2. **Analysis & Decision (after data received)**
   - Infer opponent team archetype and strategy
   - Identify key threats and win conditions
   - Select optimal 4 Pokemon and leads

# OUTPUT FORMAT (Final Response Only)

```json
{
    "opponent_analysis": {
        "archetype": "<team style: e.g., Sun Offense, Trick Room, Balance>",
        "key_threats": ["pokemon1", "pokemon2"],
        "likely_leads": ["pokemon1", "pokemon2"]
    },
    "selection": {
        "bring_4": [slot1, slot2, slot3, slot4],
        "lead_2": [slot1, slot2]
    },
    "reasoning": "<brief 1-2 sentence strategy>"
}
```

REMEMBER: Call ALL tools at once. Time is critical."""

    def _build_team_analysis_prompt(self, my_team: List[Dict], opponent_team: List[Dict]) -> str:
        """팀 분석용 유저 프롬프트 생성."""
        prompt = "## MY TEAM (Choose 4 from these 6)\n\n"
        
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
        prompt += "Call ALL tools (get_pokemon_info, get_usage_stats for each opponent) IN ONE BATCH.\n"
        prompt += "Then make your selection. You have 60 seconds total."
        
        return prompt
    
    def _process_team_analysis_response(
        self, 
        response, 
        tools: List[Dict],
        my_team: List[Dict],
        opponent_team: List[Dict]
    ) -> Optional[Dict]:
        """
        GPT 응답 처리 (function calling).
        
        - 이상적으로 1라운드에 모든 tool call을 병렬 실행
        - 최대 3라운드까지만 허용 (그 이상은 비효율)
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
    
    def _format_team_selection(self, result: Dict, my_team: List[Dict]) -> str:
        """GPT 분석 결과를 showdown 포맷으로 변환."""
        selection = result.get("selection", {})
        bring_4 = selection.get("bring_4", [1, 2, 3, 4])
        lead_2 = selection.get("lead_2", bring_4[:2] if bring_4 else [1, 2])
        
        # 분석 결과 출력
        opponent_analysis = result.get("opponent_analysis", {})
        reasoning = result.get("reasoning", "")
        
        print("\n" + "=" * 70)
        print("GPT TEAM ANALYSIS")
        print("=" * 70)
        
        if opponent_analysis and isinstance(opponent_analysis, dict):
            print(f"Opponent Archetype: {opponent_analysis.get('archetype', 'Unknown')}")
            key_threats = opponent_analysis.get('key_threats', [])
            likely_leads = opponent_analysis.get('likely_leads', [])
            if isinstance(key_threats, list):
                print(f"Key Threats: {', '.join(key_threats)}")
            if isinstance(likely_leads, list):
                print(f"Likely Leads: {', '.join(likely_leads)}")
        
        print("-" * 70)
        print("SELECTION:")
        
        # 선택된 4마리 출력
        bring_names = []
        for slot in bring_4:
            if isinstance(slot, int) and 1 <= slot <= len(my_team):
                mon = my_team[slot - 1]
                bring_names.append(mon['species'])
                lead_marker = " [LEAD]" if slot in lead_2 else ""
                print(f"  Slot {slot}: {mon['species']}{lead_marker}")
        
        print("-" * 70)
        if reasoning:
            if isinstance(reasoning, dict):
                print(f"Strategy: {reasoning.get('game_plan', reasoning.get('why_these_leads', 'N/A'))}")
            else:
                print(f"Strategy: {reasoning}")
        print("=" * 70)
        
        # Showdown 포맷: /team ABCD (선발 2마리가 앞에 오도록)
        # lead_2를 먼저, 나머지를 뒤에
        back_2 = [s for s in bring_4 if s not in lead_2]
        ordered = lead_2 + back_2
        
        team_str = "".join([str(s) for s in ordered])
        return f"/team {team_str}"

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
