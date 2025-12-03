"""
PoChamps VGC Teams Repository
- Store Showdown-format teams with metadata (format/name/source)

CLI utilities when executed as a script:
- Export in-memory teams to a JSON file
- Append a new team into a JSON file
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Dict, Any


# Default JSON path in this folder
DEFAULT_JSON_PATH = os.path.join(os.path.dirname(__file__), "gen9vgc2025regh_team.json")

# Single team entry for manual editing and registration
TEAM_ENTRY: Dict[str, Any] = {
    "format": "gen9vgc2025regh",
    "name": "2025 Taiwan Premier Ball League Champion(紫竽隊)",
    "source": " by Johnnychiou36",
    "showdown": """紫竽超可愛 (Indeedee-F) @ Psychic Seed  
Ability: Psychic Surge  
Level: 50  
Shiny: Yes  
Tera Type: Fire  
EVs: 252 HP / 252 Def / 4 SpD  
Relaxed Nature  
IVs: 0 Atk / 0 Spe  
- Trick Room  
- Follow Me  
- Helping Hand  
- Dazzling Gleam  

紫竽敲可愛 (Hatterene) @ Life Orb  
Ability: Magic Bounce  
Level: 50  
Shiny: Yes  
Tera Type: Psychic  
EVs: 212 HP / 44 Def / 252 SpA  
Quiet Nature  
IVs: 0 Atk / 0 Spe  
- Expanding Force  
- Trick Room  
- Dazzling Gleam  
- Protect  

紫竽親衛隊 (Gallade) @ Clear Amulet  
Ability: Sharpness  
Level: 50  
Shiny: Yes  
Tera Type: Grass  
EVs: 212 HP / 252 Atk / 44 Spe  
Adamant Nature  
- Psycho Cut  
- Trick Room  
- Sacred Sword  
- Wide Guard  

一拳月月熊 (Ursaluna) (M) @ Flame Orb  
Ability: Guts  
Level: 50  
Shiny: Yes  
Tera Type: Ghost  
EVs: 252 HP / 252 Atk / 4 SpD  
Brave Nature  
IVs: 0 Spe  
- Headlong Rush  
- Facade  
- Swords Dance  
- Protect  

紫竽好可愛 (Lilligant-Hisui) @ Focus Sash  
Ability: Chlorophyll  
Level: 50  
Shiny: Yes  
Tera Type: Ghost  
EVs: 4 HP / 252 Atk / 252 Spe  
Jolly Nature  
- Solar Blade  
- Sleep Powder  
- Close Combat  
- After You  

太樂巴戈斯 (Torkoal) (M) @ Choice Specs  
Ability: Drought  
Level: 50  
Shiny: Yes  
Tera Type: Fire  
EVs: 252 HP / 4 Def / 252 SpA  
Quiet Nature  
IVs: 0 Atk / 0 Spe  
- Eruption  
- Heat Wave  
- Weather Ball  
- Earth Power  
""",
}


def get_team_by_index(index: int, *, format: str | None = None) -> str:
    """
    Get a team by index.
    
    Args:
        index: Team index (0-based)
    
    Returns:
        Team string in Pokemon Showdown format
    """
    pool = _filter_teams(format=format)
    if 0 <= index < len(pool):
        return pool[index]["showdown"]
    else:
        raise IndexError(f"Team index {index} out of range (0-{len(pool)-1})")

def get_random_teams(n: int = 2, seed=None, *, format: str | None = None) -> list:
    """
    Get n random teams without replacement.
    
    Args:
        n: Number of teams to select
        seed: Random seed for reproducibility
    
    Returns:
        List of team strings (Showdown format)
    """
    import random
    if seed is not None:
        random.seed(seed)
    
    pool = _filter_teams(format=format)
    if n > len(pool):
        raise ValueError(
            f"Cannot select {n} teams from {len(pool)} available teams"
            + (f" (format={format})" if format else "")
        )
    
    indices = random.sample(range(len(pool)), n)
    return [pool[i]["showdown"] for i in indices]

def get_team_count(*, format: str | None = None) -> int:
    """Get the total number of available teams, optionally filtered."""
    return len(_filter_teams(format=format))



def _filter_teams(*, format: str | None) -> list[dict]:
    pool = json.load(open(os.path.join(os.path.dirname(__file__), "gen9vgc2025regh_team.json"), "r", encoding="utf-8"))
    if format:
        pool = [t for t in pool if t.get("format") == format]
    return pool


# ---------------------------------------------------------------------------
# JSON I/O utilities
# ---------------------------------------------------------------------------

def add_team_to_json(
    json_path: str,
    *,
    format: str,
    name: str,
    showdown: str,
    source: str = "user",
) -> None:
    """Append a team entry into a JSON file (creating it if missing).

    The JSON file stores a list of objects with keys: format, name, source, showdown.
    """
    teams: List[Dict[str, Any]] = []
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    teams = loaded
                else:
                    raise ValueError("JSON root must be a list of teams")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {json_path}: {e}") from e

    entry = {
        "format": format,
        "name": name,
        "source": source,
        "showdown": showdown.strip(),
    }
    teams.append(entry)

    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(teams, f, ensure_ascii=False, indent=2)


def normalize_showdown_team(showdown: str) -> str:
    """
    Normalize a Showdown team string by removing nicknames.
    
    Converts lines like:
      "Nickname (Species) @ Item" -> "Species @ Item"
      "Nickname (Species) (M) @ Item" -> "Species (M) @ Item"
    
    Leaves lines without nicknames unchanged:
      "Species @ Item" -> "Species @ Item"
    
    Args:
        showdown: Raw Showdown team string
    
    Returns:
        Normalized team string with nicknames removed
    """
    import re
    
    lines = showdown.split('\n')
    normalized_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        # Check if this is a Pokemon header line (contains @ or starts a new Pokemon block)
        # Pattern: "Nickname (Species) [optional (M)/(F)] [optional @ Item]"
        # We want to extract: "Species [optional (M)/(F)] [optional @ Item]"
        
        # Match: anything before '(' then '(Species)' then optional gender then optional item
        match = re.match(r'^(.+?)\s+\(([^)]+)\)\s*(\([MF]\))?\s*(@.+)?$', stripped)
        
        if match:
            # group(1): nickname
            # group(2): actual species name
            # group(3): gender like (M) or (F), if present
            # group(4): @ Item part, if present
            species = match.group(2)
            gender = match.group(3) or ""
            item = match.group(4) or ""
            
            # Reconstruct without nickname
            normalized = f"{species} {gender} {item}".strip()
            normalized_lines.append(normalized)
        else:
            # Not a nickname pattern, keep as-is
            normalized_lines.append(line)
    
    return '\n'.join(normalized_lines)


def validate_and_fix_team_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and fix a team entry by normalizing the showdown string.
    
    Args:
        entry: Team entry dict with 'showdown' field
    
    Returns:
        Fixed team entry with normalized showdown string
    """
    fixed_entry = entry.copy()
    if "showdown" in fixed_entry:
        original = fixed_entry["showdown"]
        normalized = normalize_showdown_team(original)
        fixed_entry["showdown"] = normalized
    return fixed_entry


def normalize_existing_json(json_path: str = DEFAULT_JSON_PATH) -> int:
    """
    Normalize all teams in an existing JSON file by removing nicknames.
    
    Args:
        json_path: Path to the JSON file to normalize
    
    Returns:
        Number of teams that were modified
    """
    if not os.path.exists(json_path):
        print(f"JSON file not found: {json_path}")
        return 0
    
    # Load existing teams
    with open(json_path, "r", encoding="utf-8") as f:
        teams = json.load(f)
    
    if not isinstance(teams, list):
        raise ValueError("JSON root must be a list of teams")
    
    # Normalize each team
    modified_count = 0
    normalized_teams = []
    
    for team in teams:
        if "showdown" in team:
            original = team["showdown"]
            normalized = normalize_showdown_team(original)
            
            if original != normalized:
                modified_count += 1
                print(f"Normalized team: {team.get('name', 'Unnamed')}")
            
            team["showdown"] = normalized
        
        normalized_teams.append(team)
    
    # Save back to file
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(normalized_teams, f, ensure_ascii=False, indent=2)
    
    return modified_count


def append_team_entry(entry: Dict[str, Any], json_path: str = DEFAULT_JSON_PATH) -> None:
    """Append a single team entry to the JSON store.

    If the JSON file does not exist, it will be created with a list root.
    Automatically normalizes the team by removing nicknames.
    """
    # Validate and fix the entry before adding
    fixed_entry = validate_and_fix_team_entry(entry)
    
    add_team_to_json(
        json_path,
        format=fixed_entry.get("format", "gen9vgc2025regh"),
        name=fixed_entry.get("name", "Unnamed Team"),
        showdown=fixed_entry.get("showdown", "").strip(),
        source=fixed_entry.get("source", "user"),
    )


def _build_cli_parser() -> None:  # Deprecated
    """Deprecated: CLI is no longer used. Use TEAM_ENTRY + __main__ instead."""
    return None


def _main_cli(argv: list[str] | None = None) -> int:  # Deprecated
    """Deprecated: retained for compatibility. No-op."""
    return 0


if __name__ == "__main__":
    import sys
    
    # Check if --normalize flag is provided
    if len(sys.argv) > 1 and sys.argv[1] == "--normalize":
        # Normalize existing JSON file
        print(f"Normalizing existing teams in {DEFAULT_JSON_PATH}...")
        modified = normalize_existing_json(DEFAULT_JSON_PATH)
        print(f"Completed: {modified} team(s) were normalized.")
    else:
        # Default: Append TEAM_ENTRY to the JSON file
        append_team_entry(TEAM_ENTRY, DEFAULT_JSON_PATH)
        print(f"Appended team '{TEAM_ENTRY.get('name')}' to {DEFAULT_JSON_PATH}")
        print(f"\nTip: Run 'python -m src.data.pochamps_teams --normalize' to normalize existing teams.")
