import asyncio
from tqdm import tqdm
import numpy as np
import os
import pickle as pkl
import argparse
import sys
import io
import random
import logging

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Configure global logging - INFO level for clean output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Fix Windows console encoding issue for Unicode characters
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from src.player import HeuristicsPlayer, PochampsPlayer
from src.data.pochamps_teams import get_random_teams

parser = argparse.ArgumentParser()
# Mode selection
parser.add_argument("--mode", type=str, default="local", choices=["local", "online", "ladder"],
                    help="Battle mode: 'local' (vs bot), 'online' (accept challenges), 'ladder' (ranked)")

# Account configuration (for online/ladder modes)
parser.add_argument("--username", type=str, default=None,
                    help="Bot username for local server, or PS username for ladder (default: PokeLLMonBot)")
parser.add_argument("--password", type=str, default=None,
                    help="Pokemon Showdown password (only needed for ladder mode)")
parser.add_argument("--opponent", type=str, default=None,
                    help="Opponent username to challenge (for online mode). If not set, bot waits for challenges.")

# Model configuration
parser.add_argument("--backend", type=str, default="gpt-4o-2024-08-06",
                    help="Main model for normal strategy (e.g., gpt-4o-2024-08-06, gpt-oss-20b)")
parser.add_argument("--fast_model", type=str, default=None,
                    help="Model for fast strategy (default: same as backend)")
parser.add_argument("--deep_model", type=str, default=None,
                    help="Model for deep strategy (default: same as backend)")
parser.add_argument("--base_url", type=str, default=None,
                    help="Custom API base URL for local OSS models (e.g., http://localhost:8080)")
parser.add_argument("--temperature", type=float, default=0.8)

# Battle configuration
parser.add_argument("--log_dir", type=str, default="./battle_log/pokellmon_vs_bot")
parser.add_argument("--n_battles", type=int, default=5, help="Number of battles to run")
parser.add_argument("--change_team_every_battle", action="store_true", 
                    help="If set, select new random teams for each battle. Otherwise, use the same teams for all battles.")
parser.add_argument("--open_team_sheets", action="store_true",
                    help="If set, accept Open Team Sheets (OTS). Otherwise, reject and use Closed Team Sheets (CTS).")
parser.add_argument("--debug", action="store_true",
                    help="Debug mode: use fixed teams (first 2 from pool) and cache GPT responses to avoid API costs.")
args = parser.parse_args()

"""
async def main():
    battle_format = "gen8randombattle"
    heuristic_player = HeuristicsPlayer(battle_format=battle_format)

    os.makedirs(args.log_dir, exist_ok=True)

    llm_player = LLMPlayer(battle_format=battle_format,
                           api_key=os.getenv("OPENAI_API_KEY"),
                           backend=args.backend,
                           temperature=args.temperature,
                           prompt_algo=args.prompt_algo,
                           log_dir=args.log_dir,
                           # account_configuration=AccountConfiguration("Your_account", "Your_password"),
                           save_replays=args.log_dir
                           )

    # dynamax is disabled for local battles.
    heuristic_player._dynamax_disable = True
    llm_player._dynamax_disable = True

    # play against bot for five battles
    for i in tqdm(range(5)):
        x = np.random.randint(0, 100)
        if x > 50:
            await heuristic_player.battle_against(llm_player, n_battles=1)
        else:
            await llm_player.battle_against(heuristic_player, n_battles=1)
        for battle_id, battle in llm_player.battles.items():
            with open(f"{args.log_dir}/{battle_format}_{args.backend}_{args.prompt_algo}_{battle_id}.pkl", "wb") as f:
                pkl.dump(battle, f)
"""

async def main():
    battle_format = "gen9vgc2025regh"
    os.makedirs(args.log_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Battle Configuration:")
    print(f"  - Number of battles: {args.n_battles}")
    print(f"  - Change teams every battle: {args.change_team_every_battle}")
    print(f"  - Open Team Sheets (OTS): {args.open_team_sheets}")
    print(f"  - Debug mode: {args.debug}")
    print(f"Model Configuration:")
    print(f"  - Backend (Normal): {args.backend}")
    print(f"  - Fast Model: {args.fast_model or args.backend}")
    print(f"  - Deep Model: {args.deep_model or args.backend}")
    print(f"  - Base URL: {args.base_url or 'OpenAI default'}")
    print(f"{'='*60}\n")
    
    # Debug mode: override settings
    if args.debug:
        args.change_team_every_battle = False  # 항상 고정 팀 사용
        print("[DEBUG MODE] Fixed teams + GPT response caching enabled")
    
    # Create players once without teams initially
    heuristic_player = HeuristicsPlayer(battle_format=battle_format, open_team_sheets=args.open_team_sheets)
    pochamps_player = PochampsPlayer(battle_format=battle_format,
                                     api_key=os.getenv("OPENAI_API_KEY"),
                                     backend=args.backend,
                                     fast_model=args.fast_model,
                                     deep_model=args.deep_model,
                                     base_url=args.base_url,
                                     temperature=args.temperature,
                                     log_dir=args.log_dir,
                                     save_replays=args.log_dir,
                                     open_team_sheets=args.open_team_sheets,
                                     debug_mode=args.debug
                                     )
    
    heuristic_player._dynamax_disable = True
    pochamps_player._dynamax_disable = True
    
    # Wait for both players to be logged in, then cleanup any stale sessions
    await heuristic_player.ps_client.wait_for_login()
    await pochamps_player.ps_client.wait_for_login()
    
    # Run cleanup for both players concurrently to forfeit any auto-rejoined battles
    await asyncio.gather(
        heuristic_player.cleanup_stale_sessions(wait_time=3.0),
        pochamps_player.cleanup_stale_sessions(wait_time=3.0)
    )
    
    # Select initial teams if not changing every battle
    if not args.change_team_every_battle:
        if args.debug:
            # Debug mode: always use first 2 teams (deterministic)
            import json
            teams_json_path = "src/data/gen9vgc2025regh_team.json"
            with open(teams_json_path, "r", encoding="utf-8") as f:
                all_teams = json.load(f)
            team1 = all_teams[0]["showdown"]
            team2 = all_teams[1]["showdown"]
            print(f"[DEBUG] Using fixed teams (index 0, 1):")
        else:
            team1, team2 = get_random_teams(n=2, format=battle_format)
            print(f"Selected 2 teams for all battles:")
        print(f"Team 1 (HeuristicsPlayer): {team1.split(chr(10))[0][:50]}...")
        print(f"Team 2 (PochampsPlayer): {team2.split(chr(10))[0][:50]}...\n")
        
        # Update teams once
        heuristic_player.update_team(team1)
        pochamps_player.update_team(team2)
    
    try:
        # Battle loop
        for i in tqdm(range(args.n_battles), desc="Battles"):
            print("\n\n\n\n\n")
            print(f"{'='*20} Starting Battle {i+1}/{args.n_battles} {'='*20}\n")
            # Forfeit and clean up any stale battle sessions before starting new battle
            await heuristic_player.forfeit_all_battles()
            await pochamps_player.forfeit_all_battles()
            
            # Randomly decide who initiates the battle
            x = random.randint(0, 100)
            if x > 50:
                # heuristic_player (p1 with team1) battles against pochamps_player (p2 with team2)
                await heuristic_player.battle_against(pochamps_player, n_battles=1)
            else:
                # pochamps_player (p1 with team2) battles against heuristic_player (p2 with team1)
                await pochamps_player.battle_against(heuristic_player, n_battles=1)
            
            # Save battle logs
            for battle_id, battle in pochamps_player.battles.items():
                with open(f"{args.log_dir}/{battle_format}_{args.backend}_{battle_id}.pkl", "wb") as f:
                    pkl.dump(battle, f)
            
            # If changing teams every battle, select new teams and update
            if args.change_team_every_battle:
                team1, team2 = get_random_teams(n=2, format=battle_format)
                print(f"\nBattle {i+1}/{args.n_battles} - New teams selected:")
                print(f"  Team 1 (HeuristicsPlayer): {team1.split(chr(10))[0][:50]}...")
                print(f"  Team 2 (PochampsPlayer): {team2.split(chr(10))[0][:50]}...")
                
                # Update teams for this battle
                heuristic_player.update_team(team1)
                pochamps_player.update_team(team2)
    
    except (KeyboardInterrupt, Exception) as e:
        print(f"\n\n[CLEANUP] Error or interrupt detected: {type(e).__name__}")
        print("[CLEANUP] Forfeiting all active battles...")
        try:
            await heuristic_player.forfeit_all_battles()
            await pochamps_player.forfeit_all_battles()
            print("[CLEANUP] All battles forfeited successfully.")
        except Exception as cleanup_error:
            print(f"[CLEANUP] Warning: cleanup failed: {cleanup_error}")
        raise


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())