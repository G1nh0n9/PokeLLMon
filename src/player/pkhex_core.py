import sys
import os
from typing import Optional, List, Dict

# pythonnet을 .NET Core 런타임으로 설정 (반드시 import clr 전에!)
from pythonnet import set_runtime
set_runtime("coreclr")

import clr

# PKHeX DLL 경로 추가 (현재 파일 기준)
current_dir = os.path.dirname(os.path.abspath(__file__))
dll_path = os.path.join(current_dir, "PKHeX.Core.dll")

# DLL 파일 존재 확인
if not os.path.exists(dll_path):
    raise FileNotFoundError(f"PKHeX.Core.dll not found at: {dll_path}")

# DLL 로드 (절대 경로 사용)
try:
    clr.AddReference(dll_path)
except Exception as e:
    print(f"Failed to load DLL: {e}")
    raise

# 네임스페이스 import
try:
    from PKHeX.Core import *
    import System
    from System.Reflection import BindingFlags
except Exception as e:
    print(f"Failed to import PKHeX.Core: {e}")
    print("Available assemblies:")
    for asm in clr.ListAssemblies(True):
        if "PKHeX" in asm:
            print(f"  {asm}")
    raise


class PokemonDataExtractor:
    """PKHeX.Core를 사용하여 포켓몬 데이터를 추출하는 클래스"""

    def __init__(self, game_version=None, language="en"):
        """
        Initialize Pokemon data extractor.

        Args:
            game_version: GameVersion enum (default: GameVersion.SV for Gen 9)
            language: Language code ("ko" for Korean, "en" for English, "ja" for Japanese)
        """
        # 기본값은 9세대 SV
        if game_version is None:
            game_version = GameVersion.SV

        # GameData를 통해 Personal과 LearnSource 가져오기
        self.game_version = game_version
        self.personal_table = GameData.GetPersonal(game_version)
        self.learn_source = GameData.GetLearnSource(game_version)

        # 언어 설정
        self.language = language
        self.species_names = Util.GetSpeciesList(language)
        self.move_names = Util.GetMovesList(language)
        self.ability_names = Util.GetAbilitiesList(language)
        self.item_names = Util.GetItemsList(language)
        self.type_names = Util.GetTypesList(language)

    def get_move_id(self, move_name: str) -> Optional[int]:
        """
        기술 이름으로 기술 번호 찾기

        Args:
            move_name: 기술 이름 (한글 또는 영어)

        Returns:
            기술 번호 (move ID), 찾지 못하면 None
        """
        move_name_lower = move_name.lower().replace(" ", "").replace("-", "").replace("'", "")
        for i, name in enumerate(self.move_names):
            if name.lower().replace(" ", "").replace("-", "").replace("'", "") == move_name_lower:
                return i
        return None

    def get_move_info(self, move_name: str) -> Optional[Dict]:
        """
        기술의 상세 정보 가져오기 (gen9moves.json 사용)

        Args:
            move_name: 기술 이름

        Returns:
            기술 정보 딕셔너리 또는 None
            {
                "name": str,
                "move_id": int,
                "type": str,
                "category": str (Physical/Special/Status),
                "base_power": int,
                "accuracy": int,
                "pp": int,
                "priority": int,
                "target": str,
                "flags": Dict[str, bool]
            }
        """
        # gen9moves.json 파일에서 기술 데이터 로드
        try:
            import json
            moves_file = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                "data", "static", "moves", "gen9moves.json"
            )
            
            if not os.path.exists(moves_file):
                print(f"Move data file not found: {moves_file}")
                return None
            
            with open(moves_file, "r", encoding="utf-8") as f:
                moves_data = json.load(f)
            
            # 기술 이름 정규화
            move_name_normalized = move_name.lower().replace(" ", "").replace("-", "").replace("'", "")
            
            # gen9moves.json에서 기술 찾기
            move_data = moves_data.get(move_name_normalized)
            if not move_data:
                print(f"Move '{move_name}' not found in gen9moves.json")
                return None
            
            # 기술 ID 가져오기
            move_id = self.get_move_id(move_name)
            
            return {
                "name": move_data.get("name", move_name),
                "move_id": move_id if move_id else 0,
                "type": move_data.get("type", "Unknown"),
                "category": move_data.get("category", "Unknown"),
                "base_power": move_data.get("basePower", 0),
                "accuracy": move_data.get("accuracy", 100),
                "pp": move_data.get("pp", 0),
                "priority": move_data.get("priority", 0),
                "target": move_data.get("target", "normal"),
                "flags": move_data.get("flags", {}),
                "desc": move_data.get("desc", move_data.get("shortDesc", ""))
            }
            
        except Exception as e:
            print(f"Failed to get move info for '{move_name}': {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_species_id(self, pokemon_name: str) -> Optional[int]:
        """
        포켓몬 이름으로 종족 번호 찾기

        Args:
            pokemon_name: 포켓몬 이름 (한글 또는 영어)

        Returns:
            종족 번호 (species ID), 찾지 못하면 None
        """
        for i, name in enumerate(self.species_names):
            if name.lower() == pokemon_name.lower():
                return i
        return None

    def get_base_stats(self, species: int, form: int = 0) -> Dict[str, int]:
        """
        베이스 스탯 가져오기

        Args:
            species: 종족 번호
            form: 폼 번호 (기본값: 0)

        Returns:
            스탯 딕셔너리 (hp, atk, def, spa, spd, spe, total)
        """
        personal_info = self.personal_table[species, form]

        return {
            "hp": personal_info.HP,
            "atk": personal_info.ATK,
            "def": personal_info.DEF,
            "spa": personal_info.SPA,
            "spd": personal_info.SPD,
            "spe": personal_info.SPE,
            "total": (personal_info.HP + personal_info.ATK + personal_info.DEF +
                     personal_info.SPA + personal_info.SPD + personal_info.SPE)
        }

    def get_abilities(self, species: int, form: int = 0) -> Dict[str, Dict]:
        """
        특성 목록 가져오기

        Args:
            species: 종족 번호
            form: 폼 번호 (기본값: 0)

        Returns:
            특성 딕셔너리 (ability1, ability2, abilityH)
        """
        personal_info = self.personal_table[species, form]

        abilities = {}

        # Ability 1
        if personal_info.Ability1 > 0:
            abilities["0"] = self.ability_names[personal_info.Ability1]

        # Ability 2 (다른 경우만)
        if personal_info.Ability2 > 0 and personal_info.Ability2 != personal_info.Ability1:
            abilities["1"] = self.ability_names[personal_info.Ability2]

        # Hidden Ability
        if personal_info.AbilityH > 0 and personal_info.AbilityH != personal_info.Ability1:
            abilities["H"] = self.ability_names[personal_info.AbilityH]

        return abilities

    def get_types(self, species: int, form: int = 0) -> List[str]:
        """
        타입 가져오기

        Args:
            species: 종족 번호
            form: 폼 번호 (기본값: 0)

        Returns:
            타입 리스트 (1개 또는 2개)
        """
        personal_info = self.personal_table[species, form]

        type1 = self.type_names[personal_info.Type1]
        types = [type1]

        # 타입2가 타입1과 다른 경우만 추가
        if personal_info.Type1 != personal_info.Type2:
            type2 = self.type_names[personal_info.Type2]
            types.append(type2)

        return types

    def get_legal_moves(self, species: int, form: int = 0, verbose: bool = False) -> List[str]:
        """
        합법적으로 배울 수 있는 모든 기술 목록 (레벨업, 기술머신, 알 기술, 떠올리기)

        Args:
            species: 종족 번호
            form: 폼 번호 (기본값: 0)
            verbose: True면 카테고리별 기술 출력

        Returns:
            기술명 리스트
        """
        try:
            import System

            # 각 카테고리별로 분류
            levelup_moves = set()
            egg_moves = set()
            reminder_moves = set()
            machine_moves_set = set()

            personal_info = self.personal_table[species, form]
            learn_source_type = self.learn_source.GetType()

            # 1. 레벨업 기술
            learnset = self.learn_source.GetLearnset(species, form)
            for move_id in range(1, min(len(self.move_names), 920)):
                if learnset.GetIsLearn(move_id):
                    levelup_moves.add(move_id)

            if verbose:
                print(f"\n[1. 레벨업 기술] {len(levelup_moves)}개")
                for move_id in sorted(levelup_moves)[:10]:
                    print(f"  - {self.move_names[move_id]} (ID: {move_id})")
                if len(levelup_moves) > 10:
                    print(f"  ... 외 {len(levelup_moves) - 10}개")

            # 2. 알 기술
            egg_move_method = learn_source_type.GetMethod("GetIsEggMove")
            if egg_move_method:
                for move_id in range(1, min(len(self.move_names), 920)):
                    try:
                        result = egg_move_method.Invoke(
                            self.learn_source,
                            System.Array[System.Object]([
                                System.UInt16(species),
                                System.Byte(form),
                                System.UInt16(move_id)
                            ])
                        )
                        if result:
                            egg_moves.add(move_id)
                    except:
                        pass

            if verbose:
                print(f"\n[2. 알 기술] {len(egg_moves)}개")
                if egg_moves:
                    for move_id in sorted(egg_moves)[:10]:
                        print(f"  - {self.move_names[move_id]} (ID: {move_id})")
                    if len(egg_moves) > 10:
                        print(f"  ... 외 {len(egg_moves) - 10}개")

            # 3. 떠올리기 기술
            reminder_method = learn_source_type.GetMethod("GetIsReminderMove")
            if reminder_method:
                for move_id in range(1, min(len(self.move_names), 920)):
                    try:
                        result = reminder_method.Invoke(
                            self.learn_source,
                            System.Array[System.Object]([
                                System.UInt16(species),
                                System.Byte(form),
                                System.UInt16(move_id)
                            ])
                        )
                        if result:
                            reminder_moves.add(move_id)
                    except:
                        pass

            if verbose:
                print(f"\n[3. 떠올리기 기술] {len(reminder_moves)}개")
                if reminder_moves:
                    for move_id in sorted(reminder_moves)[:10]:
                        print(f"  - {self.move_names[move_id]} (ID: {move_id})")
                    if len(reminder_moves) > 10:
                        print(f"  ... 외 {len(reminder_moves) - 10}개")

            # 4. 기술머신 기술 (PersonalInfo9SV.MachineMoves 배열 직접 정의)
            # PersonalInfo9SV.cs 121-146번 줄에서 가져옴
            machine_moves = [
                5, 36, 204, 313, 97, 189, 184, 182, 424, 422,
                423, 352, 67, 491, 512, 522, 60, 109, 168, 574,
                885, 884, 886, 451, 83, 263, 342, 332, 523, 506,
                555, 232, 129, 345, 196, 341, 317, 577, 488, 490,
                314, 500, 101, 374, 525, 474, 419, 203, 521, 241,
                240, 201, 883, 684, 473, 91, 331, 206, 280, 428,
                369, 421, 492, 706, 339, 403, 34, 7, 9, 8,
                214, 402, 486, 409, 115, 113, 350, 127, 337, 605,
                118, 447, 86, 398, 707, 156, 157, 269, 14, 776,
                191, 390, 286, 430, 399, 141, 598, 19, 285, 442,
                349, 408, 441, 164, 334, 404, 529, 261, 242, 271,
                710, 202, 396, 366, 247, 406, 446, 304, 257, 412,
                94, 484, 227, 57, 861, 53, 85, 583, 133, 347,
                270, 676, 226, 414, 179, 58, 604, 580, 678, 581,
                417, 126, 56, 59, 519, 518, 520, 528, 188, 89,
                444, 566, 416, 307, 308, 338, 200, 315, 411, 437,
                542, 433, 405, 63, 413, 394, 87, 370, 76, 434,
                796, 851, 46, 268, 114, 92, 328, 180, 356, 479,
                360, 282, 450, 162, 410, 679, 667, 333, 503, 535,
                669, 253, 264, 311, 803, 807, 812, 814, 809, 808,
                799, 802, 220, 244, 38, 283, 572, 915, 250, 330,
                916, 527, 813, 811, 482, 815, 297, 248, 797, 806,
                800, 675, 784, 319, 174, 912, 913, 914, 917, 918,
            ]

            # 각 TM 번호에 대해 GetIsLearnTM으로 확인
            for tm_index in range(len(machine_moves)):
                try:
                    if personal_info.GetIsLearnTM(tm_index):
                        move_id = machine_moves[tm_index]
                        machine_moves_set.add(move_id)
                except:
                    pass

            if verbose:
                print(f"\n[4. 기술머신] {len(machine_moves_set)}개")
                for move_id in sorted(machine_moves_set)[:10]:
                    print(f"  - {self.move_names[move_id]} (ID: {move_id})")
                if len(machine_moves_set) > 10:
                    print(f"  ... 외 {len(machine_moves_set) - 10}개")

            # 모든 기술 합치기
            all_legal_moves = levelup_moves | egg_moves | reminder_moves | machine_moves_set

            if verbose:
                print(f"\n[총 합법 기술] {len(all_legal_moves)}개")

            # 결과 정리 - 기술명 리스트 반환
            move_list = []
            for move_id in sorted(all_legal_moves):
                if 0 < move_id < len(self.move_names):
                    move_name = self.move_names[move_id]
                    if move_name and move_name not in ["(없음)", "", "????", " ", "—"]:
                        move_list.append(move_name)

            return move_list

        except Exception as e:
            if verbose:
                print(f"합법 기술 조회 실패: {e}")
                import traceback
                traceback.print_exc()
            return []

    def get_pokemon_info(self, pokemon_name: str, form: int = 0) -> Optional[Dict]:
        """
        포켓몬의 모든 정보 가져오기 (pochamps_player.py 형식에 맞춤)

        Args:
            pokemon_name: 포켓몬 이름 (한글 또는 영어)
            form: 폼 번호 (기본값: 0)

        Returns:
            포켓몬 정보 딕셔너리 또는 None
            {
                "species": str,
                "types": List[str],
                "baseStats": {"hp": int, "atk": int, ...},
                "abilities": {"0": str, "1": str, "H": str}
            }
        """
        species = self.get_species_id(pokemon_name)

        if species is None:
            return None

        return {
            "species": self.species_names[species],
            "types": self.get_types(species, form),
            "baseStats": self.get_base_stats(species, form),
            "abilities": self.get_abilities(species, form)
        }


# 전역 extractor 인스턴스 (싱글톤 패턴)
_extractor_cache = {}

def get_extractor(language: str = "en") -> PokemonDataExtractor:
    """
    PokemonDataExtractor 인스턴스를 가져오거나 생성

    Args:
        language: 언어 코드 ("ko", "en", "ja")

    Returns:
        PokemonDataExtractor 인스턴스
    """
    if language not in _extractor_cache:
        _extractor_cache[language] = PokemonDataExtractor(language=language)
    return _extractor_cache[language]


def get_pokemon_info(pokemon_name: str, language: str = "en") -> Optional[Dict]:
    """
    포켓몬 정보 가져오기 (편의 함수)

    Args:
        pokemon_name: 포켓몬 이름
        language: 언어 코드 (기본값: "en")

    Returns:
        포켓몬 정보 딕셔너리 또는 None
    """
    extractor = get_extractor(language)
    return extractor.get_pokemon_info(pokemon_name)


def get_move_info(move_name: str, language: str = "en") -> Optional[Dict]:
    """
    기술 정보 가져오기 (편의 함수)

    Args:
        move_name: 기술 이름
        language: 언어 코드 (기본값: "en")

    Returns:
        기술 정보 딕셔너리 또는 None
    """
    extractor = get_extractor(language)
    return extractor.get_move_info(move_name)


# 테스트 코드
if __name__ == "__main__":
    # 영어 테스트
    print("=" * 80)
    print("PKHeX Core Integration Test")
    print("=" * 80)

    extractor = get_extractor("en")

    # Pikachu 테스트
    pikachu_data = extractor.get_pokemon_info("Pikachu")
    if pikachu_data:
        print("\n[Pikachu]")
        print(f"  Species: {pikachu_data['species']}")
        print(f"  Types: {', '.join(pikachu_data['types'])}")
        print(f"  Base Stats: {pikachu_data['baseStats']}")
        print(f"  Abilities: {pikachu_data['abilities']}")

    # Charizard 테스트
    charizard_data = extractor.get_pokemon_info("Charizard")
    if charizard_data:
        print("\n[Charizard]")
        print(f"  Species: {charizard_data['species']}")
        print(f"  Types: {', '.join(charizard_data['types'])}")
        print(f"  Base Stats: {charizard_data['baseStats']}")
        print(f"  Abilities: {charizard_data['abilities']}")

        # 기술 목록도 출력
        moves = extractor.get_legal_moves(
            extractor.get_species_id("Charizard"),
            verbose=True
        )
        print(f"\n  Total Legal Moves: {len(moves)}")
