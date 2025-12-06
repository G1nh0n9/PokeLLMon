import sys
import os
import io

# UTF-8 인코딩 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# pythonnet을 .NET Core 런타임으로 설정 (반드시 import clr 전에!)
os.environ["DOTNET_ROOT"] = r"C:\Program Files\dotnet"

from pythonnet import set_runtime
# 현재 디렉토리의 runtime config 파일 지정
current_dir = os.path.dirname(os.path.abspath(__file__))
runtime_config = os.path.join(current_dir, "python.runtimeconfig.json")
set_runtime("coreclr", runtime_config=runtime_config)

import clr

# PKHeX DLL 경로 추가 (현재 파일 기준)
current_dir = os.path.dirname(os.path.abspath(__file__))
dll_path = os.path.join(current_dir, "PKHeX.Core.dll")

# DLL 파일 존재 확인
if not os.path.exists(dll_path):
    raise FileNotFoundError(f"PKHeX.Core.dll not found at: {dll_path}")

print(f"Loading DLL from: {dll_path}")

# DLL 로드 (절대 경로 사용)
try:
    clr.AddReference(dll_path)
    print("DLL loaded successfully")
except Exception as e:
    print(f"Failed to load DLL: {e}")
    raise

# 네임스페이스 import
try:
    from PKHeX.Core import *
    import System
    from System.Reflection import BindingFlags
    print("PKHeX.Core namespace imported successfully")
except Exception as e:
    print(f"Failed to import PKHeX.Core: {e}")
    print("Available assemblies:")
    for asm in clr.ListAssemblies(True):
        if "PKHeX" in asm:
            print(f"  {asm}")
    raise

class PokemonDataExtractor:
    def __init__(self, game_version=None, language="ko"):
        # 기본값은 9세대 SV
        if game_version is None:
            game_version = GameVersion.SV
        
        # GameData를 통해 Personal과 LearnSource 가져오기
        self.game_version = game_version
        self.personal_table = GameData.GetPersonal(game_version)
        self.learn_source = GameData.GetLearnSource(game_version)
        
        # 언어 설정 (ko=한국어, en=영어, ja=일본어 등)
        self.language = language
        self.species_names = Util.GetSpeciesList(language)
        self.move_names = Util.GetMovesList(language)
        self.ability_names = Util.GetAbilitiesList(language)
        self.item_names = Util.GetItemsList(language)
        self.type_names = Util.GetTypesList(language)
    
    def get_species_id(self, pokemon_name):
        """포켓몬 이름으로 종족 번호 찾기"""
        for i, name in enumerate(self.species_names):
            if name == pokemon_name:
                return i
        return None
    
    def get_base_stats(self, species, form=0):
        """베이스 스탯 가져오기"""
        personal_info = self.personal_table[species, form]
        
        return {
            "HP": personal_info.HP,
            "공격": personal_info.ATK,
            "방어": personal_info.DEF,
            "특수공격": personal_info.SPA,
            "특수방어": personal_info.SPD,
            "스피드": personal_info.SPE,
            "합계": personal_info.HP + personal_info.ATK + personal_info.DEF + 
                   personal_info.SPA + personal_info.SPD + personal_info.SPE
        }
    
    def get_abilities(self, species, form=0):
        """특성 목록 가져오기"""
        personal_info = self.personal_table[species, form]
        
        abilities = {
            "특성1": {
                "id": personal_info.Ability1,
                "이름": self.ability_names[personal_info.Ability1]
            },
            "특성2": {
                "id": personal_info.Ability2,
                "이름": self.ability_names[personal_info.Ability2]
            },
            "숨겨진특성": {
                "id": personal_info.AbilityH,
                "이름": self.ability_names[personal_info.AbilityH]
            }
        }
        
        return abilities
    
    def get_types(self, species, form=0):
        """타입 가져오기"""
        personal_info = self.personal_table[species, form]
        
        type1 = self.type_names[personal_info.Type1]
        type2 = self.type_names[personal_info.Type2] if personal_info.Type1 != personal_info.Type2 else None
        
        return {"타입1": type1, "타입2": type2}
    
    def get_legal_moves(self, species, form=0):
        """합법적으로 배울 수 있는 모든 기술 목록 (레벨업, 기술머신, 알 기술, 떠올리기)"""
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

            print(f"\n[1. 레벨업 기술] {len(levelup_moves)}개")
            for move_id in sorted(levelup_moves)[:10]:  # 처음 10개만 출력
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

            print(f"\n[4. 기술머신] {len(machine_moves_set)}개")
            for move_id in sorted(machine_moves_set)[:10]:
                print(f"  - {self.move_names[move_id]} (ID: {move_id})")
            if len(machine_moves_set) > 10:
                print(f"  ... 외 {len(machine_moves_set) - 10}개")

            # 모든 기술 합치기
            all_legal_moves = levelup_moves | egg_moves | reminder_moves | machine_moves_set
            print(f"\n[총 합법 기술] {len(all_legal_moves)}개")

            # 결과 정리
            move_list = []
            for move_id in sorted(all_legal_moves):
                if 0 < move_id < len(self.move_names):
                    move_name = self.move_names[move_id]
                    if move_name and move_name not in ["(없음)", "", "????", " ", "—"]:
                        move_list.append({
                            "기술id": move_id,
                            "기술명": move_name
                        })

            return move_list

        except Exception as e:
            print(f"합법 기술 조회 실패: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_all_pokemon_data(self, pokemon_name, form=0):
        """포켓몬의 모든 정보 가져오기"""
        species = self.get_species_id(pokemon_name)
        
        if species is None:
            return {"error": f"'{pokemon_name}' 포켓몬을 찾을 수 없습니다."}
        
        data = {
            "이름": pokemon_name,
            "종족번호": species,
            "폼": form,
            "타입": self.get_types(species, form),
            "베이스스탯": self.get_base_stats(species, form),
            "특성": self.get_abilities(species, form),
            "합법기술": self.get_legal_moves(species, form)
        }
        
        return data


# 사용 예시
if __name__ == "__main__":
    # ========== 설정 ==========
    LANGUAGE = "ko"  # "ko" = 한국어, "en" = 영어, "ja" = 일본어
    # TEST_POKEMON_1 = "피카츄" if LANGUAGE == "ko" else "Pikachu"
    TEST_POKEMON_1 = "리자몽" if LANGUAGE == "ko" else "Charizard"
    TEST_POKEMON_2 = "리자몽" if LANGUAGE == "ko" else "Charizard"
    # ==========================
    
    extractor = PokemonDataExtractor(language=LANGUAGE)
    
    print("=" * 60)
    print(f"PKHeX.Core - Legal Moves Extraction Test ({LANGUAGE.upper()})")
    print("=" * 60)
    
    # 첫 번째 포켓몬 정보 가져오기
    pokemon_data = extractor.get_all_pokemon_data(TEST_POKEMON_1)
    
    # 결과 출력
    import json
    print(f"\n【 {TEST_POKEMON_1} Full Info 】")
    print(json.dumps(pokemon_data, ensure_ascii=False, indent=2))
    
    # 합법 기술 개수 출력
    if "합법기술" in pokemon_data:
        print(f"\n[OK] {TEST_POKEMON_1} can learn: {len(pokemon_data['합법기술'])} moves")
    
    print("\n" + "=" * 60)
    # print(f"【 {TEST_POKEMON_2} Info 】")
    # print("=" * 60)
    # charizard_data = extractor.get_all_pokemon_data(TEST_POKEMON_2)
    #
    # print(f"\n[Type] {charizard_data['타입']['타입1']}", end="")
    # if charizard_data['타입']['타입2']:
    #     print(f"/{charizard_data['타입']['타입2']}")
    # else:
    #     print()
    #
    # print(f"[Base Stat Total] {charizard_data['베이스스탯']['합계']}")
    # print(f"[Legal Moves] {len(charizard_data['합법기술'])} total")
    #
    # # 일부 기술만 출력
    # print("\nMove Sample (First 10):")
    # for move in charizard_data['합법기술'][:10]:
    #     print(f"  - {move['기술명']} (ID: {move['기술id']})")
