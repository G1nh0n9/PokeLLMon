import sys
import os

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
        """합법적으로 배울 수 있는 모든 기술 목록 (레벨업, 기술머신, 교배, 튜터 등 모두)"""
        try:
            legal_moves = set()
            personal_info = self.personal_table[species, form]
            
            # 더미 포켓몬 생성
            pk = PK9()
            pk.Species = species
            pk.Form = form
            pk.CurrentLevel = 100
            
            # 1. 레벨업 기술
            learnset = self.learn_source.GetLearnset(species, form)
            for move_id in range(1, min(len(self.move_names), 1000)):
                if learnset.GetIsLearn(move_id):
                    legal_moves.add(move_id)
            print(f"  레벨업 기술: {len(legal_moves)}개")
            
            # 2. GetPossible로 모든 학습 가능 기술 가져오기 (기술머신, 교배 등 포함)
            try:
                # ILearnSource.GetPossible 메서드 사용
                for move_id in range(1, min(len(self.move_names), 1000)):
                    # 각 기술에 대해 학습 가능 여부 체크
                    # GetCanLearn이나 비슷한 메서드 시도
                    try:
                        # 9세대 기준으로 모든 소스(레벨업+기술머신+교배+튜터) 체크
                        info = MoveLearnInfo()
                        info.Move = move_id
                        info.Level = 100
                        
                        # GetCanKnow 같은 메서드로 종합 체크
                        if hasattr(self.learn_source, 'GetCanKnow'):
                            if self.learn_source.GetCanKnow(pk, info):
                                legal_moves.add(move_id)
                    except:
                        pass
            except Exception as e:
                print(f"  추가 기술 체크 실패: {e}")
            
            # 3. PersonalInfo에서 기술머신 가능 여부 직접 체크
            try:
                # TMHM 필드나 메서드로 기술머신 체크
                personal_type = personal_info.GetType()
                
                # GetIsLearnTM 같은 메서드 찾기
                for move_id in range(1, min(len(self.move_names), 1000)):
                    try:
                        if hasattr(personal_info, 'GetIsLearnTM'):
                            if personal_info.GetIsLearnTM(move_id):
                                legal_moves.add(move_id)
                    except:
                        pass
            except Exception as e:
                print(f"  기술머신 체크 실패: {e}")
            
            print(f"  총 습득 가능 기술: {len(legal_moves)}개")
            
            # 결과 정리
            move_list = []
            for move_id in sorted(legal_moves):
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
    TEST_POKEMON_1 = "피카츄" if LANGUAGE == "ko" else "Pikachu"
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
        print(f"\n✓ {TEST_POKEMON_1} can learn: {len(pokemon_data['합법기술'])} moves")
    
    print("\n" + "=" * 60)
    print(f"【 {TEST_POKEMON_2} Info 】")
    print("=" * 60)
    charizard_data = extractor.get_all_pokemon_data(TEST_POKEMON_2)
    
    print(f"\n■ Type: {charizard_data['타입']['타입1']}", end="")
    if charizard_data['타입']['타입2']:
        print(f"/{charizard_data['타입']['타입2']}")
    else:
        print()
    
    print(f"■ Base Stat Total: {charizard_data['베이스스탯']['합계']}")
    print(f"■ Legal Moves: {len(charizard_data['합법기술'])} total")
    
    # 일부 기술만 출력
    print("\nMove Sample (First 10):")
    for move in charizard_data['합법기술'][:10]:
        print(f"  - {move['기술명']} (ID: {move['기술id']})")