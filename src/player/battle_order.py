from dataclasses import dataclass
from typing import Any, List, Optional, Union

from src.environment.double_battle import DoubleBattle
from src.environment.move import Move
from src.environment.pokemon import Pokemon


@dataclass
class BattleOrder:
    order: Optional[Union[Move, Pokemon, str]]
    mega: bool = False
    z_move: bool = False
    dynamax: bool = False
    terastallize: bool = False
    move_target: int = DoubleBattle.EMPTY_TARGET_POSITION

    DEFAULT_ORDER = "/choose default"

    def __str__(self) -> str:
        return self.message

    @property
    def message(self) -> str:
        if isinstance(self.order, Move):
            if self.order.id == "recharge":
                return "/choose move 1"

            message = f"/choose move {self.order.id}"
            if self.mega:
                message += " mega"
            elif self.z_move:
                message += " zmove"
            elif self.dynamax:
                message += " dynamax"
            elif self.terastallize:
                message += " terastallize"

            if self.move_target != DoubleBattle.EMPTY_TARGET_POSITION:
                message += f" {self.move_target}"
            return message
        elif isinstance(self.order, Pokemon):
            return f"/choose switch {self.order.species}"
        elif isinstance(self.order, str):
            return self.order
        else:
            return ""


class DefaultBattleOrder(BattleOrder):
    def __init__(self, *args: Any, **kwargs: Any):
        pass

    @property
    def message(self) -> str:
        return self.DEFAULT_ORDER


@dataclass
class DoubleBattleOrder(BattleOrder):
    def __init__(
        self,
        first_order: Optional[BattleOrder] = None,
        second_order: Optional[BattleOrder] = None,
    ):
        self.first_order = first_order
        self.second_order = second_order

    @property
    def message(self) -> str:
        if self.first_order and self.second_order:
            return (
                self.first_order.message
                + ", "
                + self.second_order.message.replace("/choose ", "")
            )
        elif self.first_order:
            return self.first_order.message + ", default"
        elif self.second_order:
            return self.second_order.message + ", default"
        else:
            return self.DEFAULT_ORDER

    @staticmethod
    def join_orders(first_orders: List[BattleOrder], second_orders: List[BattleOrder]):
        """Join two lists of orders into DoubleBattleOrder combinations.
        
        Filters out invalid combinations:
        - Both using mega/z-move/dynamax/terastallize
        - Both switching to the same Pokemon
        """
        def is_switch_order(order: BattleOrder) -> bool:
            """Check if this order is a switch (Pokemon object, not Move)."""
            # Pokemon is already imported at module level
            return isinstance(order.order, Pokemon)
        
        def get_switch_species(order: BattleOrder) -> Optional[str]:
            """Get species name if this is a switch order."""
            if is_switch_order(order):
                return order.order.species
            return None
        
        if first_orders and second_orders:
            orders = []
            for first_order in first_orders:
                for second_order in second_orders:
                    # Skip invalid combinations
                    if first_order.mega and second_order.mega:
                        continue
                    if first_order.z_move and second_order.z_move:
                        continue
                    if first_order.dynamax and second_order.dynamax:
                        continue
                    if first_order.terastallize and second_order.terastallize:
                        continue
                    
                    # Skip if both orders reference the exact same object
                    if first_order.order is second_order.order:
                        continue
                    
                    # Skip if both orders are switches to the same Pokemon
                    # This prevents "Can't switch to an active Pokemon" error
                    first_species = get_switch_species(first_order)
                    second_species = get_switch_species(second_order)
                    if first_species and second_species and first_species == second_species:
                        continue
                    
                    orders.append(DoubleBattleOrder(first_order=first_order, second_order=second_order))
            
            if orders:
                return orders
        elif first_orders:
            return [DoubleBattleOrder(first_order=order) for order in first_orders]
        elif second_orders:
            return [DoubleBattleOrder(first_order=order) for order in second_orders]
        return [DefaultBattleOrder()]


class ForfeitBattleOrder(BattleOrder):
    def __init__(self, *args: Any, **kwargs: Any):
        pass

    @property
    def message(self) -> str:
        return "/forfeit"
