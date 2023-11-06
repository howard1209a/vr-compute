import random

from base_station_choose_strategy import BSChooseStrategy


class BSChooseRandomStrategy(BSChooseStrategy):
    def choose(self, bss, task):
        if len(bss) == 0:
            return None
        return random.choice(bss)
