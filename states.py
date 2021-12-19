from enum import Enum

class Yomite(Enum):
    AI = 1
    PLAYER = 2

class States:
    def __init__(self, yomite):
        self.yomite = yomite
    
    def change_yomite(self):
        if self.yomite == Yomite.AI:
            self.yomite = Yomite.PLAYER
        elif self.yomite == Yomite.PLAYER:
            self.yomite = Yomite.AI
        
