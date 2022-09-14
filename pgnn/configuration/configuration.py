from typing import Optional


class Configuration():
    def __init__(self):
        self.config: list[str] = []
        self.custom_name: str = '<default>'
        self.load: Optional[str] = None
        
        