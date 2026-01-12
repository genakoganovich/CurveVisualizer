from typing import List, Dict



class ActorManager:
    """Управляет множеством акторов"""

    def __init__(self):
        self.actors: List = []
        self._actor_dict: Dict[str, List] = {
            "tangent": [],
            "normal": [],
            "binormal": [],
        }

    def add_actor(self, actor, name: str = None):
        """Добавить актор"""
        self.actors.append(actor)

        # Добавить в словарь по типу
        arrow_type = actor.arrow_type
        self._actor_dict[arrow_type].append(actor)

        print(f"✅ Добавлена стрелка ({arrow_type})")

    def remove_actor(self, actor):
        """Удалить актор"""
        self.actors.remove(actor)
        self._actor_dict[actor.arrow_type].remove(actor)

    def update_all(self, plotter, t: float):
        """Обновить все акторы"""
        for actor in self.actors:
            actor.update(plotter, t)

    def get_by_type(self, arrow_type: str):
        """Получить акторы по типу"""
        return self._actor_dict.get(arrow_type, [])

    def clear(self):
        """Очистить все акторы"""
        self.actors.clear()
        for key in self._actor_dict:
            self._actor_dict[key].clear()

    def __len__(self):
        return len(self.actors)