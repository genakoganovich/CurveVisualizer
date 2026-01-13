from typing import List, Dict
import numpy as np


class ActorManager:
    """Управляет множеством акторов"""

    def __init__(self):
        self.actors: List = []
        self._actor_dict: Dict[str, List] = {}

    def add_actor(self, actor, name: str = None):
        """Добавить актор"""
        self.actors.append(actor)

        # ★ Получаем тип актора безопасно
        # Проверяем есть ли arrow_type (для ArrowActor)
        if hasattr(actor, 'arrow_type'):
            actor_type = actor.arrow_type
        # Иначе используем имя класса
        else:
            actor_type = actor.__class__.__name__

        # ★ ИСПРАВЛЕНИЕ: Инициализируем словарь если ключа нет
        if actor_type not in self._actor_dict:
            self._actor_dict[actor_type] = []

        self._actor_dict[actor_type].append(actor)

        print(f"✅ Добавлен актор ({actor_type})")

    def remove_actor(self, actor):
        """Удалить актор"""
        self.actors.remove(actor)

        # ★ Удаляем из словаря
        if hasattr(actor, 'arrow_type'):
            actor_type = actor.arrow_type
        else:
            actor_type = actor.__class__.__name__

        if actor_type in self._actor_dict and actor in self._actor_dict[actor_type]:
            self._actor_dict[actor_type].remove(actor)

    def update_all(self, plotter, t: float):
        """Обновить все акторы"""
        for actor in self.actors:
            actor.update(plotter, t)

    def get_by_type(self, actor_type: str):
        """Получить акторы по типу"""
        return self._actor_dict.get(actor_type, [])

    def clear(self):
        """Очистить все акторы"""
        self.actors.clear()
        self._actor_dict.clear()

    def __len__(self):
        return len(self.actors)

    def __repr__(self):
        """Информация об акторах"""
        info = "ActorManager:\n"
        for actor_type, actors_list in self._actor_dict.items():
            info += f"  {actor_type}: {len(actors_list)}\n"
        return info