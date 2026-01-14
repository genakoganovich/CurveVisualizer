from enum import Enum


class AnimationMode(Enum):
    """Режимы анимации касательной"""

    # 1. Касательная движется плавно
    CONTINUOUS = "continuous"

    # 2. Касательная движется с шагом (одна касательная, скачет)
    STEPPED = "stepped"

    # 3. Добавляются новые касательные с шагом (накапливаются)
    ACCUMULATED = "accumulated"