import abc
from typing import Dict, Any

class Memory(abc.ABC):
    """
    AISR内存系统的抽象基类。

    负责在研究过程中存储、检索和管理信息状态。
    """

    @abc.abstractmethod
    def add(self, entry: Dict[str, Any]) -> None:
        """
        向内存添加新条目。

        Args:
            entry: 包含要存储数据的字典
        """
        pass

    @abc.abstractmethod
    def get_relevant(self, context: Dict[str, Any]) -> list[Dict[str, Any]]:
        """
        基于提供的上下文检索相关内存。

        Args:
            context: 包含指导内存检索的参数的字典

        Returns:
            相关内存条目的列表
        """
        pass

    @abc.abstractmethod
    def clear(self) -> None:
        """清除所有存储的内存"""
        pass

    def summarize(self, context: Dict[str, Any] = None) -> str:
        """创建相关内存的简洁摘要"""
        return ""  # 默认实现，子类应重写
