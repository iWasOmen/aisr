import abc
from typing import Dict, Any

class Component(abc.ABC):
    """
    所有AISR系统组件的抽象基类。

    作为Agent、Workflow和Tool的基础，提供统一执行接口。
    """

    @abc.abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行组件的功能。

        Args:
            context: 包含执行所需输入数据和参数的字典

        Returns:
            包含执行结果的字典
        """
        pass

    def get_id(self) -> str:
        """获取组件的唯一标识"""
        return self.__class__.__name__
