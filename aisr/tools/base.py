import abc
from typing import Dict, Any
from aisr.core.base import Component

class Tool(Component):
    """
    AISR中所有工具的抽象基类。

    工具是执行特定功能的组件，如网页搜索、爬取、数据处理等。
    """

    @abc.abstractmethod
    def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """执行工具的功能"""
        pass

    @abc.abstractmethod
    def get_description(self) -> str:
        """获取此工具在LLM提示中使用的描述"""
        pass

    def is_available(self) -> bool:
        """检查此工具当前是否可用"""
        return True  # 默认实现，子类应根据需要重写
