import abc
from typing import Dict, Any
from aisr.core.base import Component

class Workflow(Component):
    """
    AISR中所有工作流组件的抽象基类。

    工作流负责编排执行流程，协调Agent和Tool完成特定任务。
    """

    def __init__(self, router, memory):
        """
        初始化工作流。

        Args:
            router: 用于调用其他组件的路由器
            memory: 工作流的内存系统
        """
        self.router = router
        self.memory = memory

    @abc.abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行工作流逻辑"""
        pass

    def call_component(self, function: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        通过路由器调用另一个组件。

        Args:
            function: 要调用的函数的字符串标识符
            parameters: 要传递的参数字典

        Returns:
            被调用组件的结果
        """
        return self.router.route({"function": function, "parameters": parameters})
