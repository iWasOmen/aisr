import abc
from typing import Dict, Any
from aisr.core.base import Component

class Agent(Component):
    """
    AISR中所有LLM驱动智能体的抽象基类。

    智能体利用LLM能力根据提供的上下文生成决策、洞察和分析。
    """

    def __init__(self, llm_provider, memory):
        """
        初始化智能体。

        Args:
            llm_provider: LLM服务提供者
            memory: 智能体的内存系统
        """
        self.llm = llm_provider
        self.memory = memory

    @abc.abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行智能体的推理任务"""
        pass

    @abc.abstractmethod
    def build_prompt(self, context: Dict[str, Any]) -> str:
        """基于上下文和内存为LLM构建提示"""
        pass

    @abc.abstractmethod
    def parse_response(self, response: str) -> Dict[str, Any]:
        """将LLM响应解析为结构化输出"""
        pass
