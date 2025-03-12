"""
智能体内存模块，用于管理智能体的历史记忆和上下文。
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from aisr.memory.base import Memory


class AgentMemory(Memory):
    """
    智能体专用的内存系统。

    用于存储智能体的交互历史，跟踪思考过程，
    并为后续提示构建提供上下文。
    """

    def __init__(self, agent_name: str):
        """
        初始化智能体内存。

        Args:
            agent_name: 智能体的名称
        """
        self.agent_name = agent_name
        self.interactions = []  # 存储智能体的交互历史
        self.metadata = {}  # 存储额外的元数据
        logging.debug(f"已初始化 {agent_name} 的智能体内存")

    def add(self, entry: Dict[str, Any]) -> None:
        """
        向内存添加新的交互记录。

        Args:
            entry: 包含交互数据的字典。应包含'input'和'output'字段，
                  可选包含'timestamp'和'metadata'。
        """
        # 确保必要的字段存在
        if "input" not in entry:
            logging.warning("添加到智能体内存的条目缺少'input'字段")
            entry["input"] = {}

        if "output" not in entry:
            logging.warning("添加到智能体内存的条目缺少'output'字段")
            entry["output"] = {}

        # 添加时间戳（如果没有提供）
        if "timestamp" not in entry:
            entry["timestamp"] = datetime.now().isoformat()

        # 添加到交互历史
        self.interactions.append(entry)
        logging.debug(f"{self.agent_name} 内存: 已添加新的交互记录")

    def get_relevant(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        检索与当前上下文相关的历史交互。

        Args:
            context: 包含检索参数的字典
                - max_items: 返回的最大项目数（默认5）
                - recency_weight: 最近项目的权重（0-1）
                - relevance_key: 在上下文中用于相关性评估的键

        Returns:
            相关交互的列表
        """
        # 获取参数
        max_items = context.get("max_items", 5)
        recency_weight = min(max(context.get("recency_weight", 0.7), 0), 1)

        # 如果没有足够的交互，返回所有
        if len(self.interactions) <= max_items:
            return self.interactions.copy()

        # 强调最近的交互
        if recency_weight > 0.9:
            # 如果最近性非常重要，只返回最近的项目
            return self.interactions[-max_items:]

        # 这里可以实现更复杂的相关性逻辑
        # 目前简单地综合考虑最近性，保留一些最近的和一些较早的交互
        recent_count = int(max_items * recency_weight)
        older_count = max_items - recent_count

        recent_items = self.interactions[-recent_count:] if recent_count > 0 else []
        older_items = self.interactions[:-recent_count][:older_count] if older_count > 0 else []

        return older_items + recent_items

    def clear(self) -> None:
        """清除所有存储的交互。"""
        self.interactions = []
        logging.debug(f"{self.agent_name} 内存: 已清除所有交互")

    def summarize(self, context: Dict[str, Any] = None) -> str:
        """
        创建交互历史的简洁摘要。

        Args:
            context: 可选的上下文参数字典

        Returns:
            内存内容的摘要字符串
        """
        if not self.interactions:
            return "没有历史交互。"

        # 创建摘要
        total_interactions = len(self.interactions)
        recent_interactions = min(3, total_interactions)

        summary = [f"{self.agent_name} 的交互历史摘要:"]
        summary.append(f"总交互数: {total_interactions}")

        if recent_interactions > 0:
            summary.append("\n最近的交互:")
            for i in range(1, recent_interactions + 1):
                interaction = self.interactions[-i]
                summary.append(f"- {interaction.get('timestamp', 'Unknown time')}: " +
                               f"输入类型: {type(interaction.get('input', {})).__name__}, " +
                               f"输出类型: {type(interaction.get('output', {})).__name__}")

        return "\n".join(summary)

    def get_last_interaction(self) -> Optional[Dict[str, Any]]:
        """获取最近的一次交互。"""
        if not self.interactions:
            return None
        return self.interactions[-1]

    def set_metadata(self, key: str, value: Any) -> None:
        """
        设置内存元数据。

        Args:
            key: 元数据键
            value: 元数据值
        """
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        获取内存元数据。

        Args:
            key: 元数据键
            default: 默认值，如果键不存在

        Returns:
            元数据值或默认值
        """
        return self.metadata.get(key, default)