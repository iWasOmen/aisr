"""
内存管理系统，负责管理AISR系统中的各类记忆。
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from aisr.memory.base import Memory
from aisr.memory.global_memory import GlobalMemory
from aisr.memory.agent_memory import AgentMemory
from aisr.memory.workflow_memory import WorkflowMemory


class MemoryManager:
    """
    AISR系统的中央内存管理器。

    负责创建和管理各种内存视图，包括全局内存、智能体内存和工作流内存。
    """

    def __init__(self):
        """初始化内存管理器。"""
        # 全局内存存储整个研究会话的状态
        self.global_memory = GlobalMemory()

        # 各组件的专用内存
        self.component_memories: Dict[str, Memory] = {}

        logging.info("内存管理器初始化完成")

    def get_memory_view(self, component_name: str) -> Memory:
        """
        获取或创建组件的内存视图。

        Args:
            component_name: 组件名称

        Returns:
            组件的内存视图
        """
        if component_name not in self.component_memories:
            # 根据组件名称确定内存类型
            if "_agent" in component_name:
                self.component_memories[component_name] = AgentMemory(component_name)
            elif "_workflow" in component_name:
                self.component_memories[component_name] = WorkflowMemory(component_name)
            else:
                # 默认为工作流内存
                self.component_memories[component_name] = WorkflowMemory(component_name)

            logging.debug(f"为组件 '{component_name}' 创建了新的内存视图")

        return self.component_memories[component_name]

    def save_global_state(self, key: str, value: Any) -> None:
        """
        保存状态到全局内存。

        Args:
            key: 状态键
            value: 状态值
        """
        self.global_memory.set(key, value)

    def get_global_state(self, key: str, default: Any = None) -> Any:
        """
        从全局内存获取状态。

        Args:
            key: 状态键
            default: 如果键不存在，返回的默认值

        Returns:
            关联的状态值，如果不存在则返回默认值
        """
        return self.global_memory.get(key, default)

    def clear_research_state(self) -> None:
        """清除当前研究状态，为新研究做准备。"""
        # 清除全局内存中的研究相关状态
        self.global_memory.clear_research_data()

        # 清除所有组件内存
        for memory in self.component_memories.values():
            memory.clear()

        # 记录新的研究开始
        self.global_memory.set("research_start_time", datetime.now().isoformat())
        logging.info("已清除研究状态，准备新的研究")

    def record_research_step(self, step_name: str, data: Dict[str, Any]) -> None:
        """
        记录研究步骤到历史记录。

        Args:
            step_name: 步骤名称
            data: 步骤数据
        """
        # 获取现有历史记录
        history = self.global_memory.get("research_history", [])

        # 添加新步骤
        step_record = {
            "step": step_name,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }

        history.append(step_record)
        self.global_memory.set("research_history", history)