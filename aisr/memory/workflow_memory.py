"""
工作流内存模块，用于管理工作流执行状态和结果。
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from aisr.memory.base import Memory


class WorkflowMemory(Memory):
    """
    工作流专用的内存系统。

    存储工作流的执行状态、中间结果和执行历史，
    为工作流的连续和迭代执行提供状态管理。
    """

    def __init__(self, workflow_name: str):
        """
        初始化工作流内存。

        Args:
            workflow_name: 工作流的名称
        """
        self.workflow_name = workflow_name
        self.steps = {}  # 存储步骤结果 {step_name: [result1, result2, ...]}
        self.state = {}  # 存储工作流状态
        self.history = []  # 存储执行历史
        logging.debug(f"已初始化 {workflow_name} 的工作流内存")

    def add(self, entry: Dict[str, Any]) -> None:
        """
        添加工作流步骤结果或状态更新。

        Args:
            entry: 要添加的数据字典，必须包含'type'键，值为'step_result'或'state_update'
                  对于'step_result'类型，还需要'step_name'和'result'键
                  对于'state_update'类型，还需要'key'和'value'键
        """
        entry_type = entry.get("type")
        timestamp = entry.get("timestamp", datetime.now().isoformat())

        # 添加执行历史
        history_entry = {
            "timestamp": timestamp,
            "type": entry_type
        }

        if entry_type == "step_result":
            step_name = entry.get("step_name")
            result = entry.get("result")

            if not step_name:
                logging.warning("工作流内存: 步骤结果缺少步骤名称")
                return

            # 初始化步骤结果列表（如果不存在）
            if step_name not in self.steps:
                self.steps[step_name] = []

            # 添加结果
            self.steps[step_name].append(result)

            # 更新历史条目
            history_entry.update({
                "step_name": step_name,
                "result_summary": self._summarize_result(result)
            })

            logging.debug(f"{self.workflow_name} 内存: 已添加 '{step_name}' 步骤的结果")

        elif entry_type == "state_update":
            key = entry.get("key")
            value = entry.get("value")

            if not key:
                logging.warning("工作流内存: 状态更新缺少键")
                return

            # 更新状态
            self.state[key] = value

            # 更新历史条目
            history_entry.update({
                "state_key": key,
                "value_summary": str(value)[:100] + "..." if isinstance(value, str) and len(str(value)) > 100 else str(
                    value)
            })

            logging.debug(f"{self.workflow_name} 内存: 已更新状态 '{key}'")

        else:
            logging.warning(f"工作流内存: 未知的条目类型: {entry_type}")
            return

        # 添加到历史
        self.history.append(history_entry)

    def get_relevant(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        检索与上下文相关的内存内容。

        Args:
            context: 包含检索参数的字典
                - step_name: 特定步骤名称（可选）
                - state_keys: 需要的状态键列表（可选）
                - include_history: 是否包含历史（默认False）

        Returns:
            相关内存条目的列表
        """
        result = []

        # 检索特定步骤的结果
        step_name = context.get("step_name")
        if step_name and step_name in self.steps:
            for step_result in self.steps[step_name]:
                result.append({
                    "type": "step_result",
                    "step_name": step_name,
                    "result": step_result
                })

        # 检索所有步骤的最新结果
        if context.get("latest_steps", False):
            for step_name, results in self.steps.items():
                if results:  # 如果有结果
                    result.append({
                        "type": "step_result",
                        "step_name": step_name,
                        "result": results[-1]  # 最新结果
                    })

        # 检索请求的状态键
        state_keys = context.get("state_keys", [])
        if state_keys:
            for key in state_keys:
                if key in self.state:
                    result.append({
                        "type": "state",
                        "key": key,
                        "value": self.state[key]
                    })

        # 可选包含历史
        if context.get("include_history", False):
            result.append({
                "type": "history",
                "entries": self.history
            })

        return result

    def clear(self) -> None:
        """清除所有存储的内存。"""
        self.steps = {}
        self.state = {}
        self.history = []
        logging.debug(f"{self.workflow_name} 内存: 已清除所有数据")

    def summarize(self, context: Dict[str, Any] = None) -> str:
        """
        创建工作流内存内容的摘要。

        Args:
            context: 可选的上下文参数

        Returns:
            内存内容的摘要字符串
        """
        summary_parts = [f"{self.workflow_name} 工作流内存摘要:"]

        # 步骤摘要
        if self.steps:
            step_summary = []
            for step_name, results in self.steps.items():
                step_summary.append(f"- {step_name}: {len(results)} 个结果")
            summary_parts.append("步骤结果:\n" + "\n".join(step_summary))
        else:
            summary_parts.append("步骤结果: 无")

        # 状态摘要
        if self.state:
            state_summary = []
            for key, value in self.state.items():
                value_str = str(value)
                if len(value_str) > 50:
                    value_str = value_str[:47] + "..."
                state_summary.append(f"- {key}: {value_str}")
            summary_parts.append("工作流状态:\n" + "\n".join(state_summary))
        else:
            summary_parts.append("工作流状态: 无")

        # 历史摘要
        if self.history:
            summary_parts.append(f"执行历史: {len(self.history)} 个条目")
        else:
            summary_parts.append("执行历史: 无")

        return "\n\n".join(summary_parts)

    def _summarize_result(self, result: Any) -> str:
        """创建结果的简短摘要。"""
        if isinstance(result, dict):
            # 提取关键信息
            keys = list(result.keys())
            return f"字典 ({len(keys)} 个键: {', '.join(keys[:3])}{'...' if len(keys) > 3 else ''})"
        elif isinstance(result, list):
            return f"列表 ({len(result)} 项)"
        elif isinstance(result, str):
            return result[:50] + "..." if len(result) > 50 else result
        else:
            return str(result)

    def save_result(self, step_name: str, result: Any) -> None:
        """
        保存步骤结果的便捷方法。

        Args:
            step_name: 步骤名称
            result: 步骤结果
        """
        self.add({
            "type": "step_result",
            "step_name": step_name,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })

    def update_state(self, key: str, value: Any) -> None:
        """
        更新工作流状态的便捷方法。

        Args:
            key: 状态键
            value: 状态值
        """
        self.add({
            "type": "state_update",
            "key": key,
            "value": value,
            "timestamp": datetime.now().isoformat()
        })

    def get_latest_result(self, step_name: str) -> Optional[Any]:
        """
        获取步骤的最新结果。

        Args:
            step_name: 步骤名称

        Returns:
            最新结果，如果步骤不存在则返回None
        """
        if step_name not in self.steps or not self.steps[step_name]:
            return None
        return self.steps[step_name][-1]

    def get_all_results(self, step_name: str) -> List[Any]:
        """
        获取步骤的所有结果。

        Args:
            step_name: 步骤名称

        Returns:
            结果列表，如果步骤不存在则返回空列表
        """
        return self.steps.get(step_name, [])

    def get_state(self, key: str, default: Any = None) -> Any:
        """
        获取工作流状态值。

        Args:
            key: 状态键
            default: 如果键不存在，返回的默认值

        Returns:
            状态值或默认值
        """
        return self.state.get(key, default)