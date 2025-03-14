"""
任务规划代理模块，负责将用户查询分解为结构化的子任务列表。

作为外层循环的核心组件，负责研究方向规划和任务分解，
可基于前序任务结果和洞察进行重规划。
"""

import logging
from datetime import datetime
from typing import Dict, Any, List
import json
import re

from aisr.agents.base import Agent
from aisr.utils import logging_utils
logger = logging_utils.get_logger(__name__, color="green")

class TaskPlanAgent(Agent):
    """
    任务规划代理，负责将研究查询分解为子任务列表。

    输入：
    - 用户查询
    - 可选的任务规划历史（包含前序子答案、未执行计划及洞察）

    输出：
    - 结构化的子任务列表
    """

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行任务规划，将查询分解为子任务列表。

        Args:
            context: 包含执行所需输入的字典
                - query: 用户查询
                - previous_plans: 可选，前序任务计划列表
                - previous_sub_answers: 可选，前序子答案字典
                - unexecuted_plan: 可选，上一轮未执行的计划
                - plan_insights: 可选，对计划的洞察

        Returns:
            包含子任务列表的字典
        """
        logger.info(f"===TaskPlanAgent===")
        query = context.get("query")
        if not query:
            raise ValueError("任务规划必须提供查询")

        # 构建提示
        prompt = self.build_prompt(context)

        # 调用LLM生成任务规划
        functions = [{
            "name": "task_planning",
            "description": "规划研究任务并分解为子任务列表",
            "parameters": {
                "type": "object",
                "properties": {
                    "sub_tasks": {
                        "type": "array",
                        "description": "子任务列表，按执行顺序排列",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string", "description": "任务简短标题"},
                                "description": {"type": "string", "description": "任务详细描述"}
                            },
                            "required": ["title", "description"]
                        }
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "任务分解的推理过程"
                    }
                },
                "required": ["sub_tasks", "reasoning"]
            }
        }]

        result = self.llm.generate_with_function_calling(prompt, functions)

        if "name" not in result or result["name"] != "task_planning":
            # 处理LLM未返回预期函数调用的情况
            logging.error("LLM未返回任务规划函数调用")
            error_response = {
                "error": "生成任务规划失败",
                "sub_tasks": []
            }
            return error_response

        logger.info(f"生成sub_tasks:")
        logger.info(result["arguments"].get("sub_tasks", ""))
        logger.info(f"生成reasoning:")
        logger.info(result["arguments"].get("reasoning", ""))
        # 解析任务规划结果
        plan_result = result["arguments"]
        # 为每个子任务添加ID
        self._add_task_ids(plan_result)


        return plan_result

    def _add_task_ids(self, plan_result: Dict[str, Any]) -> None:
        """
        为子任务添加自动生成的ID。

        Args:
            plan_result: 任务规划结果

        Returns:
            None (直接修改plan_result)
        """
        sub_tasks = plan_result.get("sub_tasks", [])

        for i, task in enumerate(sub_tasks):
            # 使用标题生成ID
            title = task.get("title", f"task_{i + 1}")
            task_id = self._generate_id_from_title(title, i + 1)
            task["id"] = task_id

    def _generate_id_from_title(self, title: str, index: int) -> str:
        """
        从任务标题生成简短的ID。

        Args:
            title: 任务标题
            index: 任务索引

        Returns:
            生成的ID
        """
        # 移除标点符号，转换为小写，替换空格为下划线
        clean_title = re.sub(r'[^\w\s]', '', title.lower())
        clean_title = re.sub(r'\s+', '_', clean_title)

        # 取前3个词（最多）
        words = clean_title.split('_')
        short_title = '_'.join(words[:3])

        # 添加索引确保唯一性
        return f"task_{index}_{short_title}"

    def build_prompt(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        构建LLM提示。

        Args:
            context: 输入上下文

        Returns:
            LLM提示消息列表
        """
        messages = []

        # 系统提示
        system_prompt = self._get_system_prompt()
        messages.append({"role": "system", "content": system_prompt})

        # 用户查询
        query = context.get("query", "")
        user_prompt = f"我需要研究以下问题:\n\n{query}\n\n请为我规划研究任务。"

        # 添加历史信息（如果有）
        history_context = self._format_history_context(context)
        if history_context:
            user_prompt += history_context

        messages.append({"role": "user", "content": user_prompt})

        return messages

    def _get_system_prompt(self) -> str:
        """
        获取系统提示。

        Returns:
            系统提示字符串
        """
        formatted_date = datetime.now().strftime("%Y-%m-%d")
        return f"""now date:{formatted_date}\n你是一位资深研究规划专家，擅长将复杂查询分解为结构化的研究子任务。

你的职责是：
1. 分析用户的研究查询
2. 将查询分解为清晰、可执行的子任务
3. 按照合理的执行顺序排列子任务
4. 考虑任务间的依赖关系

规划思路：
- 确保子任务覆盖查询的所有关键方面
- 子任务应具体明确，便于搜索和研究
- 适当考虑任务的粒度，既不要过于宽泛也不要过于细碎
- 重要且基础的任务应放在前面
- 考虑渐进式研究策略，从基础信息到深入分析
- 如果用户提供了前序计划和洞察，应充分利用这些信息进行调整

请注意：
- 子任务数量应根据查询复杂度自行判断，一般为2-4个
- 每个子任务必须包含标题和详细描述，标题应当准确概括该任务的目标，包含所有必要实体
- 提供详细的推理过程，解释你的任务分解逻辑
- 子任务应当是针对研究阶段的，如果是根据已有研究结果进行总结、计算等阶段，不属于研究子任务

你的输出将直接用于指导自动化研究系统执行查询，因此请保持清晰和结构化。"""

    def _format_history_context(self, context: Dict[str, Any]) -> str:
        """
        格式化历史上下文信息。

        Args:
            context: 上下文

        Returns:
            格式化的历史上下文字符串，如果没有历史信息则返回空字符串
        """
        history_text = ""

        # 检查是否有历史信息
        has_history = (
                context.get("previous_plans") or
                context.get("previous_sub_answers") or
                context.get("unexecuted_plan") or
                context.get("plan_insights")
        )

        if not has_history:
            return ""

        history_text = "\n\n## 历史研究信息\n"

        # 添加前序计划摘要
        previous_plans = context.get("previous_plans", [])
        if previous_plans:
            history_text += "\n### 前序研究计划\n"
            for i, plan in enumerate(previous_plans):
                history_text += f"\n计划 {i + 1}:\n"

                tasks = plan.get("sub_tasks", [])
                if tasks:
                    for task in tasks:
                        history_text += f"- {task.get('title')}\n"

        # 添加已完成任务的子答案
        previous_sub_answers = context.get("previous_sub_answers", {})
        if previous_sub_answers:
            history_text += "\n### 已完成任务的子答案\n"
            for task_id, sub_answer in previous_sub_answers.items():
                history_text += f"\n任务: {task_id}\n"
                # 直接使用子答案作为文本，不假设特定的结构
                history_text += f"结果: {str(sub_answer)[:300]}...\n"

        # 添加未执行计划
        unexecuted_plan = context.get("unexecuted_plan")
        if unexecuted_plan:
            history_text += "\n### 上轮未执行的计划\n"
            tasks = unexecuted_plan.get("sub_tasks", [])
            for task in tasks:
                history_text += f"- {task.get('title')}: {task.get('description')}\n"

        # 添加洞察（不假设特定结构）
        plan_insights = context.get("plan_insights")
        if plan_insights:
            history_text += "\n### 对计划的洞察\n"
            # 直接使用洞察作为文本，不假设特定的结构
            history_text += str(plan_insights)

        history_text += "\n\n请基于上述历史信息和当前查询，优化研究计划。如有需要，可以保留之前计划中合理的部分，并添加新的任务以弥补知识缺口。\n"

        return history_text

    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        解析LLM响应为结构化输出。

        Args:
            response: LLM响应文本

        Returns:
            解析后的结构化输出
        """
        # 由于使用function calling，此方法在当前实现中不会被调用
        # 保留此方法以符合Agent抽象基类要求
        try:
            parsed = json.loads(response)
            return parsed
        except:
            logging.error(f"解析任务规划响应失败: {response}")
            return {
                "error": "解析响应失败",
                "sub_tasks": []
            }