"""
任务规划工作流模块，负责分解研究查询并制定计划。
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from aisr.workflows.base import Workflow


class TaskPlanningWorkflow(Workflow):
    """
    任务规划工作流。

    负责分解研究查询为子任务，制定研究计划，
    以及基于研究结果和洞察优化计划。
    """

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行任务规划工作流。

        Args:
            context: 包含研究上下文
                - query: 研究查询（必需）
                - previous_plan: 前一轮任务规划（可选）
                - previous_answers: 之前的子回答结果（可选）

        Returns:
            任务规划结果，包含复杂度、子任务和研究计划
        """
        query = context.get("query")
        previous_plan = context.get("previous_plan")
        previous_answers = context.get("previous_answers")

        if not query:
            error_msg = "缺少研究查询"
            logging.error(error_msg)
            return {"error": error_msg}

        # 确定是初始规划还是优化现有规划
        is_initial_planning = previous_plan is None
        planning_mode = "初始规划" if is_initial_planning else "规划优化"

        logging.info(f"执行任务规划工作流({planning_mode}): '{query}'")

        try:
            # 准备Agent调用上下文
            agent_context = {
                "query": query,
                "mode": planning_mode
            }

            # 添加先前的计划和子回答(如果有)
            if previous_plan:
                agent_context["previous_plan"] = previous_plan

            if previous_answers:
                agent_context["previous_answers"] = previous_answers

            # 单次调用任务规划代理 - 包含所有必要信息
            # 不再拆分为复杂度评估/任务分解/计划创建等多个步骤
            result = self.call_component("task_plan_agent.plan_research", agent_context)

            # 记录规划结果
            self.memory.save_result("task_planning_result", result)

            return result

        except Exception as e:
            logging.error(f"任务规划工作流执行错误: {str(e)}")

            # 创建一个基础的默认计划作为错误恢复措施
            return {
                "error": f"任务规划失败: {str(e)}",
                "query": query,
                "complexity": "medium",  # 默认复杂度
                "sub_tasks": [{"id": "task-1", "description": query, "type": "factual", "priority": "high"}],
                "research_plan": {
                    "overall_approach": "直接搜索并总结",
                    "estimated_iterations": 1
                }
            }