"""
任务规划工作流模块，负责分解研究查询并制定计划。
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from aisr.workflows.base import Workflow


class TaskPlanningWorkflow(Workflow):
    """
    任务规划工作流。

    负责分解研究查询为子任务，并通过与Insight Agent
    的反馈循环不断优化研究计划。
    """

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行任务规划工作流。

        Args:
            context: 包含研究查询的上下文
                - query: 研究查询

        Returns:
            任务规划结果，包含复杂度、子任务和研究计划
        """
        query = context.get("query")
        if not query:
            error_msg = "缺少研究查询"
            logging.error(error_msg)
            return {"error": error_msg}

        logging.info(f"开始任务规划工作流: '{query}'")
        self.memory.update_state("query", query)

        try:
            # 1. 生成初始计划
            initial_plan = self._generate_initial_plan(query)

            # 获取任务复杂度
            complexity = initial_plan.get("complexity", "medium")
            max_iterations = self._get_max_iterations(complexity)
            self.memory.update_state("complexity", complexity)
            self.memory.update_state("max_iterations", max_iterations)

            # 记录最新计划
            current_plan = initial_plan
            self.memory.save_result("initial_plan", current_plan)

            # 2. 迭代优化计划
            for iteration in range(max_iterations - 1):
                logging.info(f"任务规划迭代 {iteration + 1}")

                # 分析当前计划并获取洞察
                insights = self._analyze_plan(current_plan)
                self.memory.save_result(f"plan_insights_{iteration + 1}", insights)

                # 如果没有需要改进的地方，提前结束
                if not insights.get("has_improvement_suggestions", False):
                    logging.info("没有需要改进的地方，提前结束规划循环")
                    break

                # 基于洞察重新规划
                refined_plan = self._refine_plan(current_plan, insights)
                self.memory.save_result(f"refined_plan_{iteration + 1}", refined_plan)

                # 更新当前计划
                current_plan = refined_plan

            # 3. 保存最终计划结果
            final_plan = current_plan
            self.memory.save_result("final_plan", final_plan)

            return final_plan

        except Exception as e:
            logging.error(f"任务规划工作流执行错误: {str(e)}")
            # 记录错误
            self.memory.save_result("task_planning_error", {
                "query": query,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })

            # 返回基本计划(如果有)或错误
            initial_plan = self.memory.get_latest_result("initial_plan")
            if initial_plan:
                return {**initial_plan, "error": str(e)}
            else:
                return {
                    "error": f"任务规划失败: {str(e)}",
                    "query": query,
                    "complexity": "medium",  # 默认复杂度
                    "sub_tasks": [{"id": "task-1", "description": query, "type": "factual", "priority": "high"}]
                }

    def _generate_initial_plan(self, query: str) -> Dict[str, Any]:
        """
        生成初始研究计划。

        Args:
            query: 研究查询

        Returns:
            初始研究计划
        """
        logging.info("生成初始研究计划")

        # 估计查询复杂度
        complexity = self._estimate_complexity(query)
        logging.info(f"查询复杂度评估: {complexity}")

        # 分解查询为子任务
        sub_tasks = self._decompose_query(query, complexity)
        logging.info(f"查询已分解为 {len(sub_tasks)} 个子任务")

        # 生成整体研究计划
        research_plan = self._create_research_plan(query, sub_tasks, complexity)

        # 合并结果
        result = {
            "query": query,
            "complexity": complexity,
            "sub_tasks": sub_tasks,
            "research_plan": research_plan
        }

        return result

    def _estimate_complexity(self, query: str) -> str:
        """
        估计查询复杂度。

        Args:
            query: 研究查询

        Returns:
            复杂度级别: "simple", "medium", 或 "complex"
        """
        # 调用任务规划代理估计复杂度
        result = self.call_component("task_plan_agent.estimate_complexity", {
            "query": query
        })

        # 获取复杂度
        complexity = result.get("complexity", "medium")

        return complexity

    def _decompose_query(self, query: str, complexity: str) -> List[Dict[str, Any]]:
        """
        将查询分解为子任务。

        Args:
            query: 研究查询
            complexity: 复杂度级别

        Returns:
            子任务列表
        """
        # 调用任务规划代理分解查询
        result = self.call_component("task_plan_agent.decompose_query", {
            "query": query,
            "complexity": complexity
        })

        # 获取子任务
        sub_tasks = result.get("tasks", [])

        # 如果子任务为空，创建一个基本任务
        if not sub_tasks:
            sub_tasks = [{"id": "task-1", "description": query, "type": "factual", "priority": "high"}]

        return sub_tasks

    def _create_research_plan(self, query: str, sub_tasks: List[Dict[str, Any]], complexity: str) -> Dict[str, Any]:
        """
        创建研究计划。

        Args:
            query: 研究查询
            sub_tasks: 子任务列表
            complexity: 复杂度级别

        Returns:
            研究计划
        """
        # 调用任务规划代理创建研究计划
        result = self.call_component("task_plan_agent.create_research_plan", {
            "query": query,
            "sub_tasks": sub_tasks,
            "complexity": complexity
        })

        return result

    def _analyze_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析研究计划并提供洞察。

        Args:
            plan: 研究计划

        Returns:
            计划洞察
        """
        # 调用洞察代理分析计划
        result = self.call_component("insight_agent.analyze_plan", {
            "plan": plan
        })

        return result

    def _refine_plan(self, previous_plan: Dict[str, Any], insights: Dict[str, Any]) -> Dict[str, Any]:
        """
        基于洞察优化研究计划。

        Args:
            previous_plan: 之前的研究计划
            insights: 计划洞察

        Returns:
            优化后的研究计划
        """
        # 调用任务规划代理优化计划
        result = self.call_component("task_plan_agent.refine_plan", {
            "previous_plan": previous_plan,
            "insights": insights
        })

        return result

    def _get_max_iterations(self, complexity: str) -> int:
        """基于复杂度确定最大迭代次数。"""
        if complexity == "simple":
            return 1
        elif complexity == "medium":
            return 2
        else:  # complex
            return 3