"""
研究工作流模块，协调整个研究过程。
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from aisr.workflows.base import Workflow


class ResearchWorkflow(Workflow):
    """
    主研究工作流，协调整个研究过程。

    这是系统的核心工作流，负责编排其他工作流和实现
    任务规划、搜索和答案生成的三大反馈循环。
    """

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行研究工作流。

        Args:
            context: 包含研究查询的上下文
                - query: 研究查询

        Returns:
            最终研究结果
        """
        query = context.get("query")
        if not query:
            error_msg = "缺少研究查询"
            logging.error(error_msg)
            return {"error": error_msg}

        logging.info(f"开始研究工作流: '{query}'")
        self.memory.update_state("query", query)
        self.memory.update_state("start_time", datetime.now().isoformat())

        try:
            # 初始化研究状态
            self.memory.update_state("current_iteration", 0)
            self.memory.update_state("complexity", "unknown")

            # 记录研究开始
            self._record_step("research_started", {
                "query": query,
                "timestamp": datetime.now().isoformat()
            })

            # 1. 任务规划循环：分解查询并制定计划
            task_planning_result = self._execute_task_planning_loop(query)

            # 更新复杂度和最大迭代次数
            complexity = task_planning_result.get("complexity", "medium")
            self.memory.update_state("complexity", complexity)

            max_iterations = self._get_max_iterations(complexity)
            self.memory.update_state("max_iterations", max_iterations)

            # 子任务列表
            sub_tasks = task_planning_result.get("sub_tasks", [])
            self.memory.update_state("sub_tasks", sub_tasks)

            # 2. 执行研究迭代
            current_iteration = 0
            sub_answers = {}

            while current_iteration < max_iterations:
                logging.info(f"开始研究迭代 {current_iteration + 1}")
                self.memory.update_state("current_iteration", current_iteration + 1)

                # 记录迭代开始
                self._record_step("iteration_started", {
                    "iteration": current_iteration + 1,
                    "max_iterations": max_iterations
                })

                # 执行搜索规划与搜索
                search_results = self._execute_search_planning_loop(sub_tasks)

                # 执行子回答循环
                iteration_answers = self._execute_sub_answer_loop(sub_tasks, search_results)

                # 合并子回答
                for task_id, answer in iteration_answers.items():
                    sub_answers[task_id] = answer

                # 生成当前迭代的洞察
                insights = self._generate_insights(query, sub_answers)

                # 记录迭代完成
                self._record_step("iteration_completed", {
                    "iteration": current_iteration + 1,
                    "sub_tasks_completed": len(iteration_answers),
                    "total_sub_answers": len(sub_answers)
                })

                # 检查是否需要继续研究
                if not self._needs_more_research(insights, current_iteration, max_iterations):
                    logging.info(f"研究目标已满足，不需要更多迭代")
                    break

                # 考虑用户交互
                if self._should_interact_with_user(current_iteration, max_iterations):
                    user_feedback = self._request_user_feedback("继续研究", {
                        "iteration": current_iteration + 1,
                        "max_iterations": max_iterations,
                        "insights_summary": self._summarize_insights(insights)
                    })

                    if not self._should_continue_based_on_feedback(user_feedback):
                        logging.info("根据用户反馈停止迭代")
                        break

                # 准备下一轮迭代
                current_iteration += 1

            # 3. 生成最终答案
            logging.info("生成最终研究答案")

            # 规划答案结构
            answer_plan = self._plan_answer(query, sub_answers, insights)

            # 生成答案
            final_answer = self._generate_answer(query, sub_answers, answer_plan)

            # 记录研究完成
            self._record_step("research_completed", {
                "query": query,
                "iterations_executed": current_iteration + 1,
                "sub_tasks_completed": len(sub_answers),
                "end_time": datetime.now().isoformat()
            })

            return final_answer

        except Exception as e:
            logging.error(f"研究工作流执行错误: {str(e)}")
            # 记录错误
            self._record_step("research_error", {
                "query": query,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })

            return {
                "error": f"研究执行失败: {str(e)}",
                "query": query,
                "partial_results": self.memory.get_state("sub_answers", {})
            }

    def _execute_task_planning_loop(self, query: str) -> Dict[str, Any]:
        """
        执行任务规划循环。

        Args:
            query: 研究查询

        Returns:
            任务规划结果
        """
        logging.info("执行任务规划循环")

        # 调用任务规划工作流
        result = self.call_component("task_planning_workflow.execute", {
            "query": query
        })

        # 保存任务规划结果
        self.memory.save_result("task_planning", result)

        return result

    def _execute_search_planning_loop(self, sub_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        执行搜索规划循环。

        Args:
            sub_tasks: 子任务列表

        Returns:
            搜索结果
        """
        logging.info(f"执行搜索规划循环，{len(sub_tasks)} 个子任务")

        # 调用搜索规划工作流
        result = self.call_component("search_planning_workflow.execute", {
            "sub_tasks": sub_tasks
        })

        # 保存搜索结果
        self.memory.save_result("search_planning", result)

        return result

    def _execute_sub_answer_loop(self, sub_tasks: List[Dict[str, Any]], search_results: Dict[str, Any]) -> Dict[
        str, Any]:
        """
        执行子回答循环。

        Args:
            sub_tasks: 子任务列表
            search_results: 搜索结果

        Returns:
            子回答结果
        """
        logging.info("执行子回答循环")

        # 调用子回答工作流
        result = self.call_component("sub_answer_workflow.execute", {
            "sub_tasks": sub_tasks,
            "search_results": search_results
        })

        # 保存子回答结果
        self.memory.save_result("sub_answer", result)

        # 更新全局子回答集合
        current_sub_answers = self.memory.get_state("sub_answers", {})
        current_sub_answers.update(result.get("sub_answers", {}))
        self.memory.update_state("sub_answers", current_sub_answers)

        return result.get("sub_answers", {})

    def _generate_insights(self, query: str, sub_answers: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成研究洞察。

        Args:
            query: 研究查询
            sub_answers: 子回答集合

        Returns:
            研究洞察
        """
        logging.info("生成研究洞察")

        # 调用洞察代理
        result = self.call_component("insight_agent.analyze_results", {
            "query": query,
            "sub_answers": sub_answers
        })

        # 保存洞察结果
        self.memory.save_result("insights", result)

        return result

    def _plan_answer(self, query: str, sub_answers: Dict[str, Any], insights: Dict[str, Any]) -> Dict[str, Any]:
        """
        规划答案结构。

        Args:
            query: 研究查询
            sub_answers: 子回答集合
            insights: 研究洞察

        Returns:
            答案计划
        """
        logging.info("规划答案结构")

        # 调用答案规划代理
        result = self.call_component("answer_plan_agent.plan_answer", {
            "query": query,
            "sub_answers": sub_answers,
            "insights": insights
        })

        # 保存答案计划
        self.memory.save_result("answer_plan", result)

        return result

    def _generate_answer(self, query: str, sub_answers: Dict[str, Any], answer_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成最终答案。

        Args:
            query: 研究查询
            sub_answers: 子回答集合
            answer_plan: 答案计划

        Returns:
            最终答案
        """
        logging.info("生成最终答案")

        # 调用答案代理
        result = self.call_component("answer_agent.generate_answer", {
            "query": query,
            "sub_answers": sub_answers,
            "plan": answer_plan
        })

        # 保存最终答案
        self.memory.save_result("final_answer", result)

        return result

    def _get_max_iterations(self, complexity: str) -> int:
        """基于复杂度确定最大迭代次数。"""
        if complexity == "simple":
            return 1
        elif complexity == "medium":
            return 2
        else:  # complex
            return 3

    def _needs_more_research(self, insights: Dict[str, Any], current_iteration: int, max_iterations: int) -> bool:
        """
        确定是否需要更多研究。

        Args:
            insights: 当前的研究洞察
            current_iteration: 当前迭代索引
            max_iterations: 最大迭代次数

        Returns:
            是否需要继续研究
        """
        # 如果已达到最大迭代次数
        if current_iteration >= max_iterations - 1:
            return False

        # 检查未回答的问题
        unanswered_questions = insights.get("unanswered_questions", [])
        if len(unanswered_questions) > 1:
            return True

        # 检查分歧点
        areas_of_disagreement = insights.get("areas_of_disagreement", [])
        if len(areas_of_disagreement) > 1:
            return True

        # 检查意外发现
        unexpected_findings = insights.get("unexpected_findings", [])
        if len(unexpected_findings) > 0 and current_iteration < 1:
            return True

        # 默认不需要更多研究
        return False

    def _should_interact_with_user(self, current_iteration: int, max_iterations: int) -> bool:
        """确定是否应该与用户交互。"""
        # 简单示例：在第一次迭代后与用户交互
        complexity = self.memory.get_state("complexity", "medium")
        if complexity == "complex":
            return True  # 复杂查询总是请求用户反馈
        elif complexity == "medium" and current_iteration >= 1:
            return True  # 中等复杂度在第一次迭代后请求反馈

        return False

    def _request_user_feedback(self, interaction_point: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        请求用户反馈（模拟）。

        在实际实现中，这将调用UI组件或API来获取用户输入。
        这里我们提供一个简单的模拟实现。
        """
        logging.info(f"请求用户反馈：{interaction_point}")

        # 记录交互点
        self._record_step("user_interaction_requested", {
            "interaction_point": interaction_point,
            "context": context
        })

        # 模拟用户反馈 - 在实际实现中应替换为真实交互
        return {
            "continue": True,
            "feedback": "继续研究",
            "timestamp": datetime.now().isoformat()
        }

    def _should_continue_based_on_feedback(self, feedback: Dict[str, Any]) -> bool:
        """根据用户反馈决定是否继续研究。"""
        return feedback.get("continue", True)

    def _record_step(self, step_name: str, data: Dict[str, Any]) -> None:
        """记录研究步骤。"""
        # 保存到工作流内存
        self.memory.save_result(step_name, data)

        # 更新研究历史
        self.call_component("memory_manager.record_research_step", {
            "step_name": step_name,
            "data": data
        })

    def _summarize_insights(self, insights: Dict[str, Any]) -> str:
        """创建洞察的简短摘要。"""
        summary_parts = ["研究洞察摘要:"]

        if "key_themes" in insights and insights["key_themes"]:
            themes = insights["key_themes"][:3]
            summary_parts.append(f"主题: {', '.join(themes)}")

        if "unanswered_questions" in insights and insights["unanswered_questions"]:
            questions = insights["unanswered_questions"][:2]
            summary_parts.append(f"未回答问题: {', '.join(questions)}")

        if "unexpected_findings" in insights and insights["unexpected_findings"]:
            findings = insights["unexpected_findings"][:2]
            summary_parts.append(f"意外发现: {', '.join(findings)}")

        return "\n".join(summary_parts)