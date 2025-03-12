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
            self.memory.update_state("sub_answers", {})

            # 记录研究开始
            self._record_step("research_started", {
                "query": query,
                "timestamp": datetime.now().isoformat()
            })

            # 获取初始复杂度估计（只用于设置初始迭代次数）
            complexity = self._estimate_initial_complexity(query)
            max_iterations = self._get_max_iterations(complexity)
            self.memory.update_state("complexity", complexity)
            self.memory.update_state("max_iterations", max_iterations)

            # 研究迭代主循环
            current_iteration = 0

            while current_iteration < max_iterations:
                logging.info(f"开始研究迭代 {current_iteration + 1}/{max_iterations}")
                self.memory.update_state("current_iteration", current_iteration + 1)

                # 记录迭代开始
                self._record_step("iteration_started", {
                    "iteration": current_iteration + 1,
                    "max_iterations": max_iterations
                })

                # 1. 任务规划（基于之前的结果优化规划）
                task_planning_context = {
                    "query": query
                }

                # 如果不是第一次迭代，添加前一轮的结果
                if current_iteration > 0:
                    task_planning_context["previous_plan"] = self.memory.get_latest_result("task_planning")
                    task_planning_context["previous_answers"] = self.memory.get_state("sub_answers", {})

                # 执行任务规划
                task_planning_result = self.call_component("task_planning_workflow.execute", task_planning_context)
                self.memory.save_result("task_planning", task_planning_result)

                # 获取子任务列表
                sub_tasks = task_planning_result.get("sub_tasks", [])
                self.memory.update_state("sub_tasks", sub_tasks)

                # 更新复杂度（如有变化）
                complexity = task_planning_result.get("complexity", complexity)
                self.memory.update_state("complexity", complexity)

                # 2. 按顺序处理每个子任务
                iteration_answers = {}

                for index, task in enumerate(sub_tasks):
                    task_id = task.get("id", f"task-{index+1}")

                    # 2.1 为当前任务生成搜索策略并执行搜索
                    search_context = {
                        "current_task": task,
                        "task_index": index,
                        "total_tasks": len(sub_tasks)
                    }

                    # 如果有前一个任务的结果，添加到上下文
                    if index > 0 and sub_tasks[index-1]["id"] in iteration_answers:
                        prev_task_id = sub_tasks[index-1]["id"]
                        prev_task = sub_tasks[index-1]
                        search_context["previous_task"] = prev_task
                        search_context["previous_answer"] = iteration_answers[prev_task_id]

                    # 执行搜索
                    search_result = self.call_component("search_planning_workflow.execute", search_context)
                    self.memory.save_result(f"search_result_{task_id}", search_result)

                    # 2.2 根据搜索结果生成子回答
                    if "search_result" in search_result and not search_result.get("error"):
                        answer_context = {
                            "task": task,
                            "search_result": search_result["search_result"]
                        }

                        # 执行子回答生成
                        sub_answer_result = self.call_component("sub_answer_workflow.execute", answer_context)

                        # 如果生成了子回答，添加到结果
                        if "sub_answer" in sub_answer_result:
                            iteration_answers[task_id] = sub_answer_result["sub_answer"]

                # 3. 更新全局子回答集合
                current_sub_answers = self.memory.get_state("sub_answers", {})
                current_sub_answers.update(iteration_answers)
                self.memory.update_state("sub_answers", current_sub_answers)

                # 记录本轮迭代的子回答
                self.memory.save_result(f"iteration_answers_{current_iteration + 1}", iteration_answers)

                # 4. 生成当前迭代的洞察
                insights = self._generate_insights(query, current_sub_answers)
                self.memory.save_result(f"insights_iteration_{current_iteration + 1}", insights)

                # 记录迭代完成
                self._record_step("iteration_completed", {
                    "iteration": current_iteration + 1,
                    "new_sub_answers": len(iteration_answers),
                    "total_sub_answers": len(current_sub_answers)
                })

                # 5. 检查是否需要继续研究
                if not self._needs_more_research(insights, current_iteration, max_iterations):
                    logging.info("研究目标已满足，不需要更多迭代")
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

            # 所有迭代完成，使用最终的子回答生成答案
            all_sub_answers = self.memory.get_state("sub_answers", {})
            final_insights = self._generate_insights(query, all_sub_answers)

            # 生成最终答案
            answer_plan = self._plan_answer(query, all_sub_answers, final_insights)
            final_answer = self._generate_answer(query, all_sub_answers, answer_plan)

            # 记录研究完成
            self._record_step("research_completed", {
                "query": query,
                "iterations_executed": current_iteration + 1,
                "sub_tasks_completed": len(all_sub_answers),
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

    def _estimate_initial_complexity(self, query: str) -> str:
        """
        初步估计查询复杂度，用于设置初始迭代次数。

        Args:
            query: 研究查询

        Returns:
            复杂度评估: "simple", "medium", 或 "complex"
        """
        # 直接调用任务规划代理
        result = self.call_component("task_plan_agent.estimate_complexity", {
            "query": query
        })

        complexity = result.get("complexity", "medium")
        logging.info(f"初步复杂度评估: {complexity}")

        return complexity

    def _generate_insights(self, query: str, sub_answers: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成研究洞察。

        Args:
            query: 研究查询
            sub_answers: 子回答集合

        Returns:
            研究洞察
        """
        # 调用洞察代理
        result = self.call_component("insight_agent.analyze_results", {
            "query": query,
            "sub_answers": sub_answers
        })

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
        # 调用答案规划代理
        result = self.call_component("answer_plan_agent.plan_answer", {
            "query": query,
            "sub_answers": sub_answers,
            "insights": insights
        })

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
        # 调用答案代理
        result = self.call_component("answer_agent.generate_answer", {
            "query": query,
            "sub_answers": sub_answers,
            "plan": answer_plan
        })

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