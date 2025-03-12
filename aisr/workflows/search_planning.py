"""
搜索规划工作流模块，负责规划和执行搜索策略。
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from aisr.workflows.base import Workflow


class SearchPlanningWorkflow(Workflow):
    """
    搜索规划工作流。

    负责为单个子任务生成搜索策略，执行搜索，并根据搜索结果
    不断优化搜索策略的反馈循环。
    """

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行搜索规划工作流，处理单个子任务。

        Args:
            context: 包含当前任务的上下文
                - current_task: 当前要执行搜索的子任务
                - task_index: 当前任务在列表中的索引（可选）
                - total_tasks: 总任务数量（可选）
                - previous_task: 前一个子任务（可选）
                - previous_search_result: 前一个任务的搜索结果（可选）

        Returns:
            当前任务的搜索结果
        """
        current_task = context.get("current_task")
        previous_task = context.get("previous_task")
        previous_search_result = context.get("previous_search_result")

        if not current_task:
            error_msg = "缺少当前任务"
            logging.error(error_msg)
            return {"error": error_msg}

        task_id = current_task.get("id", "unknown-task")
        task_index = context.get("task_index", 0)
        total_tasks = context.get("total_tasks", 1)

        logging.info(f"开始搜索规划工作流，任务 {task_id} ({task_index+1}/{total_tasks})")

        # 记录任务和上下文信息
        self.memory.update_state("current_task", current_task)
        self.memory.update_state("task_index", task_index)
        self.memory.update_state("total_tasks", total_tasks)

        if previous_task:
            self.memory.update_state("previous_task", previous_task)

        if previous_search_result:
            self.memory.update_state("previous_search_result", previous_search_result)

        try:
            # 生成搜索策略（考虑前一个任务的结果）
            search_strategy = self._generate_search_strategy(current_task, previous_task, previous_search_result)
            self.memory.save_result("search_strategy", search_strategy)

            # 执行搜索循环
            search_result = self._execute_search_loop(current_task, search_strategy)

            # 记录任务搜索完成
            self.memory.save_result("search_completed", {
                "task_id": task_id,
                "task_index": task_index,
                "timestamp": datetime.now().isoformat(),
                "search_queries": search_result.get("queries", []),
                "result_count": len(search_result.get("results", []))
            })

            # 返回结果
            return {
                "task_id": task_id,
                "search_result": search_result,
                "task_index": task_index,
                "total_tasks": total_tasks
            }

        except Exception as e:
            logging.error(f"搜索规划工作流执行错误: {str(e)}")
            # 返回错误
            return {
                "task_id": task_id,
                "error": f"搜索规划失败: {str(e)}",
                "task_index": task_index,
                "total_tasks": total_tasks
            }

    def _generate_search_strategy(self,
                                 current_task: Dict[str, Any],
                                 previous_task: Optional[Dict[str, Any]] = None,
                                 previous_search_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        为当前子任务生成搜索策略，可能考虑前一个任务的结果。

        Args:
            current_task: 当前子任务
            previous_task: 前一个子任务（可选）
            previous_search_result: 前一个任务的搜索结果（可选）

        Returns:
            搜索策略
        """
        task_id = current_task.get("id", "unknown-task")
        logging.info(f"为任务 {task_id} 生成搜索策略")

        # 准备上下文
        strategy_context = {
            "task": current_task
        }

        # 如果有前一个任务和结果，添加到上下文
        if previous_task and previous_search_result:
            strategy_context["previous_task"] = previous_task
            strategy_context["previous_search_result"] = previous_search_result

            # 调用搜索规划代理
            result = self.call_component("search_plan_agent.generate_strategy_with_context", strategy_context)
        else:
            # 调用搜索规划代理（无上下文）
            result = self.call_component("search_plan_agent.generate_strategy", strategy_context)

        # 提取策略
        strategy = result.get("strategy", {})

        # 如果返回为空，使用默认策略
        if not strategy:
            strategy = self._create_default_strategy(current_task)

        return strategy

    def _create_default_strategy(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """创建默认搜索策略。"""
        task_id = task.get("id", "unknown-task")
        task_description = task.get("description", "")

        # 创建简单的默认查询
        default_query = task_description
        if len(default_query) > 100:
            default_query = default_query[:97] + "..."

        return {
            "task_id": task_id,
            "approach": "direct",
            "queries": [default_query],
            "tools": ["web_search"],
            "is_default": True
        }

    def _execute_search_loop(self, task: Dict[str, Any], search_strategy: Dict[str, Any], max_iterations: int = 3) -> Dict[str, Any]:
        """
        执行搜索循环，根据结果优化搜索策略。

        Args:
            task: 子任务
            search_strategy: 初始搜索策略
            max_iterations: 最大迭代次数

        Returns:
            搜索结果
        """
        task_id = task.get("id", "unknown-task")
        task_type = task.get("type", "factual")

        logging.info(f"执行任务 {task_id} ({task_type}) 的搜索循环")

        # 初始化迭代跟踪
        current_iteration = 0
        current_strategy = search_strategy
        current_results = None
        initial_strategy = search_strategy

        # 搜索循环
        while current_iteration < max_iterations:
            # 执行搜索
            if current_iteration == 0:
                # 首次搜索
                search_results = self._execute_search(task, current_strategy)
                # 保存初始结果
                initial_results = search_results
            else:
                # 使用优化后的策略再次搜索
                search_results = self._execute_search(task, current_strategy)

            # 记录搜索结果
            self.memory.save_result(f"search_iteration_{current_iteration + 1}", {
                "strategy": current_strategy,
                "results": search_results
            })

            # 更新当前结果
            current_results = search_results

            # 评估搜索结果质量
            quality_score = self._evaluate_search_results(search_results, task)

            # 记录质量评估
            self.memory.save_result(f"search_quality_{current_iteration + 1}", {
                "quality_score": quality_score,
                "iteration": current_iteration + 1
            })

            # 如果结果质量足够好，或已达到最大迭代次数，退出循环
            if quality_score >= 0.7 or current_iteration >= max_iterations - 1:
                break

            # 基于结果优化搜索策略
            refined_strategy = self._refine_search_strategy(task, current_strategy, search_results)

            # 记录策略优化
            self.memory.save_result(f"refined_strategy_{current_iteration + 1}", refined_strategy)

            # 更新搜索策略
            current_strategy = refined_strategy

            # 增加迭代计数
            current_iteration += 1

        # 构建最终结果
        final_result = {
            "task_id": task_id,
            "iterations": current_iteration + 1,
            "results": current_results.get("results", []),
            "queries": current_results.get("queries", []),
            "quality_score": self._evaluate_search_results(current_results, task)
        }

        # 如果进行了多次迭代，添加初始和最终信息
        if current_iteration > 0:
            final_result.update({
                "initial_strategy": initial_strategy,
                "final_strategy": current_strategy,
                "initial_results": initial_results.get("results", []),
                "improvement": self._calculate_improvement(initial_results, current_results)
            })

        return final_result

    def _execute_search(self, task: Dict[str, Any], strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行搜索。

        Args:
            task: 子任务
            strategy: 搜索策略

        Returns:
            搜索结果
        """
        task_id = task.get("id", "unknown-task")

        logging.info(f"为任务 {task_id} 执行搜索")

        # 获取搜索查询和工具
        queries = strategy.get("queries", [])
        tools = strategy.get("tools", ["web_search"])

        # 记录搜索开始
        self.memory.save_result("search_started", {
            "task_id": task_id,
            "queries": queries,
            "tools": tools,
            "timestamp": datetime.now().isoformat()
        })

        # 汇总所有结果
        all_results = []
        executed_queries = []

        # 对每个查询执行搜索
        for query in queries:
            # 记录查询
            executed_queries.append(query)

            # 对每个工具执行搜索
            for tool_name in tools:
                try:
                    # 执行搜索
                    if tool_name == "web_search":
                        tool_results = self.call_component("web_search.execute", {
                            "query": query
                        })
                    elif tool_name == "web_crawler":
                        # 对于网页爬虫，可能需要先获取URL
                        # 这里我们假设URL已经在之前的web_search中获取
                        continue
                    else:
                        logging.warning(f"未知的搜索工具: {tool_name}")
                        continue

                    # 添加工具和查询信息
                    for result in tool_results.get("results", []):
                        result["tool"] = tool_name
                        result["query"] = query
                        all_results.append(result)

                except Exception as e:
                    logging.error(f"执行工具 {tool_name} 搜索错误: {str(e)}")
                    # 添加错误结果
                    all_results.append({
                        "tool": tool_name,
                        "query": query,
                        "error": str(e),
                        "is_error": True
                    })

        # 整理结果
        search_results = {
            "task_id": task_id,
            "queries": executed_queries,
            "results": all_results,
            "result_count": len(all_results),
            "timestamp": datetime.now().isoformat()
        }

        return search_results

    def _evaluate_search_results(self, search_results: Dict[str, Any], task: Dict[str, Any]) -> float:
        """
        评估搜索结果的质量。

        Args:
            search_results: 搜索结果
            task: 子任务

        Returns:
            质量评分 (0.0 到 1.0)
        """
        # 获取结果列表
        results = search_results.get("results", [])

        # 如果没有结果，返回低分
        if not results:
            return 0.1

        # 计算带错误的结果
        error_results = [r for r in results if r.get("is_error", False)]
        error_ratio = len(error_results) / len(results) if results else 1.0

        # 基本质量评分 - 可以替换为更复杂的逻辑
        base_score = 0.5 + (0.5 * (1.0 - error_ratio))

        # 根据结果数量调整
        count_factor = min(len(results) / 5, 1.0)  # 最多5个结果记满分

        # 综合评分
        score = base_score * count_factor

        return min(1.0, max(0.0, score))

    def _refine_search_strategy(self, task: Dict[str, Any], strategy: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
        """
        基于搜索结果优化搜索策略。

        Args:
            task: 子任务
            strategy: 当前搜索策略
            results: 搜索结果

        Returns:
            优化后的搜索策略
        """
        logging.info(f"优化任务 {task.get('id')} 的搜索策略")

        # 调用搜索规划代理优化策略
        refined_strategy = self.call_component("search_plan_agent.refine_strategy", {
            "task": task,
            "previous_strategy": strategy,
            "search_results": results
        })

        return refined_strategy

    def _calculate_improvement(self, initial_results: Dict[str, Any], final_results: Dict[str, Any]) -> Dict[str, Any]:
        """计算搜索结果改进情况。"""
        initial_count = len(initial_results.get("results", []))
        final_count = len(final_results.get("results", []))

        return {
            "initial_result_count": initial_count,
            "final_result_count": final_count,
            "result_count_change": final_count - initial_count,
            "percentage_change": ((final_count - initial_count) / initial_count * 100) if initial_count > 0 else 0
        }