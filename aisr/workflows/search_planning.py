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

    负责为子任务生成搜索策略，执行搜索，并根据搜索结果
    不断优化搜索策略的反馈循环。
    """

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行搜索规划工作流。

        Args:
            context: 包含子任务的上下文
                - sub_tasks: 子任务列表

        Returns:
            搜索结果，按任务ID组织
        """
        sub_tasks = context.get("sub_tasks", [])
        if not sub_tasks:
            error_msg = "缺少子任务"
            logging.error(error_msg)
            return {"error": error_msg}

        logging.info(f"开始搜索规划工作流，{len(sub_tasks)} 个任务")
        self.memory.update_state("sub_tasks", sub_tasks)

        try:
            # 为每个任务生成搜索策略
            all_search_strategies = self._generate_search_strategies(sub_tasks)
            self.memory.save_result("search_strategies", all_search_strategies)

            # 存储每个任务的搜索结果
            all_search_results = {}

            # 为每个任务执行搜索
            for task in sub_tasks:
                task_id = task.get("id", "unknown-task")

                # 获取该任务的搜索策略
                task_strategy = all_search_strategies.get(task_id, {})

                # 如果没有策略，跳过此任务
                if not task_strategy:
                    logging.warning(f"任务 {task_id} 缺少搜索策略，跳过")
                    continue

                # 执行搜索循环
                search_result = self._execute_search_loop(task, task_strategy)
                all_search_results[task_id] = search_result

                # 记录任务搜索完成
                self.memory.save_result(f"search_completed_{task_id}", {
                    "task_id": task_id,
                    "timestamp": datetime.now().isoformat(),
                    "search_queries": search_result.get("queries", []),
                    "result_count": len(search_result.get("results", []))
                })

            # 保存所有搜索结果
            self.memory.save_result("all_search_results", all_search_results)

            return {
                "search_results": all_search_results,
                "task_count": len(sub_tasks),
                "completed_count": len(all_search_results)
            }

        except Exception as e:
            logging.error(f"搜索规划工作流执行错误: {str(e)}")
            # 返回已完成的结果或错误
            search_results = self.memory.get_latest_result("all_search_results")
            if search_results:
                return {
                    "search_results": search_results,
                    "error": str(e),
                    "partial_results": True
                }
            else:
                return {
                    "error": f"搜索规划失败: {str(e)}",
                    "search_results": {}
                }

    def _generate_search_strategies(self, sub_tasks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        为所有子任务生成搜索策略。

        Args:
            sub_tasks: 子任务列表

        Returns:
            按任务ID索引的搜索策略字典
        """
        logging.info("为所有子任务生成搜索策略")

        # 调用搜索规划代理
        result = self.call_component("search_plan_agent.generate_search_strategies", {
            "sub_tasks": sub_tasks
        })

        # 获取搜索策略
        search_strategies = result.get("search_strategies", {})

        # 保存每个任务的搜索策略
        for task_id, strategy in search_strategies.items():
            self.memory.save_result(f"search_strategy_{task_id}", strategy)

        return search_strategies

    def _execute_search_loop(self, task: Dict[str, Any], search_strategy: Dict[str, Any], max_iterations: int = 3) -> \
    Dict[str, Any]:
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
            self.memory.save_result(f"search_iteration_{task_id}_{current_iteration + 1}", {
                "strategy": current_strategy,
                "results": search_results
            })

            # 更新当前结果
            current_results = search_results

            # 评估搜索结果质量
            quality_score = self._evaluate_search_results(search_results, task)

            # 记录质量评估
            self.memory.save_result(f"search_quality_{task_id}_{current_iteration + 1}", {
                "quality_score": quality_score,
                "iteration": current_iteration + 1
            })

            # 如果结果质量足够好，或已达到最大迭代次数，退出循环
            if quality_score >= 0.7 or current_iteration >= max_iterations - 1:
                break

            # 基于结果优化搜索策略
            refined_strategy = self._refine_search_strategy(task, current_strategy, search_results)

            # 记录策略优化
            self.memory.save_result(f"refined_strategy_{task_id}_{current_iteration + 1}", refined_strategy)

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
        self.memory.save_result(f"search_started_{task_id}", {
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

    def _refine_search_strategy(self, task: Dict[str, Any], strategy: Dict[str, Any], results: Dict[str, Any]) -> Dict[
        str, Any]:
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