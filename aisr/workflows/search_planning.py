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

    负责为单个子任务规划和执行搜索，
    包括搜索策略生成、执行搜索和结果评估。
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

        if not current_task:
            error_msg = "缺少当前任务"
            logging.error(error_msg)
            return {"error": error_msg}

        task_id = current_task.get("id", "unknown-task")
        task_index = context.get("task_index", 0)
        total_tasks = context.get("total_tasks", 1)

        logging.info(f"执行搜索规划工作流，任务 {task_id} ({task_index+1}/{total_tasks})")

        try:
            # 1. 直接调用搜索规划代理生成搜索策略
            # 传入所有可用的上下文信息，而不是分开处理
            search_strategy = self.call_component("search_plan_agent.generate_search_strategy", context)

            # 记录策略
            self.memory.save_result("search_strategy", search_strategy)

            # 2. 执行搜索
            search_results = self._execute_search(current_task, search_strategy)

            # 记录结果
            self.memory.save_result("search_results", search_results)

            # 3. 返回结果
            return {
                "task_id": task_id,
                "search_result": search_results,
                "task_index": task_index,
                "total_tasks": total_tasks
            }

        except Exception as e:
            logging.error(f"搜索规划工作流执行错误: {str(e)}")
            # 返回错误结果
            return {
                "task_id": task_id,
                "error": f"搜索失败: {str(e)}",
                "task_index": task_index,
                "total_tasks": total_tasks
            }

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