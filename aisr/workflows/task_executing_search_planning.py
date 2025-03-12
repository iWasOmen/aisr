"""
任务执行与搜索规划工作流模块，负责执行子任务列表并管理搜索策略。

实现中层循环，负责高效执行研究任务并生成子回答。
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from aisr.workflows.base import Workflow


class TaskExecutingSearchPlanningWorkflow(Workflow):
    """
    任务执行与搜索规划工作流。

    作为中层循环，负责执行上层任务规划生成的子任务，
    为每个子任务规划搜索策略，并按优先级和依赖关系
    有序地处理子任务，最终收集所有子回答。
    """

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行子任务列表，为每个任务规划搜索并生成子回答。

        Args:
            context: 执行上下文
                - query: 原始查询
                - sub_tasks: 子任务列表
                - previous_answers: 之前轮次已完成的子回答（可选）

        Returns:
            包含子回答集合的结果字典
        """
        query = context.get("query")
        sub_tasks = context.get("sub_tasks", [])
        previous_answers = context.get("previous_answers", {})

        if not sub_tasks:
            error_msg = "缺少子任务列表"
            logging.error(error_msg)
            return {"error": error_msg, "sub_answers": {}}

        logging.info(f"开始任务执行工作流，待处理子任务: {len(sub_tasks)}")

        # 初始化执行状态
        self.memory.update_state("query", query)
        self.memory.update_state("sub_tasks", sub_tasks)
        self.memory.update_state("execution_start_time", datetime.now().isoformat())

        try:
            # 规划任务执行顺序
            prioritized_tasks = self._prioritize_tasks(sub_tasks)
            self.memory.save_result("prioritized_tasks", prioritized_tasks)

            # 存储本轮执行的子回答
            current_answers = {}

            # 按优先级顺序处理子任务
            for index, task in enumerate(prioritized_tasks):
                task_id = task.get("id", f"task-{index+1}")

                # 记录开始处理任务
                self._record_step(f"task_started_{task_id}", {
                    "task_id": task_id,
                    "task_index": index,
                    "total_tasks": len(prioritized_tasks),
                    "task_description": task.get("description", ""),
                    "timestamp": datetime.now().isoformat()
                })

                # 为当前任务规划搜索策略
                search_strategy = self._plan_search_strategy(
                    task,
                    index,
                    prioritized_tasks,
                    # 合并之前和当前的子回答作为上下文
                    {**previous_answers, **current_answers}
                )

                # 记录搜索策略
                self.memory.save_result(f"search_strategy_{task_id}", search_strategy)

                # 执行搜索并生成子回答
                task_result = self._execute_search_for_task(
                    task,
                    search_strategy,
                    index,
                    len(prioritized_tasks)
                )

                # 如果成功生成子回答，保存结果
                if "sub_answer" in task_result and not task_result.get("error"):
                    current_answers[task_id] = task_result["sub_answer"]

                # 记录任务完成
                self._record_step(f"task_completed_{task_id}", {
                    "task_id": task_id,
                    "task_index": index,
                    "success": "sub_answer" in task_result,
                    "timestamp": datetime.now().isoformat()
                })

                # 检查是否需要更新后续任务的优先级
                if index < len(prioritized_tasks) - 1 and len(current_answers) > 0:
                    # 根据已完成的任务结果动态调整后续任务
                    self._adjust_remaining_tasks(
                        prioritized_tasks[index+1:],
                        {**previous_answers, **current_answers}
                    )

            # 记录执行完成
            self.memory.update_state("execution_end_time", datetime.now().isoformat())
            self.memory.update_state("tasks_completed", len(current_answers))

            # 返回本轮执行结果
            return {
                "sub_answers": current_answers,
                "tasks_total": len(sub_tasks),
                "tasks_completed": len(current_answers),
                "execution_time": self._calculate_execution_time()
            }

        except Exception as e:
            logging.error(f"任务执行工作流错误: {str(e)}")

            # 返回已完成的子回答和错误信息
            return {
                "error": f"任务执行失败: {str(e)}",
                "sub_answers": current_answers,
                "tasks_total": len(sub_tasks),
                "tasks_completed": len(current_answers) if 'current_answers' in locals() else 0
            }

    def _prioritize_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        根据任务重要性和依赖关系确定执行顺序。

        Args:
            tasks: 原始子任务列表

        Returns:
            按优先级排序的子任务列表
        """
        # 简单起见，先按原有优先级排序
        priority_map = {"high": 3, "medium": 2, "low": 1}

        prioritized = sorted(
            tasks,
            key=lambda t: priority_map.get(t.get("priority", "medium"), 2),
            reverse=True
        )

        # 可以在此添加更复杂的依赖分析和排序逻辑

        logging.info(f"任务已按优先级排序，优先顺序: {[t.get('id') for t in prioritized]}")
        return prioritized

    def _plan_search_strategy(self,
                             task: Dict[str, Any],
                             task_index: int,
                             all_tasks: List[Dict[str, Any]],
                             existing_answers: Dict[str, Any]) -> Dict[str, Any]:
        """
        为特定子任务规划搜索策略。

        Args:
            task: 当前子任务
            task_index: 任务在队列中的索引
            all_tasks: 所有待执行任务
            existing_answers: 已有的子回答

        Returns:
            搜索策略
        """
        task_id = task.get("id", f"task-{task_index+1}")
        logging.info(f"为任务 {task_id} 规划搜索策略")

        # 构建上下文信息
        search_context = {
            "task": task,
            "task_index": task_index,
            "total_tasks": len(all_tasks)
        }

        # 添加前一个任务的信息（如果有）
        if task_index > 0:
            prev_task = all_tasks[task_index - 1]
            prev_task_id = prev_task.get("id")

            if prev_task_id in existing_answers:
                search_context["previous_task"] = prev_task
                search_context["previous_answer"] = existing_answers[prev_task_id]

        # 添加已有回答的主题和发现
        if existing_answers:
            search_context["existing_answers_count"] = len(existing_answers)

            # 提取关键主题作为上下文（可选择性添加）
            key_themes = self._extract_key_themes(existing_answers)
            if key_themes:
                search_context["key_themes"] = key_themes

        # 调用搜索规划代理生成策略
        search_strategy = self.call_component("search_plan_agent.generate_search_strategy", search_context)

        return search_strategy

    def _execute_search_for_task(self,
                                task: Dict[str, Any],
                                search_strategy: Dict[str, Any],
                                task_index: int,
                                total_tasks: int) -> Dict[str, Any]:
        """
        执行任务的搜索和子回答生成。

        Args:
            task: 当前子任务
            search_strategy: 搜索策略
            task_index: 任务索引
            total_tasks: 总任务数

        Returns:
            包含子回答的结果
        """
        task_id = task.get("id", f"task-{task_index+1}")
        logging.info(f"执行任务 {task_id} 的搜索 ({task_index+1}/{total_tasks})")

        # 调用搜索执行工作流
        search_result = self.call_component("search_executing_workflow.execute", {
            "task": task,
            "search_strategy": search_strategy
        })

        # 记录搜索结果
        self.memory.save_result(f"search_result_{task_id}", search_result)

        return search_result

    def _adjust_remaining_tasks(self,
                               remaining_tasks: List[Dict[str, Any]],
                               current_answers: Dict[str, Any]) -> None:
        """
        根据已完成任务的结果，动态调整剩余任务的优先级或策略。

        Args:
            remaining_tasks: 剩余待执行任务
            current_answers: 当前已有的子回答
        """
        # 这个功能可以在未来迭代中实现
        # 例如，基于当前发现调整任务重要性，或者添加新的子任务
        pass

    def _extract_key_themes(self, answers: Dict[str, Any]) -> List[str]:
        """
        从已有子回答中提取关键主题，用于后续任务的上下文。

        Args:
            answers: 已有子回答

        Returns:
            关键主题列表
        """
        themes = []

        for task_id, answer in answers.items():
            if isinstance(answer, dict):
                # 从key_points中提取主题
                key_points = answer.get("key_points", [])
                if key_points and isinstance(key_points, list):
                    themes.extend(key_points[:2])  # 只取前两个关键点

        # 去重并限制数量
        unique_themes = list(set(themes))
        return unique_themes[:5]  # 最多返回5个主题

    def _calculate_execution_time(self) -> str:
        """计算执行耗时。"""
        start_time = self.memory.get_state("execution_start_time")
        end_time = self.memory.get_state("execution_end_time")

        if not start_time or not end_time:
            return "unknown"

        try:
            start = datetime.fromisoformat(start_time)
            end = datetime.fromisoformat(end_time)
            duration = end - start
            return str(duration)
        except Exception:
            return "error calculating"

    def _record_step(self, step_name: str, data: Dict[str, Any]) -> None:
        """记录执行步骤。"""
        # 保存到工作流内存
        self.memory.save_result(step_name, data)

        # 记录到全局状态日志
        self.memory.update_state("latest_step", {
            "name": step_name,
            "data": data,
            "timestamp": datetime.now().isoformat()
        })