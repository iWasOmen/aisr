"""
任务执行与搜索规划工作流模块，负责执行子任务列表并管理搜索策略。

实现中层循环，负责高效执行研究任务并生成子回答。
包含对每个任务的搜索策略迭代，直到任务解决或达到最大尝试次数。
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from aisr.workflows.base import Workflow


class TaskExecutingSearchPlanningWorkflow(Workflow):
    """
    任务执行与搜索规划工作流。

    作为中层循环，负责执行上层任务规划生成的子任务，
    为每个子任务规划搜索策略并迭代执行直到任务解决，
    最终收集所有子回答。
    """

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行子任务列表，为每个任务规划搜索并生成子回答。

        Args:
            context: 执行上下文
                - query: 原始查询
                - sub_tasks: 子任务列表 (已按优先级排序)
                - previous_answers: 之前轮次已完成的子回答（可选）
                - max_search_attempts: 每个任务的最大搜索尝试次数（默认3）

        Returns:
            包含子回答集合的结果字典
        """
        query = context.get("query")
        sub_tasks = context.get("sub_tasks", [])
        previous_answers = context.get("previous_answers", {})
        max_search_attempts = context.get("max_search_attempts", 3)

        if not sub_tasks:
            error_msg = "缺少子任务列表"
            logging.error(error_msg)
            return {"error": error_msg, "sub_answers": {}}

        logging.info(f"开始任务执行工作流，待处理子任务: {len(sub_tasks)}")

        # 初始化执行状态
        self.memory.update_state("query", query)
        self.memory.update_state("sub_tasks", sub_tasks)
        self.memory.update_state("execution_start_time", datetime.now().isoformat())
        self.memory.update_state("max_search_attempts", max_search_attempts)

        try:
            # 直接使用传入的子任务列表，保持原有优先级
            self.memory.save_result("tasks_to_execute", sub_tasks)

            # 存储本轮执行的子回答
            current_answers = {}

            # 按顺序处理子任务
            for index, task in enumerate(sub_tasks):
                task_id = task.get("id", f"task-{index+1}")

                # 记录开始处理任务
                self._record_step(f"task_started_{task_id}", {
                    "task_id": task_id,
                    "task_index": index,
                    "total_tasks": len(sub_tasks),
                    "task_description": task.get("description", ""),
                    "timestamp": datetime.now().isoformat()
                })

                # ======= 对当前任务执行搜索循环，直到解决或达到最大尝试次数 =======
                search_attempt = 0
                task_resolved = False
                task_result = None

                # 累积的搜索结果和评估
                cumulative_search_results = []

                while not task_resolved and search_attempt < max_search_attempts:
                    search_attempt += 1
                    logging.info(f"任务 {task_id} 的搜索尝试 {search_attempt}/{max_search_attempts}")

                    # 1. 为当前任务规划搜索策略（考虑之前的尝试结果）
                    search_context = self._prepare_search_context(
                        task,
                        index,
                        sub_tasks,
                        {**previous_answers, **current_answers},
                        cumulative_search_results,
                        search_attempt
                    )

                    search_strategy = self.call_component("search_plan_agent.generate_search_strategy", search_context)

                    # 记录搜索策略
                    self.memory.save_result(f"search_strategy_{task_id}_attempt_{search_attempt}", search_strategy)

                    # 2. 执行搜索
                    search_result = self.call_component("search_sub_answer_executing.execute", {
                        "task": task,
                        "search_strategy": search_strategy,
                        "attempt": search_attempt,
                        "previous_attempts": cumulative_search_results
                    })

                    # 记录搜索结果
                    self.memory.save_result(f"search_result_{task_id}_attempt_{search_attempt}", search_result)

                    # 添加到累积的搜索结果
                    cumulative_search_results.append({
                        "attempt": search_attempt,
                        "strategy": search_strategy,
                        "result": search_result
                    })

                    # 3. 评估任务是否已解决
                    task_resolved, task_result = self._evaluate_task_completion(
                        task,
                        search_result,
                        cumulative_search_results
                    )

                    # 记录当前尝试的结果
                    self._record_step(f"task_{task_id}_attempt_{search_attempt}", {
                        "search_attempt": search_attempt,
                        "resolved": task_resolved,
                        "timestamp": datetime.now().isoformat()
                    })

                    # 如果解决了，或者已有子回答且当前是最后一次尝试，保存结果
                    if task_resolved or (search_attempt == max_search_attempts and "sub_answer" in search_result):
                        if not task_result and "sub_answer" in search_result:
                            task_result = search_result

                # 如果有任务结果并且包含子回答，保存到当前回答
                if task_result and "sub_answer" in task_result:
                    current_answers[task_id] = task_result["sub_answer"]

                # 记录任务完成，无论解决与否
                final_status = "resolved" if task_resolved else "max_attempts_reached"
                self._record_step(f"task_completed_{task_id}", {
                    "task_id": task_id,
                    "task_index": index,
                    "status": final_status,
                    "attempts": search_attempt,
                    "success": task_id in current_answers,
                    "timestamp": datetime.now().isoformat()
                })

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
                "sub_answers": current_answers if 'current_answers' in locals() else {},
                "tasks_total": len(sub_tasks),
                "tasks_completed": len(current_answers) if 'current_answers' in locals() else 0
            }

    def _prepare_search_context(self,
                              task: Dict[str, Any],
                              task_index: int,
                              all_tasks: List[Dict[str, Any]],
                              existing_answers: Dict[str, Any],
                              previous_attempts: List[Dict[str, Any]] = None,
                              current_attempt: int = 1) -> Dict[str, Any]:
        """
        准备搜索上下文信息，包含前一次搜索结果和尝试信息。

        Args:
            task: 当前子任务
            task_index: 任务在队列中的索引
            all_tasks: 所有待执行任务
            existing_answers: 已有的子回答
            previous_attempts: 当前任务的前几次搜索尝试结果
            current_attempt: 当前是第几次尝试

        Returns:
            搜索上下文
        """
        task_id = task.get("id", f"task-{task_index+1}")
        logging.info(f"为任务 {task_id} 准备第 {current_attempt} 次搜索上下文")

        # 构建上下文信息
        search_context = {
            "task": task,
            "task_index": task_index,
            "total_tasks": len(all_tasks),
            "current_attempt": current_attempt
        }

        # 添加前几次搜索尝试信息
        if previous_attempts:
            search_context["previous_attempts"] = previous_attempts

            # 分析前几次尝试的问题
            if len(previous_attempts) > 0:
                search_context["previous_queries"] = [
                    attempt.get("strategy", {}).get("queries", [])
                    for attempt in previous_attempts
                ]
                search_context["previous_tools"] = [
                    attempt.get("strategy", {}).get("tools", [])
                    for attempt in previous_attempts
                ]

                # 获取最近一次尝试的搜索结果数量
                last_attempt = previous_attempts[-1]
                result_count = last_attempt.get("result", {}).get("result_count", 0)
                search_context["last_attempt_result_count"] = result_count

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

            # 提取关键主题作为上下文
            key_themes = self._extract_key_themes(existing_answers)
            if key_themes:
                search_context["key_themes"] = key_themes

        return search_context

    def _evaluate_task_completion(self,
                                task: Dict[str, Any],
                                current_result: Dict[str, Any],
                                all_attempts: List[Dict[str, Any]]) -> tuple:
        """
        评估任务是否已经成功解决。

        Args:
            task: 当前子任务
            current_result: 当前搜索尝试的结果
            all_attempts: 所有搜索尝试的结果

        Returns:
            (is_resolved, result_to_use) 元组:
            - is_resolved: 布尔值，表示任务是否已解决
            - result_to_use: 要使用的结果，如果已解决
        """
        # 如果当前结果包含error，任务未解决
        if current_result.get("error"):
            return False, None

        # 如果current_result中有解决状态，使用它
        if "task_resolved" in current_result:
            return current_result["task_resolved"], current_result

        # 检查是否有子回答
        if "sub_answer" not in current_result:
            return False, None

        sub_answer = current_result.get("sub_answer", {})

        # 检查子回答的质量和完整性
        confidence = sub_answer.get("confidence", 0)
        completeness = sub_answer.get("completeness", 0)

        # 如果子回答明确标记为需要进一步搜索
        if sub_answer.get("needs_further_search", False):
            return False, None

        # 如果置信度和完整性足够高
        if confidence >= 0.8 and completeness >= 0.8:
            return True, current_result

        # 如果已经有至少2次尝试，且当前结果比前一次好
        if len(all_attempts) >= 2:
            previous_confidence = all_attempts[-2].get("result", {}).get("sub_answer", {}).get("confidence", 0)
            previous_completeness = all_attempts[-2].get("result", {}).get("sub_answer", {}).get("completeness", 0)

            # 如果当前结果显著优于前一次，且达到了可接受的水平
            if (confidence >= 0.6 and completeness >= 0.6 and
                confidence > previous_confidence and completeness > previous_completeness):
                return True, current_result

        # 默认情况下，认为任务未完全解决，需要继续尝试
        return False, None

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