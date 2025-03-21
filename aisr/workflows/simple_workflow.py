"""
AISR主工作流模块，协调整个研究过程。

包含主工作流函数，负责协调各个组件的执行，从查询到最终答案。
"""
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

import logging
from typing import Dict, Any, List, Optional
import time

from aisr.agents.task_plan import TaskPlanAgent
from aisr.agents.search_plan import SearchPlanAgent
from aisr.agents.sub_answer import SubAnswerAgent
from aisr.agents.insight import InsightAgent
from aisr.agents.answer_plan import AnswerPlanAgent
from aisr.agents.answer import AnswerAgent
from aisr.tools.search_tools import web_search
from aisr.core.llm_provider import LLMProvider
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def main_workflow(query: str, max_iterations: int = 2) -> Dict[str, Any]:
    """
    执行完整的AISR研究工作流。

    Args:
        query: 用户研究查询
        max_iterations: 最大规划迭代次数，默认3次

    Returns:
        包含最终答案的字典
    """
    # 记录开始执行
    start_time = time.time()
    logging.info(f"开始AISR工作流，查询: '{query}'")

    # 初始化LLM提供者
    llm = LLMProvider()

    # 初始化所有代理
    task_plan_agent = TaskPlanAgent(llm, memory=None)
    search_plan_agent = SearchPlanAgent(llm, memory=None)
    sub_answer_agent = SubAnswerAgent(llm, memory=None)
    insight_agent = InsightAgent(llm, memory=None)
    answer_plan_agent = AnswerPlanAgent(llm, memory=None)
    answer_agent = AnswerAgent(llm, memory=None)

    # 跟踪所有子答案
    all_sub_answers = {}

    # 添加任务记忆系统
    task_memory = {}  # 存储每个任务的记忆
    iteration_tasks = {}  # 存储每次迭代的任务

    # 外层循环：任务规划 -> 执行 -> 洞察 -> 再规划
    for iteration in range(max_iterations):
        logging.info(f"开始规划迭代 {iteration + 1}/{max_iterations}")

        # 1. 任务规划
        task_plan_context = {
            "query": query,
            "previous_sub_answers": all_sub_answers
        }

        # 如果不是第一次迭代，添加前序计划
        if iteration > 0:
            task_plan_context["previous_plans"] = [previous_task_plan]
            task_plan_context["plan_insights"] = insights

        # 执行任务规划
        task_plan = task_plan_agent.execute(task_plan_context)
        previous_task_plan = task_plan  # 保存，以便下一轮使用

        # 获取子任务列表
        sub_tasks = task_plan.get("sub_tasks", [])
        if not sub_tasks:
            logging.warning("任务规划未返回子任务，中止工作流")
            return {
                "query": query,
                "answer": "无法为您的查询生成研究计划。",
                "error": "任务规划失败"
            }

        logging.info(f"计划生成了 {len(sub_tasks)} 个子任务")

        # 更新当前迭代的任务记录
        current_iteration_tasks = [task.get("id") for task in sub_tasks]
        iteration_tasks[iteration] = current_iteration_tasks

        # 2. 内层循环：执行所有子任务
        iteration_sub_answers = {}

        for task_index, task in enumerate(sub_tasks):
            task_id = task.get("id")
            task_title = task.get("title", "未命名任务")

            logging.info(f"执行子任务 {task_index + 1}/{len(sub_tasks)}: {task_title}")

            # 如果这个任务已有答案(在之前的迭代中)，则跳过
            if task_id in all_sub_answers:
                logging.info(f"子任务 '{task_title}' 已有答案，跳过执行")
                iteration_sub_answers[task_id] = all_sub_answers[task_id]
                continue

            # 初始化或获取任务记忆
            if task_id not in task_memory:
                task_memory[task_id] = {
                    "search_plans": [],
                    "search_results": [],
                    "sub_answers": [],
                    "iterations": 0
                }

            # 收集相关任务的答案（当前迭代中已完成的任务）
            related_answers = {}
            for completed_task_id in iteration_sub_answers:
                related_answers[completed_task_id] = iteration_sub_answers[completed_task_id]

            # 2.1 搜索规划
            search_plan_context = {
                "task": task,
                "previous_search_plans": task_memory[task_id]["search_plans"],
                "related_tasks_answers": related_answers
            }

            # 如果该任务之前有子答案，添加到上下文
            if task_memory[task_id]["sub_answers"]:
                search_plan_context["previous_sub_answer"] = task_memory[task_id]["sub_answers"][-1]

            search_plan = search_plan_agent.execute(search_plan_context)

            # 存储搜索计划到任务记忆
            task_memory[task_id]["search_plans"].append(search_plan)

            # 获取搜索查询
            queries = search_plan.get("queries", [])
            if not queries:
                logging.warning(f"子任务 '{task_title}' 没有生成搜索查询，跳过执行")
                continue

            # 2.2 执行搜索
            all_search_results = []
            for query_index, query in enumerate(queries[:3]):  # 限制最多使用前3个查询
                logging.info(f"执行搜索 {query_index + 1}/{min(len(queries), 3)}: '{query}'")
                search_result = web_search(query)
                all_search_results.append(search_result)

            # 合并所有搜索结果
            merged_results = {
                "results": [],
                "result_count": 0
            }

            for result in all_search_results:
                merged_results["results"].extend(result.get("results", []))

            merged_results["result_count"] = len(merged_results["results"])

            # 存储搜索结果到任务记忆
            task_memory[task_id]["search_results"].append(merged_results)

            # 2.3 生成子答案
            sub_answer_context = {
                "task": task,
                "search_results": merged_results
            }

            sub_answer = sub_answer_agent.execute(sub_answer_context)
            answer_text = sub_answer.get("answer", "")

            # 保存子答案
            iteration_sub_answers[task_id] = answer_text

            # 更新任务记忆
            task_memory[task_id]["sub_answers"].append(answer_text)
            task_memory[task_id]["iterations"] += 1

            # 延迟一小段时间，避免API调用过快
            time.sleep(1)

        # 更新所有子答案集合
        all_sub_answers.update(iteration_sub_answers)

        logging.info(f"迭代 {iteration + 1} 完成，累计子答案: {len(all_sub_answers)}/{len(sub_tasks)}")

        if iteration == max_iterations-1:
            break

        # 4. 生成洞察
        insight_context = {
            "query": query,
            "sub_answers": all_sub_answers,
            "unexecuted_plan": task_plan
        }

        insights = insight_agent.execute(insight_context)

        logging.info("已生成洞察，准备下一轮规划迭代")

        # 延迟一小段时间
        time.sleep(1)

    # 5. 生成答案大纲
    answer_plan_context = {
        "query": query,
        "sub_answers": all_sub_answers
    }

    answer_plan = answer_plan_agent.execute(answer_plan_context)

    # 6. 生成最终答案
    answer_context = {
        "query": query,
        "sub_answers": all_sub_answers,
        "outline": answer_plan.get("outline", "")
    }

    final_answer = answer_agent.execute(answer_context)

    # 计算总耗时
    total_time = time.time() - start_time
    logging.info(f"AISR工作流完成，总耗时: {total_time:.2f}秒")

    # 返回结果
    return {
        "query": query,
        "answer": final_answer.get("answer", ""),
        "sub_answers": all_sub_answers,
        "execution_time": f"{total_time:.2f}秒",
        "task_count": len(sub_tasks),
        "completed_tasks": len(all_sub_answers)
    }

# 测试代码保持不变
query = "请研究deep research"
query = "统计下美国纽约历届市长的大学毕业院校"
#query = "北京到南京共有几条中图路过济南的高铁线"
result = main_workflow(query)
print(result["answer"])