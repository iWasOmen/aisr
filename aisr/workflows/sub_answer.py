"""
子回答工作流模块，负责分析搜索结果并生成子任务答案。
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from aisr.workflows.base import Workflow


class SubAnswerWorkflow(Workflow):
    """
    子回答工作流。

    负责分析搜索结果、生成子回答，并在必要时
    通过深度爬取改进回答质量。
    """

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行子回答工作流。

        Args:
            context: 包含任务和搜索结果的上下文
                - task: 当前子任务
                - search_result: 当前任务的搜索结果

        Returns:
            子回答结果
        """
        task = context.get("task")
        search_result = context.get("search_result")

        if not task:
            error_msg = "缺少任务信息"
            logging.error(error_msg)
            return {"error": error_msg}

        if not search_result:
            error_msg = "缺少搜索结果"
            logging.error(error_msg)
            return {"error": error_msg}

        task_id = task.get("id", "unknown-task")

        logging.info(f"执行子回答工作流，任务 {task_id}")

        try:
            # 1. 生成初始子回答
            sub_answer = self.call_component("sub_answer_agent.analyze_results", {
                "task": task,
                "search_results": search_result
            })

            # 记录初始答案
            self.memory.save_result("initial_answer", sub_answer)

            # 2. 检查是否需要深度爬取
            urls_to_analyze = sub_answer.get("urls_for_deep_analysis", [])

            # 如果需要深度分析且有URL
            if sub_answer.get("needs_further_analysis", False) and urls_to_analyze:

                # 存储爬虫结果
                crawl_results = {}

                # 爬取URLs
                for url in urls_to_analyze:
                    logging.info(f"爬取URL: {url}")

                    # 调用网页爬虫
                    crawl_result = self.call_component("web_crawler.execute", {
                        "url": url
                    })

                    # 添加到结果
                    crawl_results[url] = crawl_result

                # 如果有爬虫结果，更新子回答
                if crawl_results:
                    # 更新子回答上下文
                    updated_context = {
                        "task": task,
                        "search_results": search_result,
                        "deep_analysis_results": crawl_results
                    }

                    # 生成更新的答案
                    sub_answer = self.call_component("sub_answer_agent.refine_answer", updated_context)

                    # 记录更新的答案
                    self.memory.save_result("updated_answer", sub_answer)

            # 3. 添加元数据
            sub_answer["task_id"] = task_id
            sub_answer["task_description"] = task.get("description", "")
            sub_answer["task_type"] = task.get("type", "")

            # 4. 返回最终子回答
            return {
                "task_id": task_id,
                "sub_answer": sub_answer
            }

        except Exception as e:
            logging.error(f"子回答工作流执行错误: {str(e)}")

            # 返回错误结果
            return {
                "task_id": task_id,
                "error": f"子回答生成失败: {str(e)}"
            }