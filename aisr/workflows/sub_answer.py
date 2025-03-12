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

    负责分析搜索结果，生成初步子回答，并通过对重要信息源的
    深度爬取和分析，不断完善子回答的反馈循环。
    """

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行子回答工作流。

        Args:
            context: 包含子任务和搜索结果的上下文
                - sub_tasks: 子任务列表
                - search_results: 搜索结果

        Returns:
            子回答结果，按任务ID组织
        """
        sub_tasks = context.get("sub_tasks", [])
        search_results = context.get("search_results", {}).get("search_results", {})

        if not sub_tasks:
            error_msg = "缺少子任务"
            logging.error(error_msg)
            return {"error": error_msg}

        if not search_results:
            error_msg = "缺少搜索结果"
            logging.error(error_msg)
            return {"error": error_msg}

        logging.info(f"开始子回答工作流，{len(sub_tasks)} 个任务")
        self.memory.update_state("sub_tasks", sub_tasks)

        try:
            # 存储每个任务的子回答
            all_sub_answers = {}

            # 为每个任务生成子回答
            for task in sub_tasks:
                task_id = task.get("id", "unknown-task")

                # 获取该任务的搜索结果
                task_search_results = search_results.get(task_id, {})

                # 如果没有搜索结果，跳过此任务
                if not task_search_results:
                    logging.warning(f"任务 {task_id} 缺少搜索结果，跳过")
                    continue

                # 执行子回答循环
                sub_answer = self._execute_sub_answer_loop(task, task_search_results)
                all_sub_answers[task_id] = sub_answer

                # 记录任务子回答完成
                self.memory.save_result(f"sub_answer_completed_{task_id}", {
                    "task_id": task_id,
                    "timestamp": datetime.now().isoformat(),
                    "confidence": sub_answer.get("confidence", 0)
                })

            # 保存所有子回答
            self.memory.save_result("all_sub_answers", all_sub_answers)

            return {
                "sub_answers": all_sub_answers,
                "task_count": len(sub_tasks),
                "completed_count": len(all_sub_answers)
            }

        except Exception as e:
            logging.error(f"子回答工作流执行错误: {str(e)}")
            # 返回已完成的结果或错误
            sub_answers = self.memory.get_latest_result("all_sub_answers")
            if sub_answers:
                return {
                    "sub_answers": sub_answers,
                    "error": str(e),
                    "partial_results": True
                }
            else:
                return {
                    "error": f"子回答生成失败: {str(e)}",
                    "sub_answers": {}
                }

    def _execute_sub_answer_loop(self, task: Dict[str, Any], search_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行子回答循环，通过深度爬取和分析完善答案。

        Args:
            task: 子任务
            search_results: 搜索结果

        Returns:
            子回答结果
        """
        task_id = task.get("id", "unknown-task")

        logging.info(f"执行任务 {task_id} 的子回答循环")

        # 初始化子回答上下文
        sub_answer_context = {
            "task": task,
            "search_results": search_results
        }

        # 1. 生成初始子回答
        initial_answer = self._generate_sub_answer(sub_answer_context)
        self.memory.save_result(f"initial_answer_{task_id}", initial_answer)

        # 当前子回答
        current_answer = initial_answer

        # 最大爬虫迭代次数
        max_crawler_iterations = 3
        crawler_iteration = 0

        # 存储深度分析结果
        deep_analysis_results = {}

        # 只要需要进一步分析且未达到最大迭代次数，就继续执行爬虫分析循环
        while (current_answer.get("needs_further_analysis", False) and
               crawler_iteration < max_crawler_iterations):

            # 获取需要深入分析的URL
            urls_to_analyze = current_answer.get("urls_for_deep_analysis", [])

            # 如果没有URL，跳出循环
            if not urls_to_analyze:
                logging.info(f"任务 {task_id} 没有URL需要深度分析，结束爬虫循环")
                break

            # 记录需要分析的URL
            self.memory.save_result(f"urls_to_analyze_{task_id}_{crawler_iteration}", {
                "urls": urls_to_analyze,
                "iteration": crawler_iteration + 1
            })

            logging.info(f"执行任务 {task_id} 的爬虫深度分析，迭代 {crawler_iteration + 1}")

            # 筛选新的URL（之前未分析过的）
            new_urls = [url for url in urls_to_analyze if url not in deep_analysis_results]

            # 如果没有新URL，跳出循环
            if not new_urls:
                logging.info(f"任务 {task_id} 没有新URL需要分析，结束爬虫循环")
                break

            # 对每个新URL使用爬虫
            for url in new_urls:
                # 爬取URL
                crawler_result = self._crawl_url(url)

                # 保存爬虫结果
                self.memory.save_result(f"crawler_result_{task_id}_{crawler_iteration}_{url}", crawler_result)

                # 添加到深度分析结果
                deep_analysis_results[url] = crawler_result

            # 更新子回答上下文
            sub_answer_context["deep_analysis_results"] = deep_analysis_results
            sub_answer_context["crawler_iteration"] = crawler_iteration + 1

            # 基于深度分析生成更新的子回答
            updated_answer = self._generate_sub_answer(sub_answer_context)

            # 保存更新的子回答
            self.memory.save_result(f"updated_answer_{task_id}_{crawler_iteration + 1}", updated_answer)

            # 更新当前子回答
            current_answer = updated_answer

            # 增加爬虫迭代计数
            crawler_iteration += 1

            # 如果置信度足够高，提前结束循环
            if current_answer.get("confidence", 0) >= 0.8:
                logging.info(f"任务 {task_id} 子回答置信度已达标，提前结束爬虫循环")
                break

        # 最终子回答
        final_answer = current_answer

        # 添加元数据
        final_answer["task_description"] = task.get("description", "")
        final_answer["task_type"] = task.get("type", "")
        final_answer["crawler_iterations"] = crawler_iteration
        final_answer["urls_analyzed"] = list(deep_analysis_results.keys())

        # 如果进行了爬虫迭代，添加改进信息
        if crawler_iteration > 0:
            initial_confidence = initial_answer.get("confidence", 0)
            final_confidence = final_answer.get("confidence", 0)

            final_answer["confidence_improvement"] = final_confidence - initial_confidence

        return final_answer

    def _generate_sub_answer(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成子回答。

        Args:
            context: 子回答上下文
                - task: 子任务
                - search_results: 搜索结果
                - deep_analysis_results: 可选的深度分析结果
                - crawler_iteration: 可选的爬虫迭代次数

        Returns:
            子回答
        """
        # 获取迭代信息
        crawler_iteration = context.get("crawler_iteration", 0)

        # 确定要使用的代理方法
        if crawler_iteration == 0:
            # 初始子回答
            method = "sub_answer_agent.analyze_results"
        else:
            # 基于深度分析的更新子回答
            method = "sub_answer_agent.refine_answer"

        # 调用子回答代理
        result = self.call_component(method, context)

        return result

    def _crawl_url(self, url: str) -> Dict[str, Any]:
        """
        爬取并分析URL内容。

        Args:
            url: 要爬取的URL

        Returns:
            爬虫分析结果
        """
        logging.info(f"爬取URL: {url}")

        try:
            # 调用网页爬虫工具
            result = self.call_component("web_crawler.execute", {
                "url": url
            })

            return result

        except Exception as e:
            logging.error(f"爬取URL错误: {str(e)}")
            # 返回错误结果
            return {
                "url": url,
                "error": str(e),
                "is_error": True,
                "timestamp": datetime.now().isoformat()
            }