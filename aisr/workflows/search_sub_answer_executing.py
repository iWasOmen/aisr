"""
搜索与子回答执行工作流模块，负责执行特定搜索策略并直接生成子答案。

作为内层循环，负责执行单个任务的搜索策略，包括搜索执行、
由agent决定的深度爬取，以及子答案生成。
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from aisr.workflows.base import Workflow


class SearchSubAnswerExecutingWorkflow(Workflow):
    """
    搜索与子回答执行工作流。

    作为内层循环，负责执行指定的搜索策略，进行网络搜索，
    由子回答代理决定是否需要深度内容爬取，并直接生成结构化子回答。
    """

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行搜索策略并生成子回答。

        Args:
            context: 执行上下文
                - task: 当前子任务
                - search_strategy: 搜索策略
                - attempt: 当前是第几次尝试（可选）
                - previous_attempts: 前几次搜索尝试信息（可选）

        Returns:
            包含搜索结果和子回答的结果
        """
        task = context.get("task")
        search_strategy = context.get("search_strategy")
        attempt = context.get("attempt", 1)

        if not task:
            error_msg = "缺少任务信息"
            logging.error(error_msg)
            return {"error": error_msg}

        if not search_strategy:
            error_msg = "缺少搜索策略"
            logging.error(error_msg)
            return {"error": error_msg}

        task_id = task.get("id", "unknown-task")

        logging.info(f"执行搜索与子回答工作流，任务 {task_id}，第 {attempt} 次尝试")

        try:
            # 记录执行开始
            self.memory.update_state("search_start_time", datetime.now().isoformat())
            self.memory.update_state("current_task", task)
            self.memory.update_state("current_strategy", search_strategy)

            # 1. 执行初步搜索
            search_results = self._execute_search(task, search_strategy)

            # 记录初步搜索结果
            self.memory.save_result("initial_search_results", search_results)

            # 如果搜索结果为空或出错，直接返回
            if search_results.get("error") or search_results.get("result_count", 0) == 0:
                return {
                    "task_id": task_id,
                    "error": search_results.get("error", "搜索未返回任何结果"),
                    "search_result": search_results,
                    "needs_further_search": True,
                    "search_attempt": attempt
                }

            # 2. 首先让子回答代理分析初步搜索结果
            initial_analysis = self.call_component("sub_answer_agent.analyze_results", {
                "task": task,
                "search_results": search_results,
                "is_initial_analysis": True  # 标记这是初步分析
            })

            # 记录初步分析
            self.memory.save_result("initial_analysis", initial_analysis)

            # 3. 检查子回答代理是否需要深度爬取
            needs_deep_crawling = initial_analysis.get("needs_further_analysis", False)
            urls_for_crawling = initial_analysis.get("urls_for_deep_analysis", [])

            # 4. 如果需要深度爬取，执行爬取
            crawl_results = {}
            if needs_deep_crawling and urls_for_crawling:
                logging.info(f"任务 {task_id} 需要深度爬取 {len(urls_for_crawling)} 个URL")

                for url in urls_for_crawling:
                    logging.info(f"爬取URL: {url}")

                    # 调用网页爬虫工具
                    crawl_result = self.call_component("web_crawler.execute", {
                        "url": url,
                        "depth": search_strategy.get("crawl_depth", 1),
                        "max_pages": search_strategy.get("max_crawl_pages", 3)
                    })

                    if not crawl_result.get("error"):
                        crawl_results[url] = crawl_result

                # 记录爬取结果
                self.memory.save_result("crawl_results", crawl_results)

            # 5. 准备最终分析的上下文
            final_analysis_context = {
                "task": task,
                "search_results": search_results
            }

            # 如果有爬取结果，合并到上下文
            if crawl_results:
                # 合并搜索结果和爬取结果
                combined_results = self._combine_results(search_results, crawl_results)
                final_analysis_context["search_results"] = combined_results
                final_analysis_context["deep_analysis_results"] = crawl_results
                final_analysis_context["is_post_crawl_analysis"] = True

            # 6. 生成最终子回答
            # 如果没有执行深度爬取，使用初步分析结果
            if not crawl_results and not needs_deep_crawling:
                sub_answer = initial_analysis
            else:
                # 否则，使用合并结果生成新的子回答
                sub_answer = self.call_component("sub_answer_agent.analyze_results", final_analysis_context)

            # 记录子回答
            self.memory.save_result("sub_answer", sub_answer)

            # 7. 构建最终结果
            result = {
                "task_id": task_id,
                "search_result": final_analysis_context.get("search_results", search_results),
                "sub_answer": sub_answer,
                "needs_further_search": sub_answer.get("needs_further_search", False),
                "search_attempt": attempt,
                "urls_crawled": list(crawl_results.keys()) if crawl_results else [],
                "timestamp": datetime.now().isoformat()
            }

            # 记录执行完成
            self.memory.update_state("search_end_time", datetime.now().isoformat())

            return result

        except Exception as e:
            logging.error(f"搜索与子回答工作流错误: {str(e)}")

            # 返回错误结果
            return {
                "task_id": task_id,
                "error": f"搜索与子回答执行失败: {str(e)}",
                "search_attempt": attempt,
                "needs_further_search": True
            }

    def _execute_search(self, task: Dict[str, Any], strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行搜索策略中定义的搜索。

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

        # 如果没有提供搜索查询，尝试从任务描述生成一个默认查询
        if not queries:
            description = task.get("description", "")
            if description:
                queries = [description]
            else:
                logging.warning(f"任务 {task_id} 没有搜索查询和任务描述")
                return {"error": "未提供搜索查询"}

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
                    # 获取工具参数
                    tool_params = strategy.get(f"{tool_name}_params", {})

                    # 执行搜索
                    if tool_name == "web_search":
                        tool_results = self.call_component("web_search.execute", {
                            "query": query,
                            **tool_params
                        })
                    elif tool_name == "web_crawler":
                        # 对于网页爬虫，这里仅爬取已知URL
                        # 深度爬取在单独的步骤中进行
                        if "url" in tool_params:
                            tool_results = self.call_component("web_crawler.execute", {
                                "url": tool_params["url"],
                                **{k: v for k, v in tool_params.items() if k != "url"}
                            })
                        else:
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
            "tools_used": tools,
            "results": all_results,
            "result_count": len(all_results),
            "timestamp": datetime.now().isoformat()
        }

        return search_results

    def _combine_results(self, search_results: Dict[str, Any], crawl_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并搜索结果和爬取结果。

        Args:
            search_results: 初步搜索结果
            crawl_results: 深度爬取结果

        Returns:
            合并后的结果
        """
        # 创建搜索结果的副本
        combined = {
            **search_results,
            "has_crawl_results": bool(crawl_results),
            "crawled_urls": list(crawl_results.keys())
        }

        # 如果有爬取结果，更新结果列表
        if crawl_results:
            # 获取原始结果列表的副本
            results = combined.get("results", [])[:]
            updated_results = []

            # 标记哪些URL已经在原始结果中
            processed_urls = set()

            # 先处理已有结果，如果URL在爬取结果中则更新它
            for result in results:
                url = result.get("url")

                if url and url in crawl_results:
                    # 更新现有结果
                    crawl_data = crawl_results[url]
                    updated_result = {
                        **result,  # 保留原始字段
                        "content": crawl_data.get("content", result.get("content", "")),
                        "is_crawled": True,
                        "crawl_timestamp": datetime.now().isoformat()
                    }
                    updated_results.append(updated_result)
                    processed_urls.add(url)
                else:
                    # 保持原样
                    updated_results.append(result)

            # 添加未处理的爬取结果作为新条目
            for url, crawl_data in crawl_results.items():
                if url not in processed_urls:
                    new_result = {
                        "url": url,
                        "title": crawl_data.get("title", url),
                        "content": crawl_data.get("content", ""),
                        "tool": "web_crawler",
                        "is_crawled": True,
                        "crawl_timestamp": datetime.now().isoformat()
                    }
                    updated_results.append(new_result)

            # 更新结果列表和计数
            combined["results"] = updated_results
            combined["result_count"] = len(updated_results)

        return combined