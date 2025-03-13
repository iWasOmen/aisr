"""
搜索工具模块，提供Web搜索和网页爬取功能。

包含两个原子功能(web_api和web_crawler)以及一个复合功能(web_search)。
"""
import json
import os
import requests
import logging
from typing import Dict, Any, List
import time
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SearchTools:
    """
    搜索工具类，提供网络搜索和内容爬取功能。
    """

    @staticmethod
    def web_api(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        执行Web搜索API调用，获取搜索结果。

        Args:
            query: 搜索查询
            max_results: 最大结果数量，默认5条

        Returns:
            搜索结果列表，每个结果包含标题、摘要和链接
        """
        try:
            # 设置Bing搜索API环境变量
            os.environ["BING_SUBSCRIPTION_KEY"] = "xx"  # 这里应该填入有效的API密钥
            os.environ["BING_SEARCH_URL"] = "https://api.bing.microsoft.com/v7.0/search"

            # 导入Bing搜索包装器
            from langchain_community.utilities import BingSearchAPIWrapper
            search = BingSearchAPIWrapper()

            # 执行搜索
            results = search.results(query, max_results)

            # 格式化结果
            formatted_results = []
            for result in results:
                formatted_result = {
                    "title": result.get("title", ""),
                    "snippet": result.get("snippet", ""),
                    "link": result.get("link", ""),
                    "query": query
                }
                formatted_results.append(formatted_result)

            return formatted_results

        except Exception as e:
            logger.error(f"Web API搜索错误: {str(e)}")
            return [{"error": str(e), "query": query}]

    @staticmethod
    def web_crawler(url: str) -> Dict[str, Any]:
        """
        爬取指定URL的网页内容。

        Args:
            url: 要爬取的网页URL

        Returns:
            包含页面标题和内容的字典
        """
        logger.info(f"正在爬取: {url}")
        try:
            # 使用Jina AI的网页渲染服务
            jina_url = f"https://r.jina.ai/{url}"
            headers = {
                "Authorization": "Bearer jina_xx"
            }

            # 发送请求
            response = requests.get(jina_url, headers=headers, timeout=30)

            # 检查响应状态
            if response.status_code != 200:
                return {
                    "url": url,
                    "error": f"HTTP错误: {response.status_code}",
                    "content": "",
                    "title": ""
                }

            # 简单提取标题（实际实现可能需要更复杂的HTML解析）
            content = response.text
            temp_dic = {"res":content[:200]}
            logger.info(f"爬取结果: {json.dumps(temp_dic,ensure_ascii=False)}")
            title = ""

            # 非常简单的标题提取
            title_start = content.find("<title>")
            title_end = content.find("</title>")
            if title_start > -1 and title_end > -1:
                title = content[title_start + 7:title_end].strip()

            return {
                "url": url,
                "content": content,
                "title": title
            }

        except Exception as e:
            logger.error(f"网页爬取错误 ({url}): {str(e)}")
            return {
                "url": url,
                "error": str(e),
                "content": "",
                "title": ""
            }

    @staticmethod
    def web_search(query: str, max_results: int = 1) -> Dict[str, Any]:
        """
        执行综合Web搜索，包括搜索和内容爬取。

        Args:
            query: 搜索查询
            max_results: 最大结果数量，默认5条

        Returns:
            包含搜索结果和爬取内容的字典
        """
        try:
            # 1. 执行初步搜索
            search_results = SearchTools.web_api(query, max_results)

            # 记录初步搜索完成
            logger.info(f"搜索完成: '{query}'，获取了{len(search_results)}条结果")

            # 2. 爬取每个搜索结果的内容
            results = []
            for idx,result in enumerate(search_results):
                break
                logger.info(f"结果{idx+1}:{result}")
                # 获取URL
                url = result.get("link", "")
                if not url:
                    continue

                # 爬取内容
                try:
                    # 提取有效URL
                    parsed_url = urlparse(url)
                    if not parsed_url.scheme or not parsed_url.netloc:
                        logger.warning(f"无效URL: {url}")
                        continue

                    # 添加延迟以避免请求过快
                    time.sleep(2)

                    # 执行爬取
                    crawl_result = SearchTools.web_crawler(url)

                    # 合并搜索结果和爬取内容
                    combined_result = {
                        **result,
                        "content": crawl_result.get("content", ""),
                        "is_crawled": True,
                        "crawl_error": crawl_result.get("error", "")
                    }

                    results.append(combined_result)

                except Exception as e:
                    logger.error(f"爬取过程中出错 ({url}): {str(e)}")
                    # 添加错误信息，但仍然包含搜索结果
                    result["is_crawled"] = False
                    result["crawl_error"] = str(e)
                    results.append(result)

            return {
                "query": query,
                "results": results,
                "result_count": len(results),
                "has_error": any(r.get("crawl_error", "") for r in results)
            }

        except Exception as e:
            logger.error(f"综合搜索错误: {str(e)}")
            return {
                "query": query,
                "error": str(e),
                "results": [],
                "result_count": 0,
                "has_error": True
            }


# 单独函数版本（如果不想使用类）

def web_api(query: str, max_results: int = 1) -> List[Dict[str, Any]]:
    """
    执行Web搜索API调用，获取搜索结果。

    Args:
        query: 搜索查询
        max_results: 最大结果数量，默认5条

    Returns:
        搜索结果列表，每个结果包含标题、摘要和链接
    """
    return SearchTools.web_api(query, max_results)


def web_crawler(url: str) -> Dict[str, Any]:
    """
    爬取指定URL的网页内容。

    Args:
        url: 要爬取的网页URL

    Returns:
        包含页面标题和内容的字典
    """
    return SearchTools.web_crawler(url)


def web_search(query: str, max_results: int = 1) -> Dict[str, Any]:
    """
    执行综合Web搜索，包括搜索和内容爬取。

    Args:
        query: 搜索查询
        max_results: 最大结果数量，默认5条

    Returns:
        包含搜索结果和爬取内容的字典
    """
    return SearchTools.web_search(query, max_results)