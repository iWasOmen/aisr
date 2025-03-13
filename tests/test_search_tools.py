"""
搜索工具模块的测试。
"""

import unittest
import time
from aisr.tools.search_tools import web_api, web_crawler, web_search


class TestSearchTools(unittest.TestCase):
    """搜索工具函数的测试用例"""

    def test_web_api(self):
        """测试Web API搜索功能"""
        query = "深度学习"

        print("\n=== 测试Web API搜索 ===")
        print(f"查询: {query}")

        results = web_api(query, max_results=3)

        print(f"搜索结果数: {len(results)}")
        for i, result in enumerate(results):
            print(f"\n结果 {i + 1}:")
            print(f"标题: {result.get('title', 'N/A')}")
            print(f"链接: {result.get('link', 'N/A')}")
            snippet = result.get('snippet', 'N/A')
            print(f"摘要: {snippet[:100]}..." if len(snippet) > 100 else f"摘要: {snippet}")

        # 验证返回了搜索结果
        self.assertIsInstance(results, list)
        if not any(r.get("error") for r in results):  # 如果没有错误
            self.assertTrue(len(results) > 0)
            for result in results:
                self.assertIn("title", result)
                self.assertIn("link", result)
                self.assertIn("snippet", result)

    def test_web_crawler(self):
        """测试Web爬虫功能"""
        # 使用一个稳定的URL进行测试
        url = "https://en.wikipedia.org/wiki/Deep_learning"

        print("\n=== 测试Web爬虫 ===")
        print(f"URL: {url}")

        result = web_crawler(url)

        print(f"标题: {result.get('title', 'N/A')}")

        content = result.get('content', '')
        content_preview = content[:200] + "..." if len(content) > 200 else content
        print(f"内容预览: {content_preview}")
        print(f"内容长度: {len(content)} 字符")

        # 验证返回了内容
        self.assertIn("url", result)
        self.assertEqual(result["url"], url)

        if "error" not in result:
            self.assertIn("content", result)
            self.assertTrue(len(result["content"]) > 0)

    def test_web_search(self):
        """测试综合Web搜索功能"""
        query = "深度学习与传统机器学习的区别"

        print("\n=== 测试综合Web搜索 ===")
        print(f"查询: {query}")
        print("执行中，这可能需要一些时间...")

        start_time = time.time()
        result = web_search(query, max_results=2)  # 限制为2个结果以加快测试
        duration = time.time() - start_time

        print(f"搜索耗时: {duration:.2f}秒")
        print(f"结果数: {result.get('result_count', 0)}")

        results = result.get("results", [])
        for i, res in enumerate(results):
            print(f"\n结果 {i + 1}:")
            print(f"标题: {res.get('title', 'N/A')}")
            print(f"链接: {res.get('link', 'N/A')}")

            # 检查是否成功爬取
            is_crawled = res.get("is_crawled", False)
            print(f"爬取状态: {'成功' if is_crawled and not res.get('crawl_error') else '失败'}")

            if is_crawled and not res.get('crawl_error'):
                content = res.get('content', '')
                print(f"内容长度: {len(content)} 字符")
                print(f"内容预览: {content[:150]}..." if len(content) > 150 else f"内容预览: {content}")

        # 验证返回了搜索结果
        self.assertIn("results", result)
        self.assertIn("result_count", result)
        if not result.get("has_error"):
            self.assertTrue(len(results) > 0)


if __name__ == '__main__':
    unittest.main()