"""
日志工具模块，为AISR系统提供日志记录功能。
"""

import logging
import sys
from typing import Optional


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    设置日志配置。

    Args:
        log_level: 日志级别 ("DEBUG", "INFO", "WARNING", "ERROR")
        log_file: 可选的日志文件路径
    """
    # 转换日志级别字符串为常量
    level = getattr(logging, log_level.upper(), logging.INFO)

    # 基本配置
    handlers = [logging.StreamHandler(sys.stdout)]

    # 如果提供了日志文件，添加文件处理器
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    # 配置日志
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )

    # 设置特定库的日志级别
    # 例如，将一些噪音大的库设置为更高级别
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    logging.info(f"日志系统已配置，级别: {log_level}")


def get_logger(name: str) -> logging.Logger:
    """
    获取命名的日志记录器。

    Args:
        name: 日志记录器名称

    Returns:
        命名的日志记录器
    """
    return logging.getLogger(name)