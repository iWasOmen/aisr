"""
日志工具模块，为AISR系统提供日志记录功能。
"""
import logging
import sys
from typing import Optional, Dict
import colorama
from colorama import Fore, Style
try:
    import colorlog
    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False
    print("colorlog不可用，将使用基本的colorama颜色支持。安装: pip install colorlog")

# 初始化colorama
colorama.init(autoreset=True)

# 可用的颜色映射
COLORS = {
    "red": "red",
    "green": "green",
    "yellow": "yellow",
    "blue": "blue",
    "magenta": "magenta",
    "cyan": "cyan",
    "white": "white"
}

# 全局模块颜色配置字典
MODULE_COLORS: Dict[str, str] = {}

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    设置日志配置。

    Args:
        log_level: 日志级别 ("DEBUG", "INFO", "WARNING", "ERROR")
        log_file: 可选的日志文件路径
    """
    # 转换日志级别字符串为常量
    level = getattr(logging, log_level.upper(), logging.INFO)

    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 清除现有的处理器
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    # 如果提供了日志文件，添加文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        root_logger.addHandler(file_handler)

    # 设置特定库的日志级别
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    logging.info(f"日志系统已配置，级别: {log_level}")

def get_logger(name: str, color: Optional[str] = None) -> logging.Logger:
    """
    获取命名的日志记录器，并可选地设置其颜色。

    Args:
        name: 日志记录器名称
        color: 日志颜色 (red, green, yellow, blue, magenta, cyan, white)

    Returns:
        命名的日志记录器
    """
    logger = logging.getLogger(name)

    # 如果指定了颜色并且是有效的颜色
    if color and color in COLORS:
        MODULE_COLORS[name] = color

        # 防止日志消息被传递到父记录器
        logger.propagate = False

        # 清除现有的处理器
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

        # 创建处理器
        handler = logging.StreamHandler(sys.stdout)

        # 根据是否可用colorlog选择不同的格式化器
        if COLORLOG_AVAILABLE:
            # 创建颜色映射
            log_colors = {
                'DEBUG': 'cyan',
                'INFO': color,
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }

            formatter = colorlog.ColoredFormatter(
                '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                log_colors=log_colors
            )
        else:
            # 使用基本的colorama颜色
            class SimpleColoredFormatter(logging.Formatter):
                def format(self, record):
                    formatted_message = super().format(record)
                    color_code = getattr(Fore, COLOR_MAP.get(MODULE_COLORS.get(record.name, 'WHITE'), 'WHITE'))
                    return f"{color_code}{formatted_message}{Style.RESET_ALL}"

            formatter = SimpleColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger