from logging import Logger as PythonLogger

import rclpy
from typing_extensions import override
from typing import Union


class Logger:
    def __init__(self, name: str, debug: bool = False) -> None:
        self.name = name
        self.debug = debug

    def warning(self, message: Union[str, Exception]):
        raise NotImplementedError()

    def info(self, message: Union[str, Exception]):
        raise NotImplementedError()

    def error(self, message: Union[str, Exception]):
        raise NotImplementedError()

    def format_message(self, message: Union[str, Exception]) -> str:
        if isinstance(message, Exception):
            return f"[{self.name}]: {message}\n{message.__traceback__}"
        return f"[{self.name}]: {message}"


class ROSLogger(Logger):
    def __init__(self, name: str, debug: bool = False) -> None:
        super().__init__(name, debug)

    @override
    def warning(self, message: Union[str, Exception]):
        if self.debug:
            rclpy.logging.get_logger('kp_based_detector').warn(self.format_message(message))
            # rospy.logwarn(self.format_message(message))

    @override
    def info(self, message: Union[str, Exception]):
        if self.debug:
            rclpy.logging.get_logger('kp_based_detector').info(self.format_message(message))

    @override
    def error(self, message: Union[str, Exception]):
        if self.debug:
            rclpy.logging.get_logger('kp_based_detector').error(self.format_message(message))


class BasicLogger(Logger):
    def __init__(self, name: str, debug: bool = False) -> None:
        super().__init__(name, debug)

    @override
    def warning(self, message: Union[str, Exception]):
        if self.debug:
            print(self.format_message(message))

    @override
    def info(self, message: Union[str, Exception]):
        if self.debug:
            print(self.format_message(message))

    @override
    def error(self, message: Union[str, Exception]):
        if self.debug:
            print(self.format_message(message))


class PyLogger(Logger):
    def __init__(self, name: str, debug: bool = False) -> None:
        super().__init__(name, debug)
        self.logger = PythonLogger(name)

    def warning(self, message: Union[str, Exception]):
        if self.debug:
            self.logger.warning(self.format_message(message))

    def info(self, message: Union[str, Exception]):
        if self.debug:
            self.logger.info(self.format_message(message))

    def error(self, message: Union[str, Exception]):
        if self.debug:
            self.logger.error(self.format_message(message))
