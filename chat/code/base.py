from abc import ABC, abstractmethod


class CodeExecutor(ABC):
    @abstractmethod
    def execute(self, code: str) -> str:
        pass
