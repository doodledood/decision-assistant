import io
import sys
import traceback

from .base import CodeExecutor


class LocalCodeExecutor(CodeExecutor):
    def execute(self, code: str) -> str:
        captured_output = io.StringIO()
        saved_stdout = sys.stdout
        sys.stdout = captured_output

        local_vars = {}

        try:
            for line in code.splitlines(keepends=False):
                if not line:
                    continue

                exec(code, None, local_vars)
        except:
            return f'Error executing code: {traceback.format_exc()}'
        finally:
            sys.stdout = saved_stdout

        res = captured_output.getvalue()

        return res
