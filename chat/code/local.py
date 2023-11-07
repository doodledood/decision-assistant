import os
import subprocess

from .base import CodeExecutor


class LocalCodeExecutor(CodeExecutor):
    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir

    def execute(self, code: str) -> str:
        # Save the code to a temporary file in the workspace directory
        os.makedirs(self.workspace_dir, exist_ok=True)
        file_path = os.path.join(self.workspace_dir, 'script.py')

        with open(file_path, 'w') as file:
            file.write(code)

        # Execute the code and capture the output
        result = subprocess.run(
            ['python', file_path],
            capture_output=True,
            text=True
        )

        # Clean up the temporary file
        os.remove(file_path)

        # Return the result, including stdout and stderr
        return result.stdout if result.returncode == 0 else result.stderr
