import os
import tempfile
from typing import Optional, Set

import docker
from docker.errors import ContainerError
from halo import Halo

from .base import CodeExecutor


class DockerCodeExecutor(CodeExecutor):
    def __init__(self, client: Optional[docker.DockerClient] = None,
                 image_tag: str = 'python-executor:latest',
                 base_image: str = 'python:3.11-slim',
                 default_dependencies: Optional[Set[str]] = None,
                 spinner: Optional[Halo] = None):
        self.client = client or docker.from_env()
        self.image_tag = image_tag
        self.base_image = base_image
        self.default_dependencies = default_dependencies or {'requests'}
        self.spinner = spinner

    def build_image_with_code(self, python_code: str, dependencies: Optional[Set[str]] = None):
        spinner_text = None
        if self.spinner is not None:
            spinner_text = self.spinner.text
            self.spinner.start('Building Docker image...')

        run_install_template = ('RUN pip install {package} --trusted-host pypi.org --trusted-host '
                                'files.pythonhosted.org')
        run_commands = [run_install_template.format(package=package) for package in dependencies or []]
        run_commands_str = '\n'.join(run_commands)

        # Helper function to construct Dockerfile
        dockerfile = f'''
        FROM {self.base_image}
        
        RUN apt-get update
        
        {run_commands_str}
        
        COPY script.py /code/script.py
        
        WORKDIR /code
        '''

        # Create a temporary build directory
        with tempfile.TemporaryDirectory() as build_dir:
            dockerfile_path = os.path.join(build_dir, 'Dockerfile')

            # Write the Dockerfile to the build directory
            with open(dockerfile_path, 'w') as f:
                f.write(dockerfile)

            # Write script file
            with open(os.path.join(build_dir, 'script.py'), 'w') as f:
                f.write(python_code)

            # Build the image
            image, build_log = self.client.images.build(path=build_dir, tag=self.image_tag, rm=True)

        if self.spinner is not None:
            self.spinner.stop_and_persist(symbol='🐳', text='Docker image built.')

            if spinner_text is not None:
                self.spinner.start(spinner_text)

        return image

    def execute(self, code: str) -> str:
        try:
            # Ensure the image is built before execution
            self.build_image_with_code(code, dependencies=self.default_dependencies)
        except Exception as e:
            return f'Failed to build Docker image (did not run code yet): {e}'

        # Run the code inside the container
        try:
            container = self.client.containers.run(
                image=self.image_tag,
                command=["python", "script.py"],
                remove=True,
                stdout=True,
                stderr=True,
                detach=False
            )
            return container.decode('utf-8')
        except ContainerError as e:
            return e.stderr.decode('utf-8')
