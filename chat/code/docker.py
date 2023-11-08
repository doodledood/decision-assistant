import os
from typing import Optional

import docker
from docker.errors import ContainerError

from .base import CodeExecutor


class DockerCodeExecutor(CodeExecutor):
    def __init__(self, client: Optional[docker.DockerClient] = None, image_tag: str = 'python-executor:latest',
                 base_image: str = 'python:3.11-slim'):
        self.client = client or docker.from_env()
        self.image_tag = image_tag
        self.base_image = base_image
        self.previously_built_image = None

    def build_image(self):
        if self.previously_built_image is None:
            return self.previously_built_image

        # Helper function to construct Dockerfile
        dockerfile = f'''
        FROM {self.base_image}
        WORKDIR /app
        '''

        # Create a temporary build directory
        build_dir = 'tmp_build_dir'
        os.makedirs(build_dir, exist_ok=True)
        dockerfile_path = os.path.join(build_dir, 'Dockerfile')

        # Write the Dockerfile to the build directory
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile)

        # Build the image
        image, build_log = self.client.images.build(path=build_dir, tag=self.image_tag, rm=True)

        # Optionally, handle build_log to check for errors or for logging

        # Clean up the temporary build directory
        os.remove(dockerfile_path)
        os.rmdir(build_dir)

        self.previously_built_image = image

        return image

    def execute(self, code: str) -> str:
        # Ensure the image is built before execution
        self.build_image()

        # Run the code inside the container
        try:
            container = self.client.containers.run(
                image=self.image_tag,
                command=["python", "-c", code],
                remove=True,
                stdout=True,
                stderr=True,
                detach=False
            )
            return container.decode('utf-8')

        except ContainerError as e:
            return e.stderr.decode('utf-8')
