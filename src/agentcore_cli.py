"""
agentcore_cli.py
================
AgentCore Runtime deployment helper.

Follows the same configure -> deploy workflow as the AgentCore starter
toolkit CLI (`agentcore configure` / `agentcore launch`), but talks to the
bedrock-agentcore-control API directly and ships the agent as a direct
code deployment (ZIP on S3), so no Docker build is needed.

Usage:
    cli = AgentCoreCLI(region_name="us-east-1")
    cli.configure_runtime(agent_name=..., execution_role_arn=..., ...)
    cli.deploy()
    runtime_arn = cli.deployed_runtime_arn()
"""

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from typing import Optional

import boto3

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# AgentCore Runtime runs on Graviton (ARM64) Linux, so dependencies are
# installed as aarch64 manylinux wheels regardless of the local OS.
_TARGET_PLATFORMS = ("manylinux2014_aarch64", "manylinux_2_28_aarch64")

_VALID_NETWORK_MODES = ("PUBLIC", "VPC")
_VALID_PROTOCOLS     = ("HTTP", "MCP", "A2A")


def _walk_files(directory: str):
    """Yield every file under directory, skipping bytecode caches."""
    for dirpath, dirnames, filenames in os.walk(directory):
        dirnames[:] = [d for d in dirnames if d != '__pycache__']
        for filename in filenames:
            if not filename.endswith('.pyc'):
                yield os.path.join(dirpath, filename)


def _arcname(path: str, base: str) -> str:
    """Path of a file inside the ZIP, relative to base, with forward slashes."""
    return os.path.relpath(path, base).replace(os.sep, '/')


class AgentCoreCLI:
    """Configure, package, and deploy an agent to AgentCore Runtime."""

    def __init__(self, region_name: str, project_root: str = PROJECT_ROOT):
        self.region_name  = region_name
        self.project_root = project_root
        self.control      = boto3.client('bedrock-agentcore-control', region_name=region_name)
        self.s3           = boto3.client('s3', region_name=region_name)
        self.runtime_config: Optional[dict] = None
        self._runtime_arn: Optional[str] = None

    # ─────────────────────────────────────────────
    # STEP 1 - CONFIGURE
    # ─────────────────────────────────────────────
    def configure_runtime(
        self,
        agent_name: str,
        execution_role_arn: str,
        artifact_bucket: str,
        environment_variables: dict,
        entrypoint: str = "main.py",
        source_paths: tuple = ("main.py", "config.py", "src"),
        requirements_file: str = "requirements.txt",
        network_mode: str = "PUBLIC",
        server_protocol: str = "HTTP",
        python_runtime: str = "PYTHON_3_12",
        description: str = "",
    ) -> dict:
        """
        Validate and store the runtime configuration used by deploy().

        Args:
            agent_name: Runtime name ([a-zA-Z][a-zA-Z0-9_]{0,47}).
            execution_role_arn: IAM role the runtime assumes for AWS access.
            artifact_bucket: S3 bucket that receives the code ZIP.
            environment_variables: Application settings for the deployed code.
            entrypoint: File (relative to the project root) that starts the app.
            source_paths: Project files/directories bundled with the dependencies.
            requirements_file: pip requirements installed into the bundle.
            network_mode: PUBLIC or VPC.
            server_protocol: HTTP, MCP, or A2A.
            python_runtime: Managed runtime, e.g. PYTHON_3_12.
            description: Human-readable runtime description.

        Returns:
            The stored runtime configuration.
        """
        if network_mode not in _VALID_NETWORK_MODES:
            raise ValueError(f"network_mode must be one of {_VALID_NETWORK_MODES}")
        if server_protocol not in _VALID_PROTOCOLS:
            raise ValueError(f"server_protocol must be one of {_VALID_PROTOCOLS}")
        if not execution_role_arn:
            raise ValueError("execution_role_arn is required")
        if not os.path.isfile(os.path.join(self.project_root, entrypoint)):
            raise FileNotFoundError(f"Entrypoint not found: {entrypoint}")

        missing = [k for k, v in environment_variables.items() if v in (None, '')]
        if missing:
            raise ValueError(f"Runtime environment variables have no value: {missing}")

        self.runtime_config = {
            'agent_name':            agent_name,
            'execution_role_arn':    execution_role_arn,
            'artifact_bucket':       artifact_bucket,
            'artifact_key':          f"agentcore-artifacts/{agent_name}/agentcore-runtime.zip",
            'environment_variables': {k: str(v) for k, v in environment_variables.items()},
            'entrypoint':            entrypoint,
            'source_paths':          tuple(source_paths),
            'requirements_file':     requirements_file,
            'network_mode':          network_mode,
            'server_protocol':       server_protocol,
            'python_runtime':        python_runtime,
            'description':           description,
        }
        print(f"  Configured runtime '{agent_name}' "
              f"({network_mode} network, {server_protocol} protocol, {python_runtime})")
        return self.runtime_config

    # ─────────────────────────────────────────────
    # STEP 2 - DEPLOY
    # ─────────────────────────────────────────────
    def deploy(self, wait: bool = True, timeout: int = 900) -> dict:
        """
        Package the code, upload it to S3, and create or update the runtime.

        Args:
            wait: Block until the runtime reaches READY.
            timeout: Maximum seconds to wait for READY.

        Returns:
            The final get_agent_runtime response (or the create/update
            response when wait is False).
        """
        if self.runtime_config is None:
            raise RuntimeError("Call configure_runtime() before deploy()")
        cfg = self.runtime_config

        zip_path = self._build_artifact()
        try:
            print(f"  Uploading code to s3://{cfg['artifact_bucket']}/{cfg['artifact_key']}")
            self.s3.upload_file(zip_path, cfg['artifact_bucket'], cfg['artifact_key'])
        finally:
            os.remove(zip_path)

        params = {
            'agentRuntimeArtifact': {
                'codeConfiguration': {
                    'code': {'s3': {'bucket': cfg['artifact_bucket'],
                                    'prefix': cfg['artifact_key']}},
                    'runtime':    cfg['python_runtime'],
                    'entryPoint': [cfg['entrypoint']],
                }
            },
            'roleArn':               cfg['execution_role_arn'],
            'networkConfiguration':  {'networkMode': cfg['network_mode']},
            'protocolConfiguration': {'serverProtocol': cfg['server_protocol']},
            'environmentVariables':  cfg['environment_variables'],
            'description':           cfg['description'],
        }

        existing = self._find_runtime(cfg['agent_name'])
        if existing:
            runtime_id = existing['agentRuntimeId']
            print(f"  Updating existing runtime {runtime_id}...")
            response = self.control.update_agent_runtime(agentRuntimeId=runtime_id, **params)
        else:
            print(f"  Creating runtime '{cfg['agent_name']}'...")
            response = self.control.create_agent_runtime(
                agentRuntimeName=cfg['agent_name'], **params
            )
            runtime_id = response['agentRuntimeId']

        self._runtime_arn = response['agentRuntimeArn']
        if not wait:
            return response
        return self._wait_for_ready(runtime_id, timeout)

    # ─────────────────────────────────────────────
    # STEP 3 - RESULT
    # ─────────────────────────────────────────────
    def deployed_runtime_arn(self) -> str:
        """Return the ARN of the runtime created or updated by deploy()."""
        if not self._runtime_arn:
            raise RuntimeError("No runtime deployed yet - call deploy() first")
        return self._runtime_arn

    # ─────────────────────────────────────────────
    # INTERNALS
    # ─────────────────────────────────────────────
    def _find_runtime(self, name: str) -> Optional[dict]:
        """Return the runtime summary with this name, if one exists."""
        kwargs = {}
        while True:
            page = self.control.list_agent_runtimes(**kwargs)
            for runtime in page.get('agentRuntimes', []):
                if runtime.get('agentRuntimeName') == name:
                    return runtime
            if not page.get('nextToken'):
                return None
            kwargs['nextToken'] = page['nextToken']

    def _wait_for_ready(self, runtime_id: str, timeout: int) -> dict:
        """Poll until the runtime is READY; raise if it fails or times out."""
        deadline = time.time() + timeout
        print("  Waiting for runtime to become READY", end='', flush=True)
        while time.time() < deadline:
            runtime = self.control.get_agent_runtime(agentRuntimeId=runtime_id)
            status = runtime.get('status')
            if status == 'READY':
                print(" ready.")
                return runtime
            if status and status.endswith('FAILED'):
                print(f" {status}.")
                raise RuntimeError(
                    f"Runtime {runtime_id} entered {status}: "
                    f"{runtime.get('failureReason', 'no reason given')}"
                )
            print('.', end='', flush=True)
            time.sleep(10)
        raise TimeoutError(f"Runtime {runtime_id} not READY after {timeout}s")

    def _install_dependencies(self) -> str:
        """pip-install aarch64 wheels into a cached build directory."""
        cfg = self.runtime_config
        requirements = os.path.join(self.project_root, cfg['requirements_file'])
        build_dir = os.path.join(self.project_root, '.agentcore_build')
        marker = os.path.join(build_dir, '.requirements.sha256')

        python_version = cfg['python_runtime'].replace('PYTHON_', '').replace('_', '.')
        with open(requirements, 'rb') as f:
            fingerprint = hashlib.sha256(f.read() + python_version.encode()).hexdigest()

        if os.path.isfile(marker):
            with open(marker, encoding='utf-8') as f:
                if f.read().strip() == fingerprint:
                    print("  Dependencies unchanged - reusing .agentcore_build/")
                    return build_dir

        print(f"  Installing dependencies for Python {python_version} (aarch64)...")
        shutil.rmtree(build_dir, ignore_errors=True)
        command = [
            sys.executable, '-m', 'pip', 'install', '--quiet',
            '-r', requirements, '--target', build_dir,
            '--implementation', 'cp', '--python-version', python_version,
            '--only-binary=:all:',
        ]
        for platform in _TARGET_PLATFORMS:
            command += ['--platform', platform]
        subprocess.run(command, check=True)

        with open(marker, 'w', encoding='utf-8') as f:
            f.write(fingerprint)
        return build_dir

    def _build_artifact(self) -> str:
        """Create the deployment ZIP (dependencies + app sources); return its path."""
        build_dir = self._install_dependencies()
        fd, zip_path = tempfile.mkstemp(suffix='.zip', prefix='agentcore-')
        os.close(fd)

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Dependencies go at the ZIP root so they are importable at runtime.
            for full in _walk_files(build_dir):
                if not full.endswith('.sha256'):
                    zf.write(full, _arcname(full, build_dir))

            # Application sources: Python files only (no backups or notes).
            for source in self.runtime_config['source_paths']:
                path = os.path.join(self.project_root, source)
                files = [path] if os.path.isfile(path) else _walk_files(path)
                for full in files:
                    if full.endswith('.py'):
                        zf.write(full, _arcname(full, self.project_root))

        size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        print(f"  Built deployment package ({size_mb:.1f} MB)")
        return zip_path
