import sys

from veripy_agent.environment.paths import SandboxPaths
from veripy_agent.environment.worker import SandboxWorker

SandboxWorker(SandboxPaths("/inputs", "/scratch")).serve(sys.stdin)
