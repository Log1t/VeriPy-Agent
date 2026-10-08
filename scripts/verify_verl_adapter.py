#!/usr/bin/env python3
from agent_system.environments.base import EnvironmentManagerBase

from veripy_agent.verl_adapter import VeriPyEnvironmentManager


def main() -> None:
    if not issubclass(VeriPyEnvironmentManager, EnvironmentManagerBase):
        raise TypeError("VeriPy manager does not satisfy the installed verl-agent base class")
    print("verl-adapter-contract: passed")


if __name__ == "__main__":
    main()
