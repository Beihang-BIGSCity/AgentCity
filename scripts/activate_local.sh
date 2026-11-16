# Source from bash/zsh while in the project root: source scripts/activate_local.sh
if [ ! -f .venv/bin/activate ]; then
    echo "Run this script from the AgentCity root after creating .venv" >&2
    return 1
fi
source .venv/bin/activate
export AGENTCITY_LIBCITY_PYTHON="$PWD/.venv-libcity/bin/python"
# Device and training budgets come from the task/CLI, not shell activation.
