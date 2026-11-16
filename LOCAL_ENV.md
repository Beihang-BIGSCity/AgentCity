# Local Python environments

Use Python 3.12. Create `.venv` for orchestration and `.venv-libcity` for LibCity.
Install `requirements.local-agent.txt` and `requirements.local-libcity.txt` into the
respective environments, then run `source scripts/activate_local.sh` from the project root.

The activation script selects interpreters only. Configure API credentials through
`AGENTCITY_API_KEY` / `AGENTCITY_API_BASE` or an ignored local harness profile.
Use `--gpu on` for an accelerator or `--gpu off` for CPU. An explicit
`AGENTCITY_DEVICE=mps` or `cuda:0` selects an accelerator when GPU is enabled.

See `documentation/runtime_configuration.md` for iteration/final training settings.
Local installation logs, datasets, backups and experiment records belong in `local-env/`.
