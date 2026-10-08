#!/usr/bin/env bash
# Compatibility entry point for existing Colab users.
set -euo pipefail
export PDE_DEMO_ENV=${PDE_DEMO_ENV:-${PDE_COLAB_ENV:-/content/pde-env}}
export PDE_DEMO_DEPS=${PDE_DEMO_DEPS:-${PDE_COLAB_DEPS:-/content/pde-deps}}
export PDE_SKIP_SYSTEM_DEPS=${PDE_SKIP_SYSTEM_DEPS:-0}
exec bash "$(dirname "${BASH_SOURCE[0]}")/install_demo.sh" "$@"
