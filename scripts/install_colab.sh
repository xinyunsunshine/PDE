#!/usr/bin/env bash
# Inference-only Colab environment; does not alter the notebook's Python kernel.
set -euo pipefail
PDE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PDE_ENV=${PDE_COLAB_ENV:-/content/pde-env}
PDE_DEPS=${PDE_COLAB_DEPS:-/content/pde-deps}
OPENPI_REV=c5dc4b9296a1a4739bf52828f28a579f12dce763
LIBERO_REV=0c5e40cc4ae63e09c14e7df6f74481e9ee8585f7
mkdir -p "$PDE_DEPS" "$HOME/.libero"
apt-get update -qq
apt-get install -y -qq libegl1 libgl1 libgles2 libglfw3 libosmesa6 ffmpeg git
python -m pip install -q uv
uv venv --python 3.11.14 --allow-existing "$PDE_ENV"
for component in openpi LIBERO; do
    if [ ! -d "$PDE_DEPS/$component/.git" ]; then
        git clone --filter=blob:none "https://github.com/RLinf/$component.git" "$PDE_DEPS/$component"
    fi
done
git -C "$PDE_DEPS/openpi" checkout "$OPENPI_REV"
git -C "$PDE_DEPS/LIBERO" checkout "$LIBERO_REV"
# OpenPI's LeRobot revision is required; the latest PyPI LeRobot is not equivalent.
uv pip install --python "$PDE_ENV/bin/python" \
    -e "$PDE_ROOT/RLinf[embodied]" -e "$PDE_ROOT[integration]" \
    -e "$PDE_DEPS/openpi/packages/openpi-client" -e "$PDE_DEPS/openpi" \
    -e "$PDE_DEPS/LIBERO" \
    'lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5' \
    'robosuite==1.4.1' 'mujoco==3.2.7' bddl easydict cloudpickle \
    'numpy<2' 'imageio[ffmpeg]' 'transformers==4.53.2'
# Match RLinf's OpenPI installation procedure in this isolated environment.
"$PDE_ENV/bin/python" - <<'PY'
from pathlib import Path
import shutil
import openpi
import transformers
source = Path(openpi.__file__).parent / 'models_pytorch/transformers_replace'
shutil.copytree(source, Path(transformers.__file__).parent, dirs_exist_ok=True)
PY
PATH="$PDE_ENV/bin:$PATH" bash "$PDE_ROOT/RLinf/requirements/embodied/download_assets.sh" --assets openpi
"$PDE_ENV/bin/python" - <<'PYCODE'
from pathlib import Path
import shutil
from openpi.shared.download import get_cache_dir
# Match the cache key of gs://big_vision/paligemma_tokenizer.model.
destination = get_cache_dir() / 'big_vision/paligemma_tokenizer.model'
destination.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(Path.home() / '.cache/openpi/paligemma_tokenizer.model', destination)
PYCODE
uv pip freeze --python "$PDE_ENV/bin/python" > "$PDE_ENV/installed-requirements.txt"
echo "Installed. Notebook inference Python: $PDE_ENV/bin/python"
