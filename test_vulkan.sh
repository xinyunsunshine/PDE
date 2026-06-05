#!/bin/bash

export VULKAN_SDK=/PROJECT_ROOT/vulkan/1.4.341.1/x86_64
export PATH=$VULKAN_SDK/bin:$PATH
export LD_LIBRARY_PATH=$VULKAN_SDK/lib:$LD_LIBRARY_PATH
export VK_LAYER_PATH=$VULKAN_SDK/share/vulkan/explicit_layer.d
# SAPIEN checks VK_ICD_FILENAMES (not VK_DRIVER_FILES); system has nvidia_icd.x86_64.json, not nvidia_icd.json
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
export VK_DRIVER_FILES=$VK_ICD_FILENAMES


echo "=== Vulkan ICD Verification ==="
echo ""
echo "1. NVIDIA driver:"
nvidia-smi --query-gpu=driver_version,name --format=csv,noheader 2>/dev/null || echo "nvidia-smi failed"
echo ""
echo "2. VK_ICD_FILENAMES: ${VK_ICD_FILENAMES:-'(not set)'}"
echo "3. VK_DRIVER_FILES:  ${VK_DRIVER_FILES:-'(not set)'}"
echo ""
echo "4. ICD files in standard locations:"
ls -la /usr/share/vulkan/icd.d/*.json 2>/dev/null || echo "  /usr/share/vulkan/icd.d: none"
ls -la /etc/vulkan/icd.d/*.json 2>/dev/null || echo "  /etc/vulkan/icd.d: none"
echo ""
echo "5. SAPIEN import test:"
source /PROJECT_ROOT/.venv_maniskill_libero/bin/activate 2>/dev/null
python -c "import sapien; print('OK - no Vulkan ICD warning')" 2>&1