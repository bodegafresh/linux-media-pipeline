# Workstation Protection

This project must not destabilize the Fedora workstation used for OBS, Unreal
Engine, DaVinci Resolve, Blender, Mesa/RADV Vulkan, OpenCL, and C++ development.

Do not run these automatically:

```bash
sudo dnf install rocm
sudo dnf install rocm-devel
sudo dnf install --allowerasing ...
sudo dnf swap ...
sudo dnf distro-sync ...
sudo dnf remove ...
```

Any package change must first go through:

```bash
./scripts/snapshot-workstation.sh
./scripts/dnf-migraphx-dry-run.sh
```

Only proceed if the dry run does not remove, replace, downgrade, or conflict
with Mesa, Vulkan, OpenCL ICD Loader, FFmpeg, OBS, GCC, Clang, or kernel
packages.
