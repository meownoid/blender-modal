"""Enable the trusted FLIP Fluids add-on before an uploaded scene is opened."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import addon_utils
import bpy


def configure_material_library(module: str) -> None:
    """Keep FLIP's scene-persisted library path valid in a container render."""
    material_library = importlib.import_module(f"{module}.materials.material_library")
    material_library_objects = importlib.import_module(
        f"{module}.objects.flip_fluid_material_library"
    )
    material_library_class = material_library_objects.FLIPFluidMaterialLibrary

    library_path = str(Path(material_library.__file__).resolve().parent / "material_library")
    material_library_root = Path(library_path)
    original_check_icons_initialized = material_library_class.check_icons_initialized
    original_generate_icons = material_library_class._generate_material_library_icons

    def check_icons_initialized(library: object) -> None:
        # `library_path` is stored in .blend files. A scene made on another host can therefore
        # point at its creator's add-on directory before FLIP's load handler has reset it.
        if library.library_path != library_path:
            library.initialize(library_path)
            return
        original_check_icons_initialized(library)

    material_library_class.check_icons_initialized = check_icons_initialized

    def generate_material_library_icons(library: object) -> None:
        original_generate_icons(library)
        for data_library in tuple(bpy.data.libraries):
            if not data_library.filepath:
                continue
            filepath = Path(bpy.path.abspath(data_library.filepath)).resolve()
            if not filepath.is_relative_to(material_library_root):
                continue
            try:
                # FLIP appends these files only to produce previews and removes the copied
                # materials before returning. Do not leave the now-unreferenced libraries in
                # bpy.data: BlendRender correctly treats all remaining libraries as scene input.
                bpy.data.libraries.remove(data_library, do_unlink=False)
            except RuntimeError:
                # Preserve any unexpected linked data instead of unlinking it implicitly.
                continue

    material_library_class._generate_material_library_icons = generate_material_library_icons

    @bpy.app.handlers.persistent
    def reset_material_library(_: object) -> None:
        library = bpy.context.scene.flip_fluid_material_library
        if library.library_path != library_path:
            library.initialize(library_path)

    bpy.app.handlers.load_post.append(reset_material_library)


def run() -> None:
    module = os.getenv("FLIP_FLUIDS_ADDON", "").strip()
    if not module:
        raise RuntimeError("FLIP_FLUIDS_ADDON is required when running the FLIP Fluids bootstrap")

    addon_utils.modules_refresh()
    _, loaded = addon_utils.check(module)
    if not loaded:
        # FLIP Fluids reads its preferences while registering. Blender creates that entry only
        # when the add-on is enabled with default_set=True; nothing is persisted unless Blender
        # subsequently saves user preferences.
        addon_utils.enable(module, default_set=True, persistent=False)
    _, loaded = addon_utils.check(module)
    if not loaded:
        raise RuntimeError(f"Unable to enable bundled Blender add-on: {module}")
    configure_material_library(module)
    print(f"BLENDRENDER_FLIP_FLUIDS enabled {module}", flush=True)


run()
