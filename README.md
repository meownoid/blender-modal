# blender-modal

`blender-modal` is a direct command-line client for cached Blender GPU renders
on Modal. It has no web service or persistent CPU workers. Projects and results
live in a versioned Modal Volume; each render starts a temporary GPU-only Modal
app that can be detached from the terminal.

## Install and use

~~~sh
uv sync --all-groups
uv run modal setup
uv run blender-modal upload ./project/scenes/shot.blend --name shot
uv run blender-modal upload ./project --blend scenes/shot.blend --name shot
uv run blender-modal list scenes
uv run blender-modal render SCENE_ID --frames 1:120 --gpu L4 --instances 4
uv run blender-modal download RESULTS_ID --output ./renders
~~~

Use `--detach` to submit a render and return immediately, then inspect it with
`info JOB_ID` or stop it with `cancel JOB_ID`. `--json` provides structured
machine-readable output; `-v/--verbose` logs detailed progress to standard
error. `--volume` and `--environment` select a non-default Modal workspace.

`upload` preserves project-relative paths and accepts either a `.blend` file
(uploading only that file by default), or a project root with an explicit
entrypoint `.blend` (uploading every regular file below the root). Use `--include`
to add selected files or directories to a direct `.blend` upload. It hashes every regular file;
the same scene ID is skipped, while unchanged file blobs are reused by SHA-256.
The GPU worker validates external Blender assets when it opens the scene.
Output is human-readable with colors by default, with a minimal status line
during long operations. Pass `-v` for detailed progress on standard error
(hashing, blob transfer, scene materialization), or `--json` for
machine-readable output on standard output; both can be combined.

`cleanup` retains unreferenced upload blobs for 24 hours to avoid interfering with
an active upload. Use `cleanup --force` to immediately remove unreferenced blobs
from an interrupted upload.

The bundled GPU image retains the pinned Blender 5.2.1 and FLIP Fluids Demo
integration. Cycles renders PNG results with OptiX or CUDA; scene settings are
preserved unless samples, tile size, or resolution overrides are supplied.

For local checks:

~~~sh
uv run pytest -q
uv run ruff check blender_modal tests renderer
uv run mypy blender_modal
~~~
