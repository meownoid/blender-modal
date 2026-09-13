# blender-modal

`blender-modal` is a direct command-line client for cached Blender GPU renders
on Modal. It has no web service or persistent CPU workers. Projects and results
live in a versioned Modal Volume; each render starts a temporary GPU-only Modal
app that can be detached from the terminal.

## Install and use

~~~sh
uv sync --all-groups
uv run modal setup
uv run blender-modal scene upload ./project/scenes/shot.blend --name shot
uv run blender-modal scene upload ./project --blend scenes/shot.blend --name shot
uv run blender-modal scene list
uv run blender-modal scene render SCENE_ID --frames 1:120 --gpu L4 --instances 4
uv run blender-modal result download RESULTS_ID --output ./renders
~~~

Use `--detach` to submit a render and return immediately, then inspect it with
`job info JOB_ID` or stop it with `job cancel JOB_ID`. `--json` provides structured
machine-readable output; `-v/--verbose` logs detailed progress to standard
error. `--volume` and `--environment` select a non-default Modal workspace.
These global options work before the resource, between the resource and action,
or after the action. Later explicit values take precedence:

~~~sh
uv run blender-modal --json scene list
uv run blender-modal scene --json list
uv run blender-modal scene list --json
~~~

Scene and result IDs display their first 12 characters in human-readable output.
Commands accept the full ID or any nonempty unique prefix, including `result list
--scene`. If a prefix matches multiple items, the error lists their full IDs;
use a longer prefix or a full ID. Full IDs remain in JSON output and storage,
so existing scenes and cached renders keep working. Job IDs are unchanged.

## Commands

| Command | Purpose |
| --- | --- |
| `scene upload [ROOT] [options]` | Upload an immutable scene and its assets |
| `scene list` | List uploaded scenes |
| `scene render SCENE_ID --frames FRAMES [options]` | Render missing frames |
| `scene remove SCENE_ID [--dry-run]` | Remove an uploaded scene |
| `job list` | List stored render jobs |
| `job info JOB_ID [--watch]` | Inspect job state, workers, and per-job billing |
| `job cancel JOB_ID` | Cancel an active render job |
| `result list [--scene SCENE_ID]` | List result sets, optionally filtered by scene |
| `result download RESULTS_ID --output DIRECTORY [options]` | Download completed PNG frames |
| `result remove RESULTS_ID [--frames FRAMES] [--dry-run]` | Remove a result set or selected frames |
| `cleanup [--dry-run] [--force]` | Remove abandoned staging and unreferenced blobs |
| `billing` | Show workspace billing rates and summary |

Use `--help` at any command level for available options. The previous command
paths have been replaced: for example, `upload` becomes `scene upload`,
`list results` becomes `result list`, and `info JOB_ID` becomes `job info JOB_ID`.
The previous no-ID `info` command is split into `job list` and `billing`.
With `--json`, these return `{"jobs": [...]}` and `{"billing": {...}}`, respectively.
Other command payloads retain their existing fields; scene and result listings
emit one JSON object per line.

## Uploads and rendering

`scene upload` preserves project-relative paths and accepts either a `.blend` file
(uploading only that file by default), or a project root with an explicit
entrypoint `.blend` (uploading every regular file below the root). Use `--include`
to add selected files or directories to a direct `.blend` upload. It hashes every regular file;
the same scene ID is skipped, while unchanged file blobs are reused by SHA-256.
Scanning skips files that never affect rendering: OS metadata (`.DS_Store`,
`Thumbs.db`, `desktop.ini`), editor backups (`*~`, `*.swp`), Blender backup
saves (`*.blend1`, …), Python caches (`__pycache__`, `*.pyc`), and version
control internals (`.git`, `.hg`, `.svn`). Explicitly `--include`d files are
always kept.
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
