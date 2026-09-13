# blender-modal

`blender-modal` is a direct command-line client for cached Blender GPU renders
on [Modal](https://modal.com).

## Installation

You need Python 3.12 or newer, `uv`, and a Modal account. From the root of this
repository, install the CLI and connect your Modal account:

~~~sh
uv tool install --reinstall .
uvx --from 'modal>=1.5.4,<2' modal setup
~~~

If the command is not on your PATH, run `uv tool update-shell` and
restart your shell.

## Quick start

### 1. Upload your scene

For a self-contained `.blend` file:

~~~sh
blender-modal scene upload ./shot.blend --name shot
~~~

If your scene uses external textures, linked files, or simulation caches, upload
the project directory instead. The `--blend` path is relative to that directory:

~~~sh
blender-modal scene upload ./project --blend scenes/shot.blend --name shot
~~~

Copy the scene ID printed by the upload command. Use it in place of `SCENE_ID`
below; `shot` is a display name, not an ID.

### 2. Render frames

~~~sh
blender-modal scene render SCENE_ID --frames 1:120 --gpu L4 --instances 4
~~~

This renders frames 1 through 120, inclusive, using up to four GPU workers.
The command prints a job ID and a results ID, then waits for rendering to finish.

### 3. Download the results

Replace `RESULTS_ID` with the results ID from the render command:

~~~sh
blender-modal result download RESULTS_ID --output ./renders
~~~

Your PNGs are saved as `frame_000001.png`, `frame_000002.png`, and so on.
To find IDs again, use `scene list`, `job list`, or `result list`.

## In-depth usage

### Scenes, jobs, and results

A **scene** is an immutable snapshot of your uploaded files. A **job** is one
render submission. A **result set** stores completed frames for a scene and a
specific render configuration; multiple jobs can contribute to it.

Scene and result IDs appear as 12-character prefixes in human-readable output.
Commands accept full IDs or any nonempty unique prefix, including for
`result list --scene`. If a prefix is ambiguous, the error lists the matching
full IDs so you can choose a longer prefix. Job IDs must be supplied in full.

### Uploading scenes and assets

Choose the upload form that matches your project:

| Upload form | Files included |
| --- | --- |
| `scene upload ./shot.blend` | Only the `.blend` file |
| `scene upload ./shot.blend --include textures --include cache` | The `.blend` file and the selected files or directories |
| `scene upload ./project --blend scenes/shot.blend` | Every regular file under the project root, except ignored files |

Uploads preserve project-relative paths. For a direct `.blend` upload,
`--include` paths are relative to the `.blend` file's directory and must stay
within it. Use a project-root upload when assets live in sibling directories.
The GPU worker checks external Blender assets when it opens the scene.

Files are hashed with SHA-256, so unchanged file blobs are reused. Uploading the
same scene again skips the upload; changing its files creates a new scene ID.

Automatic directory scanning ignores OS metadata (`.DS_Store`, `Thumbs.db`,
`desktop.ini`), editor backups (`*~`, `*.swp`), Blender backup saves (`*.blend1`,
…), Python caches (`__pycache__`, `*.pyc`), and version control internals (`.git`,
`.hg`, `.svn`). Explicitly included files are kept even if they match these rules.

### Selecting frames and render settings

`--frames` accepts single frames, inclusive ranges, stepped ranges, and
comma-separated combinations:

| Selection | Frames |
| --- | --- |
| `10` | Frame 10 |
| `1:120` | Every frame from 1 through 120 |
| `1:7:3` | Frames 1, 4, and 7 |
| `1:7:3,2` | Frames 1, 2, 4, and 7 |

The same syntax works with `result download --frames` and
`result remove --frames`.

The bundled image uses Blender 5.2.1 with the FLIP Fluids Demo integration.
Rendering uses Cycles and produces PNGs. Scene samples, tile size, and resolution
are preserved unless you override them:

| Option | Default or behavior |
| --- | --- |
| `--gpu` | `L4` |
| `--instances` | Maximum parallel workers; defaults to `1` |
| `--gpus-per-instance` | GPUs per worker; defaults to `1` |
| `--backend` | `OPTIX` (default) or `CUDA` |
| `--samples` | Override the scene's render samples |
| `--tile-size` | Override the render tile size |
| `--resolution-x`, `--resolution-y` | Override width and height; supply both together |
| `--resolution-percentage` | Scale the render resolution from `1` to `100` percent |

For example, render a smaller preview:

~~~sh
blender-modal scene render SCENE_ID --frames 1 --samples 32 --resolution-percentage 50
~~~

Completed frames are cached by scene, render settings, and renderer fingerprint.
Repeating a render with the same configuration renders only missing frames;
if all requested frames are complete, no job is submitted. Changing render
settings creates a separate result set. Changing the GPU type or worker count
does not change the result set.

### Running and monitoring jobs

Add `--detach` to return after submission while the render continues on Modal:

~~~sh
blender-modal scene render SCENE_ID --frames 1:120 --instances 4 --detach
blender-modal job list
blender-modal job info JOB_ID
blender-modal job info JOB_ID --watch
~~~

`job info` shows job state, worker progress, and per-job billing. `--watch`
refreshes the status until the job finishes. To stop an active render:

~~~sh
blender-modal job cancel JOB_ID
~~~

For workspace billing rates and a summary, use `blender-modal billing`.

### Reading worker logs

~~~sh
blender-modal job logs JOB_ID
blender-modal job logs JOB_ID --tail 1000
blender-modal job logs JOB_ID --follow
~~~

By default, `job logs` shows the latest 100 Modal log entries and exits.
`--tail` accepts 1–20,000 entries. `-f/--follow` streams until the Modal app stops
or you press Ctrl-C; stopping the log stream does not cancel the render.
`--tail` and `--follow` cannot be combined.

Logs include timestamps, container IDs, and available Modal runtime diagnostics
across all workers. Blender output is prefixed with its zero-based `[worker N]`
index. Logs are text only; this command rejects `--json`.

Log availability depends on Modal's retention and what was captured when the
job ran. Older jobs may lack full Blender output, and jobs without a stored
Modal app ID cannot retrieve logs.

### Finding and downloading results

~~~sh
blender-modal scene list
blender-modal result list --scene SCENE_ID
blender-modal result download RESULTS_ID --output ./renders
blender-modal result download RESULTS_ID --frames 1:10 --output ./preview
~~~

Downloads include all completed frames by default. Use `--frames` to select
specific completed frames. Local files with matching checksums are skipped;
if an existing file differs, the command fails unless you pass `--overwrite`.
Downloaded files are checked against their stored checksums.

### Removing scenes, results, and unused files

Preview deletions with `--dry-run`, then omit it to apply them:

~~~sh
blender-modal scene remove SCENE_ID --dry-run
blender-modal result remove RESULTS_ID --frames 1:10 --dry-run
blender-modal result remove RESULTS_ID --dry-run
blender-modal cleanup --dry-run
~~~

`scene remove` removes the uploaded scene record. `result remove` deletes a
whole result set, or only the frames selected with `--frames`.

`cleanup` removes abandoned staging files and upload blobs no longer referenced
by any scene. It retains files less than 24 hours old to avoid interfering with
active uploads. Use `cleanup --force` to remove those fresh files too, such as
after an interrupted upload.

### Global options and scripting

| Option | Purpose |
| --- | --- |
| `--volume NAME` | Select a Modal Volume; defaults to `blender-modal-v2` |
| `--environment NAME` | Select a Modal environment |
| `--json` | Write machine-readable output to standard output |
| `-v`, `--verbose` | Write detailed progress to standard error |

Normal output is human-readable, with colors and a status line during long
operations. Verbose output adds details such as hashing, file transfers, and
scene materialization. You can combine `--json` and `--verbose`.

Global options work before the resource, between the resource and action, or
after the action. Later explicit values take precedence. These are equivalent:

~~~sh
blender-modal --json scene list
blender-modal scene --json list
blender-modal scene list --json
~~~

Use the same `--volume` and `--environment` values when uploading, rendering,
and inspecting or downloading the associated data. For `job logs`, these
options select the job catalog and Modal environment used to retrieve logs.

JSON output keeps full scene and result IDs. Scene and result listings emit
one JSON object per line; `job list` returns `{"jobs": [...]}`, and `billing`
returns `{"billing": {...}}`.

## Command reference

All commands below follow `blender-modal`. Add `--help` at any level
to see available commands and options, for example
`blender-modal scene render --help`.

| Command | Purpose |
| --- | --- |
| `scene upload [ROOT] [options]` | Upload an immutable scene and its assets |
| `scene list` | List uploaded scenes |
| `scene render SCENE_ID --frames FRAMES [options]` | Render missing frames |
| `scene remove SCENE_ID [--dry-run]` | Remove an uploaded scene |
| `job list` | List stored render jobs |
| `job info JOB_ID [--watch]` | Inspect job state, workers, and per-job billing |
| `job logs JOB_ID [--tail N \| --follow]` | Show recent worker logs or stream live output |
| `job cancel JOB_ID` | Cancel an active render job |
| `result list [--scene SCENE_ID]` | List result sets, optionally filtered by scene |
| `result download RESULTS_ID --output DIRECTORY [options]` | Download completed PNG frames |
| `result remove RESULTS_ID [--frames FRAMES] [--dry-run]` | Remove a result set or selected frames |
| `cleanup [--dry-run] [--force]` | Remove abandoned staging and unreferenced blobs |
| `billing` | Show workspace billing rates and summary |

## Upgrading from older command names

The previous command paths have been replaced:

| Previous command | Current command |
| --- | --- |
| `upload` | `scene upload` |
| `list results` | `result list` |
| `info JOB_ID` | `job info JOB_ID` |
| `info` (without an ID) | `job list` and `billing` |

Apart from the `job list` and `billing` JSON shapes described above, command
payloads retain their existing fields. Full IDs remain in storage, so existing
scenes and cached renders keep working.

## Development

For an editable development environment, install all dependency groups and run
the local checks. Use `uv run blender-modal` to run the checkout version:

~~~sh
uv sync --all-groups
uv run pytest -q
uv run ruff check blender_modal tests renderer
uv run mypy blender_modal
~~~

## License

The original code in this repository is licensed under the [MIT License](LICENSE).
Third-party code, including FLIP Fluids in `third_party/flip-fluids`, remains under
its respective licenses.
