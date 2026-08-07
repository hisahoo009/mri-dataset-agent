---
title: MRI Lesion Dataset Finder
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.22.0
python_version: '3.13'
app_file: app.py
pinned: false
license: mit
short_description: Finds open MRI lesion datasets and checks image formats
---

# MRI Lesion Dataset Agent

A multi-step **smolagents** agent that finds open datasets for a given type of MRI
lesion and verifies the datasets actually contain images in a usable format
(`.jpg`/`.png`/`.tif` and friends, or NIfTI/DICOM if you allow it).

Built for the Hugging Face Agents course: pure smolagents + tool calls, no other agent
framework.

## Install

```bash
pip install -r requirements-dev.txt
export HF_TOKEN=hf_...        # for the inference model, not for dataset search
```

## Run

```bash
python smoke_test.py                              # tools only, no LLM — run this first
python app.py                                     # the Gradio UI, locally
python app.py --cli "open MS white matter lesion MRI datasets"
python app.py --cli "glioma MRI" --policy photographic_or_volumetric
python app.py --list-lesions
pytest tests -q                                   # 40 offline tests
```

`app.py` is both the Space entry point and the CLI. The `gradio` import is lazy, so
`--cli` works in an environment where Gradio isn't installed.

Or from Python:

```python
from lesion_finder import build_agent
agent = build_agent()                            # CodeAgent
agent = build_agent(agent_type="tool_calling")   # no code execution
print(agent.run("Find open datasets of multiple sclerosis lesions with .jpg images"))
```

## Layout

Eight Python files. Each one is a seam you might actually want to edit independently.

```
├── app.py               Gradio UI + CLI — the HF Space entry point
├── smoke_test.py        live check of the source adapters, no LLM
├── tests/
│   └── test_lesion_finder.py    40 offline tests
└── lesion_finder/
    ├── ontology.py      controlled vocabulary of MRI lesion types + synonyms
    ├── schemas.py       pydantic In/Out model for every tool; image-format policy
    ├── tools.py         ValidatedTool guardrail + the four pipeline tools
    ├── sources.py       HTTP, format classification, one adapter per repository
    └── agent.py         agent assembly + pipeline instructions
```

`sources.py` is sectioned: HTTP plumbing, then extension classification, then the
three adapters, then the registry. `tools.py` holds `ValidatedTool` alongside the
tools that subclass it — it has no other consumers, so a separate module bought
nothing.

## The four steps

| # | Tool | Does |
|---|------|------|
| 1 | `normalize_lesion_query` | free text → validated `lesion_key` + search terms |
| 2 | `search_open_datasets` | queries the repos → candidates (contents **unverified**) |
| 3 | `inspect_dataset_files` | lists one dataset's files → what formats are really in it |
| 4 | `shortlist_image_datasets` | applies the image policy → ranked shortlist + rejections |

They are separate on purpose. Step 2 returns names that merely *match*; only step 3
proves a dataset has images. The agent has to carry state between calls and decide
which candidates justify the cost of a file listing — that's the multi-step part.

## Guardrails

**Input validation** — every call goes through a pydantic model before the tool body runs.

- `normalize_lesion_query` rejects anything that doesn't map to a known lesion type,
  and refuses queries phrased as personal medical advice ("does my scan show a tumour").
- `search_open_datasets` accepts only a `lesion_key` from step 1, never a raw user
  phrase — so a prompt-injected search string can't reach the backends.
- `sources` is a `Literal` allowlist; `max_results_per_source` is 1–25;
  `max_files_scanned` is 10–5000; `top_k` is 1–20.
- `extra="forbid"` — unexpected keys are an error, not silently dropped.
- Failures raise `ToolInputError` with a message naming the valid options, so the agent
  can correct itself instead of the run dying.

**Output validation** — every return value is validated against its `*Out` model before
the agent sees it. A malformed upstream response becomes a clean `ToolOutputError`, not
a hallucination-friendly half-payload. Unknown keys from the API are dropped rather than
leaking into the agent's context.

**Image-format check** (`schemas.py`) — extensions are classified into:

- `photographic` — `.jpg .jpeg .png .bmp .tif .tiff .webp .gif` — load directly with PIL
- `volumetric` — `.nii .nii.gz .dcm .mha .nrrd` — need nibabel/pydicom
- `archive` — `.zip .tar.gz .parquet .h5` — images may be hidden inside
- `other`

`image_policy="photographic_only"` (the default) keeps only the first group. If a
dataset shows only archives, it's flagged `images_possibly_in_archives` rather than
accepted or silently dropped — the file listing genuinely can't tell.

**Agent-level** — `max_steps=12` (the pipeline needs ~8, leaving slack for one or two
self-corrections), and for the `CodeAgent`, `additional_authorized_imports` is limited to
`json`, `re`, `statistics`, so generated code can parse tool output but can't fetch
anything the tools didn't sanction.

## Deploying to a Hugging Face Space

The repo is Space-ready: `app.py` is the entry point and this README already carries the
YAML header Spaces needs.

**Option A — one-shot upload (no git).** Simplest if you already created the Space:

```bash
pip install -U huggingface_hub        # the CLI is now `hf`, not `huggingface-cli`
hf auth login
hf upload <your-username>/<space-name> . . --repo-type=space
```

**Option B — git remotes.** Use this if you also want the code on GitHub:

```bash
git remote add origin https://github.com/<you>/mri-lesion-dataset-finder.git
git remote add space  https://huggingface.co/spaces/<you>/<space-name>

git push -u origin main
git pull space main --allow-unrelated-histories   # the Space already has a README
git push space main
```

That `--allow-unrelated-histories` pull is the step people miss: a freshly created Space
is not actually empty — HF commits a `README.md` and `.gitattributes` for you, so the
first push is rejected as unrelated. Pull once, keep *your* README when resolving the
conflict (it carries the YAML header), then push.

Authentication for `git push` to HF uses an access token with **write** scope as the
password, not your account password. `hf auth login` stores one for you.

Then in **Settings → Variables and secrets**:

| Key | Required | Notes |
|---|---|---|
| `HF_TOKEN` | yes | secret — your Inference Providers token |
| `MRI_AGENT_TYPE` | no | `tool_calling` (default) or `code` |
| `MRI_AGENT_MODEL` | no | defaults to `Qwen/Qwen2.5-Coder-32B-Instruct` |
| `MRI_TOOL_CHOICE` | no | `omit` (default), or `auto` / `none` / `required` — see below |

Free **CPU Basic** (2 vCPU, 16 GB) is enough — no model runs locally, the agent only
makes API calls. Spaces have outbound internet, so all three dataset backends are
reachable.

Two things to know before you make it public:

- **The Space spends *your* inference credits.** It runs on your `HF_TOKEN`, so every
  visitor's query bills your account. Fine for a course submission; add HF OAuth sign-in
  if you share it widely.
- **`app.py` defaults to `agent_type="tool_calling"` on purpose.** A `CodeAgent` executes
  model-written Python inside the Space container, and a public text box is a
  prompt-injection surface. `ToolCallingAgent` emits structured tool calls instead —
  no code execution at all — and since all four tools take simple typed arguments,
  nothing is lost. Set `MRI_AGENT_TYPE=code` only if you understand that trade-off.

Free Spaces sleep after ~48h idle, so the first request after a quiet spell pays a cold
start on top of the usual 30–60s run.

## Notes and caveats

- **`tool_choice` errors.** smolagents defaults to `tool_choice="required"`, which
  Hugging Face Inference Providers reject in two different ways: `400
  INVALID_TOOL_CHOICE` (only `auto`/`none` accepted) or `422
  UNSUPPORTED_OPENAI_PARAMS` (the parameter isn't supported at all). Compounding it,
  a `tool_choice` set as a *model* kwarg is attached to **every** request, including the
  planning step, which sends no tools — and a provider that tolerates it alongside
  `tools` may still reject it on its own. `agent.py` therefore omits the parameter by
  default and lets the provider apply its own behaviour. Set `MRI_TOOL_CHOICE` to
  `auto`, `none` or `required` to send it explicitly.
- **Run `smoke_test.py` first.** The adapters were written against the documented API
  shapes but I couldn't reach the network to verify them live; the smoke test calls each
  backend directly and prints what came back, so any parsing mismatch shows up
  immediately and locally.
- Zenodo's public search works without credentials; set `ZENODO_TOKEN` if you hit rate
  limits.
- OpenNeuro is NIfTI-first — it will almost never satisfy `photographic_only`. That's
  correct behaviour, not a bug.
- The ontology has 13 lesion families. Add one by appending a `LesionType` to
  `LESION_ONTOLOGY` — no other file needs to change.
- Licences are reported as declared by the repository. Always verify the licence and
  de-identification status yourself before using medical imaging data.

## Extending it

- **New source**: implement `search()` / `inspect()` per the `DatasetSource` protocol in
  `sources.py`, add it to `REGISTRY` and to the `SourceName` literal in `schemas.py`.
- **New guardrail**: add a `field_validator` to the relevant `*In` model — no tool code
  changes.
- **Deeper verification**: add a step-5 tool that downloads one sample image and confirms
  it opens with PIL and looks like an MRI slice (greyscale, plausible dimensions).
