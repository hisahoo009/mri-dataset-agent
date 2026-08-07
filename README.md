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

# MRI Lesion Dataset Finder

A small **smolagents** agent that finds open MRI datasets for a lesion type and
checks whether they actually contain usable images.

Built for the Hugging Face Agents course. Four files, no framework beyond smolagents.

## The idea

Searching for "MS lesion dataset" gives you names that *match*. It doesn't tell you
whether there are any actual images inside — plenty of hits are papers, metadata, or
NIfTI volumes you'd need extra libraries to open.

So the agent works in steps, and each step is a tool:

```
normalize_lesion_query   "MS lesion datasets"  ->  lesion_key: "ms_lesion"
search_datasets          lesion_key            ->  candidate dataset ids
inspect_dataset          one dataset id        ->  how many .jpg/.jpeg/.png files
```

Only `.jpg`, `.jpeg` and `.png` count — images you can open with PIL and nothing
else. Anything else reads as zero images and gets skipped.

The agent decides which candidates are worth inspecting, then writes the report.
That decision-making between tool calls is what makes it an *agent* rather than a
script.

## Files

```
app.py       Gradio UI + a --cli flag
agent.py     builds the CodeAgent
tools.py     the three tools
lesions.py   the lesion vocabulary
```

## Run it

```bash
pip install -r requirements.txt
export HF_TOKEN=hf_...

python app.py                                       # the UI
python app.py --cli "open MS lesion datasets"       # one query, printed
```

## Guardrails

The agent takes free text from a stranger and calls APIs with it, so each tool
checks its own inputs before doing any work.

**`normalize_lesion_query` is the gate.** It maps free text onto a known lesion key
and raises on anything else — including requests for medical advice:

```python
normalize_lesion_query("datasets of cat photos")
# ValueError: No supported lesion type found...

normalize_lesion_query("does my mri show a tumor")
# ValueError: Refused: this reads as a request for medical advice...
```

**`search_datasets` only accepts a `lesion_key`**, never raw user text. By the time
anything reaches the Hugging Face API it is one of eight known strings, so a
prompt-injected search phrase has nowhere to go.

**`inspect_dataset` validates its own output** against the `DatasetReport` pydantic
model before returning, and it is the only source of truth about whether a dataset
is usable. A malformed response from the Hub becomes a clear error rather than a
half-filled dict the model might hallucinate around.

Errors are raised as plain `ValueError` with a message naming the valid options.
smolagents feeds that back to the model, which can correct itself and retry.

The agent is also capped at `max_steps=12`, and generated code may only import
`json`.

## How the agent calls tools

`CodeAgent` puts each tool's signature and docstring into the system prompt, and the
model calls them by writing Python:

```python
result = normalize_lesion_query(query="MS lesion datasets")
candidates = search_datasets(lesion_key=result["lesion_key"])
```

This is why the docstrings matter — they are the tool's interface, not decoration.

It also means no OpenAI `tools` API parameter is ever sent, so this runs on any chat
model. A `ToolCallingAgent` would send tools as a structured API field, which many
Hugging Face Inference Provider routes reject with
`422 UNSUPPORTED_OPENAI_PARAMS`.

## Deploying to a Space

`app.py` is the entry point and this README has the YAML header Spaces needs.

```bash
git push space main
```

Then set `HF_TOKEN` in **Settings → Variables and secrets**. Free CPU Basic is
enough — no model runs locally, the agent only makes API calls.

## Notes

- Licences are reported as the Hub declares them. Check the licence and
  de-identification status yourself before using medical imaging data.
- Adding a lesion type is one entry in `LESIONS` in `lesions.py`.
- Accepting more formats is one entry in `IMAGE_FORMATS` in `tools.py`.
- Adding a data source means one more `@tool` function in `tools.py`.
- Datasets that ship images inside a `.zip` or `.parquet` will read as zero images.
  That's deliberate: a file listing can't see inside an archive, so guessing would
  be worse than skipping.
