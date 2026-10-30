# Afterimage

**Film yourself painting. Afterimage takes the painter out of the footage, and the painting stays.**

Every pixel in the output is derived only from camera frames: aligned, colour-matched and
averaged over a few frames. No generative model touches the image. Nothing is invented.

Built for the Meta Global AI Developer Hackathon (Oct 26 – Nov 4, 2026), Indie track, theme
*Build for Your World*: I paint, and every process video I film is mostly my own elbow.
Built with **SAM 3.1** and **Muse Spark**, either through the Meta Model API or from the open
weights: SAM 3.1 on a Modal GPU and Muse Spark Open through Baseten, which is how every
session here was processed.

## What it does

- **A hands-free timelapse.** The painter, their hands and tools are removed and each stroke
  appears the moment the camera first saw it, dated to that moment.
- **Cameo.** You sit beside the canvas with a cup of tea while it fills in. The timelapse is
  projected into a real clip of the room (a labelled composite).
- **Pixel receipt.** Tap any spot on the finished painting and the camera footage of the last
  time that spot changed plays at natural speed, sound on, with the spot marked in the frame.
  The hand that painted it is right there.
- **Underneath.** Drag to peel the painting back one layer of changes at a time, down to the
  bare canvas. Conservators find this under old paintings with X-rays and call it pentimento;
  here it falls out of the record.
- **Ghost slider.** Fade the painter back in from 0% to 100%.
- **A checkable record.** `receipt.json` holds the footage hash, the hashes of every cached
  model response, the parameters, and a hash chain over every output frame.
  `scripts/afterimage_verify.py` re-derives the whole output from the footage and the caches
  and compares. Checkable, not proof: it shows the output came from this footage through
  this code, nothing more.

## How it uses Meta's models

| Model | Where | Job |
|---|---|---|
| **SAM 3.1** (`sam-3.1`) | [`afterimage/sam.py`](afterimage/sam.py) | Tracks every concept Spark asks for (`person`, `hand`, `paintbrush`, plus what Spark adds) through the whole session at 1–2 fps, one streaming call per concept, event streams cached and hashed. Follows the critic's boxes through the video as box prompts. |
| **Muse Spark 1.3** (`muse-spark-1.3`), scan | [`afterimage/spark.py`](afterimage/spark.py), [`afterimage/pipeline.py`](afterimage/pipeline.py) `spark_scan` | Watches a 48 s digest of the session and decides what else SAM should track (a rag, a cup, a cat), with a strict kind enum, and finds the canvas corners. Each extra concept is a SAM pass over the whole session, so Spark chooses which ones are worth it. |
| **Muse Spark 1.3**, critic | [`afterimage/critic.py`](afterimage/critic.py), `pipeline.critic_rounds` | After the clean plate has run, Spark sees (camera \| reconstruction) pairs at the moments most likely to have gone wrong and boxes anything that isn't paint. Objects go back to SAM as box prompts, shown at the frame the box's content arrived (the change timeline knows), when the brush or hand that left it was still in view; shadows, glare and smears are held, which drops the change that put them there. At most two rounds, and a round that changes nothing ends the loop; every decision cached and recorded, so verify replays it without asking. |
| **Open-weight mode** | [`scripts/sam_modal.py`](scripts/sam_modal.py), [`afterimage/sam_open.py`](afterimage/sam_open.py), `SPARK_PROVIDER=baseten` | The same pipeline with the weights instead of the API: `facebook/sam3.1` runs on a Modal GPU (text prompts for the concepts; the critic's boxes go to the SAM 2-style tracker SAM 3 ships, since SAM 3's own box prompt means "more things like this" rather than "this thing"; masks returned as a hashed file that names the tracker), and Muse Spark Open answers through Baseten's endpoint, with the schema enforced as a forced tool call and the digest sent as frames. The receipt names which path was used. |
| **Muse Image** | not used | Would contradict the promise. |

Every call goes through [`afterimage/ledger.py`](afterimage/ledger.py): model, endpoint, purpose,
prompt, response id, usage, time and estimated cost, appended to the run's `ledger.jsonl`.
Media is logged by SHA-256 only; the API key is scrubbed; the ledger's hash is in the receipt.

## How it works

```
footage ─┬─► proxy (1–2 fps) ─► Spark scan: what to track, where the canvas is
         │                        │
         │                        ▼
         │                      SAM 3.1: one mask stream per concept ──► painter masks
         │                                                                  │
         └─► canvas view (corners snapped to the canvas edges, ffmpeg warp) ─┤
                                                                            ▼
                      clean plate: a pixel changes only after its new value holds
                      for the persistence (8 s) while uncovered; every change is
                      recorded (pixel, first seen, before, after)
                                                                            │
                      transient veto: a change undone within 5 min was a shadow,  ◄── Spark critic
                      a glint or a resting hand, not paint; both changes dropped        boxes → SAM /
                                                                            │           holds
                                                                            ▼
                      replay ──► timelapse · compare · reels · receipt page · receipt.json
```

The hard cases, and what handles them:

- **A resting hand.** Masked by SAM; the persistence rule and the veto cover a mask that
  slips, and the critic covers what both miss.
- **A painted hand, or a portrait.** SAM finds painted hands too. Real ones reach in past the
  canvas edge, move, and belong to a body; painted ones sit inside the canvas, stay put and
  outlast any pose ([`afterimage/occluders.py`](afterimage/occluders.py)). Only the real ones are removed.
- **Shadows, glare, exposure drift.** Each frame is exposure-matched against a fixed
  reference (the bare canvas, or paint that has stood for five minutes), never against the
  drifting plate itself, and the 80th percentile of the per-pixel ratios is taken so that a
  shadow over part of the canvas doesn't read as a darker exposure. A shadow that flickers
  never commits; one that holds still commits and is vetoed when it lifts. A shadow that holds
  still *and never lifts* is the known limitation (below).
- **Where the canvas is.** Spark's corners land within a few percent of the frame, which is an
  easel clamp across the top of every output. [`afterimage/corners.py`](afterimage/corners.py) moves
  each edge onto the strongest bright-to-dark step nearby with a robust line fit, on the
  clearest of the first frames, and keeps Spark's quad if no clean edge is found.
- **The first frame.** Each pixel starts from its first uncovered value that holds steady, so
  a painter in frame 0 doesn't linger, and a stroke made in the first seconds still arrives
  as new paint.

## Evaluation

Everything below was measured on synthetic scenes that were written alongside the method.
They show the method does what it was designed to do; they are not a stand-in for footage.
Real-session numbers land here as soon as sessions have been processed.

Three scenes, three methods ([`eval/synthetic.md`](eval/synthetic.md), `scripts/afterimage_bench.py`):

| Scene | Afterimage | Masked median (Loomis-style) | Temporal median (no masks) |
|---|---|---|---|
| Moving hand + shadow: frames with a visible leftover | **0.0%** | 47.8% | 92.2% |
| Resting hand + still shadow: frames with a visible leftover | **0.0%** | 55.6% | 93.3% |
| Painted hand: frames with a visible leftover | **0.0%** | 27.8% | 83.3% |

The step-back protocol ([`afterimage/heldout.py`](afterimage/heldout.py), `scripts/afterimage_eval.py`)
hides the frames where nobody is over the canvas from every method, rebuilds, and scores
against what the camera saw there. A test shows hidden means unseen: a stroke first uncovered
inside a hidden window is missed (5.5% of pixels off) and caught when the method is allowed
to peek (0.3%). On the synthetic session: Afterimage ΔE00 0.27, 0.35% of pixels clearly off,
PSNR 39.6; masked median 0.74 / 1.14% / 28.4; temporal median 1.15 / 2.08% / 25.4.

The critic is measured with `scripts/critic_probe.py`: hand crops cut from the footage with
the SAM masks are planted into reconstructions, and Spark's recall and false-flag rate are
reported. The loop's plumbing is covered by tests with SAM and Spark played by mocks.

## Run it

```
python -m venv .venv && .venv\Scripts\pip install -r requirements.txt   # ffmpeg on PATH
copy .env.example .env                                                   # MODEL_API_KEY=...
python scripts/afterimage_run.py data/session.mp4 --corners 0.12,0.08 0.88,0.07 0.9,0.95 0.1,0.96
python scripts/afterimage_verify.py out/session/receipt.json --video data/session.mp4
python scripts/afterimage_eval.py data/session.mp4 --static 300-360
python scripts/cameo_run.py data/sitting.mp4 out/session/timelapse.mp4 --corners ...
```

`--no-scan` skips the Spark scan; `--corners` given by hand always win and are never snapped;
`--start 3` skips the seconds while the phone was being set down; `--out <folder>` puts the
outputs and caches somewhere other than `out/<session>`. The results page is
`out/<session>/index.html`; it runs from disk. Next to it: `receipt.json`, `stats.json`,
`frames.json` (per frame: share covered, mask stillness, disagreement with the plate, exposure
gain) and the request ledger. The core is numpy plus ffmpeg pipes (no OpenCV), so it runs on
Windows N editions too. Tests: `python -m unittest discover -s tests -t .`

Open weights instead of the API (needs Hugging Face access to `facebook/sam3.1`, a Modal
account and a Baseten key):

```
modal setup
modal secret create huggingface HF_TOKEN=hf_...
modal deploy scripts/sam_modal.py
modal run scripts/sam_modal.py --video data/session.mp4 --concepts hand,person --seconds 30   # overlay check
python scripts/afterimage_run.py data/session.mp4 --masks open --corners ...              # SPARK_PROVIDER=baseten in .env
```

## Limitations

- A fixed camera is required. Handheld footage isn't supported.
- A shadow that holds still for longer than the persistence and never lifts is kept as paint
  until the critic catches it. Side lighting avoids it; the capture notes say how.
- The camera's colour is not the pigment's colour. Steady light; a grey card helps. Lock the
  phone's exposure and white balance: the pipeline undoes auto-exposure drift, but what it can't
  see under the painter it can't correct.
- A lamp reflected off wet paint is in the footage, so it is in the output until the paint
  dries. Put the lamp high and to the side, where the phone screen shows no hot spot.
- Thin brushes under motion blur can slip past the masks; persistence and the critic are the net.
- In open-weight mode the critic's box is followed within its 30-minute stretch of the session
  (the tracker works in chunks); a leftover that outlasts the stretch is caught again by the
  next round, not carried over.
- Works for flat work under a fixed camera: paper, canvas, boards.

## Prior work

Removing the painter statistically goes back to Clady (SIGGRAPH 2006 poster) and Tan et al.
(SIGGRAPH 2015), who handled the hand and its shadow with keyframed temporal statistics for
stroke-layer decomposition. Loomis Painter (2025) uses a masked median per 10-second segment
to build a training set and reports that it cannot handle hand shadows. Whiteboard cameras
(MSR 2004, Kaptivo, Microsoft Teams, Logitech Scribe) remove the presenter from a whiteboard
with real pixels. Repentir (Manchester Metropolitan University) recognises a finished painting
in a photo and overlays photos taken while it was being made, which is the pixel receipt's
question asked the other way round. Generative erasers (Runway, Kling, Veo) regenerate the
scene and invent strokes. No shipped tool does this for physical painting video. What's new here is the
combination: open-vocabulary SAM 3.1 masks chosen by Spark, a clean plate that dates every
pixel and can revoke changes, a critic loop that re-prompts SAM, a held-out evaluation against
the standard baselines, and a receipt you can re-derive.

## Built during the hackathon

All code was written from Oct 26, 2026. The first nine commits were made together on
Oct 27 from a day of local work; everything after is committed as it was built, and the
repository went public on Oct 30 with that local history intact. Solo entry by Mishal
(Mishallic). Model responses are cached under `out/` and are not committed.

Built with Muse Spark Open.
