# Critic probe: VID_20261031_152817.mp4 (session2-api)

20 planted leftovers across 40 images (20 clean), seed 1, 2 images per Spark call.
Reported: what Spark boxed. Acted on: what the pipeline would change, i.e. objects a second look at a crop of the reconstruction alone confirms, plus glare and shadows (smears are only recorded). Plants are hand and brush cut-outs from the footage, pasted into half the reconstructions.

| Stage | Found | Recall | False flags | Flags per image |
|---|---|---|---|---|
| reported | 6 / 20 | 30% | 28 | 0.70 |
| acted on | 4 / 20 | 20% | 5 | 0.12 |

Missed after the second look, at frames: 265, 387, 444, 769, 818, 942, 1024, 1256, 1308, 1411, 1670, 2337, 2538, 2560, 2817, 2931.
Spark calls: 107, about $0.00.
