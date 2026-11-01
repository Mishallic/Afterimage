# Critic probe: VID_20261029_174252.mp4 (session1-api-v3)

20 planted leftovers across 40 images (20 clean), seed 1, 2 images per Spark call.
Reported: what Spark boxed. Acted on: what the pipeline would change, i.e. objects a second look at a crop of the reconstruction alone confirms, plus glare and shadows (smears are only recorded). Plants are hand and brush cut-outs from the footage, pasted into half the reconstructions.

| Stage | Found | Recall | False flags | Flags per image |
|---|---|---|---|---|
| reported | 9 / 20 | 45% | 16 | 0.40 |
| acted on | 9 / 20 | 45% | 4 | 0.10 |

Missed after the second look, at frames: 137, 719, 1244, 1651, 2274, 2522, 3219, 3771, 4136, 4331, 4739.
Spark calls: 43, about $0.00.
