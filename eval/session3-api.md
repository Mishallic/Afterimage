# Held-out evaluation: VID_20261102_133026.mp4 (session3-api)

32 stretches where the painter stepped back, 3686 frames at 1 fps. Each stretch was hidden from every method, rebuilt from the rest of the session and compared with what the camera saw. Lower ΔE00 is better; higher PSNR is better. "Clearly off" counts pixels with ΔE00 above 5.

| Method | ΔE00 | Clearly off % | PSNR (dB) |
|---|---|---|---|
| Afterimage | 1.86 | 3.56 | 32.01 |
| masked median (Loomis-style) | 1.81 | 2.29 | 33.45 |
| temporal median (no masks) | 3.41 | 10.85 | 25.41 |

| Stretch (s) | Frames | Spread | Afterimage ΔE00 | masked median (Loomis-style) ΔE00 | temporal median (no masks) ΔE00 |
|---|---|---|---|---|---|
| 159.0–165.0 | 6 | 4.26 | 1.96 | 1.52 | 2.73 |
| 245.0–248.0 | 3 | 4.19 | 1.73 | 1.3 | 5.4 |
| 277.0–284.0 | 7 | 4.53 | 1.5 | 1.45 | 4.85 |
| 290.0–295.0 | 5 | 4.78 | 1.65 | 1.71 | 1.51 |
| 349.0–352.0 | 3 | 4.51 | 1.58 | 1.68 | 3.98 |
| 369.0–374.0 | 5 | 4.84 | 1.53 | 1.52 | 4.89 |
| 383.0–390.0 | 7 | 5.64 | 1.64 | 1.56 | 3.66 |
| 408.0–414.0 | 6 | 4.56 | 1.88 | 1.64 | 3.21 |
| 426.0–429.0 | 3 | 4.5 | 1.65 | 1.41 | 2.92 |
| 447.0–451.0 | 4 | 5.56 | 1.59 | 1.8 | 3.65 |
| 500.0–508.0 | 8 | 5.5 | 1.67 | 1.4 | 1.73 |
| 529.0–532.0 | 3 | 5.65 | 1.57 | 1.9 | 2.54 |
| 732.0–735.0 | 3 | 4.99 | 1.56 | 1.44 | 1.53 |
| 747.0–751.0 | 4 | 5.04 | 1.51 | 1.59 | 2.18 |
| 799.0–809.0 | 10 | 5.37 | 1.73 | 1.63 | 1.77 |
| 817.0–884.0 | 67 | 5.27 | 1.92 | 1.92 | 1.78 |
| 962.0–1021.0 | 59 | 5.89 | 2.03 | 2.02 | 3.81 |
| 1054.0–1058.0 | 4 | 5.15 | 1.63 | 1.53 | 2.59 |
| 1084.0–1125.0 | 41 | 4.82 | 1.95 | 2.27 | 3.23 |
| 1146.0–1151.0 | 5 | 5.68 | 1.7 | 1.62 | 2.18 |
| 1156.0–1170.0 | 14 | 6.0 | 1.79 | 1.62 | 1.55 |
| 1590.0–1594.0 | 4 | 5.91 | 1.83 | 2.01 | 2.33 |
| 1708.0–1756.0 | 48 | 5.77 | 1.98 | 2.23 | 7.41 |
| 1884.0–1891.0 | 7 | 5.76 | 2.75 | 2.29 | 5.01 |
| 2056.0–2064.0 | 8 | 5.72 | 2.33 | 1.79 | 5.84 |
| 2101.0–2179.0 | 78 | 5.88 | 2.43 | 1.96 | 4.15 |
| 2642.0–2712.0 | 70 | 5.68 | 1.75 | 2.22 | 6.49 |
| 2936.0–2941.0 | 5 | 5.98 | 1.92 | 2.12 | 2.3 |
| 3191.0–3244.0 | 53 | 5.97 | 2.0 | 2.14 | 5.39 |
| 3300.0–3319.0 | 19 | 5.53 | 1.98 | 2.73 | 3.33 |
| 3425.0–3431.0 | 6 | 6.0 | 1.47 | 1.68 | 1.99 |
| 3604.0–3686.0 | 82 | 5.8 | 3.26 | 2.23 | 3.24 |

Not scored: 558.0–703.0 s moved too much to be a still canvas (spread 14.51), so something was likely there that the masks missed.

Not scored: 914.0–941.0 s moved too much to be a still canvas (spread 10.13), so something was likely there that the masks missed.

Not scored: 1219.0–1395.0 s moved too much to be a still canvas (spread 7.23), so something was likely there that the masks missed.

Not scored: 1413.0–1428.0 s moved too much to be a still canvas (spread 6.07), so something was likely there that the masks missed.

Not scored: 1667.0–1671.0 s moved too much to be a still canvas (spread 6.4), so something was likely there that the masks missed.

Not scored: 1680.0–1692.0 s moved too much to be a still canvas (spread 6.03), so something was likely there that the masks missed.

Not scored: 1774.0–1801.0 s moved too much to be a still canvas (spread 6.41), so something was likely there that the masks missed.

Not scored: 1936.0–1939.0 s moved too much to be a still canvas (spread 6.24), so something was likely there that the masks missed.

Not scored: 1945.0–1948.0 s moved too much to be a still canvas (spread 6.8), so something was likely there that the masks missed.

Not scored: 1987.0–1991.0 s moved too much to be a still canvas (spread 6.04), so something was likely there that the masks missed.

Not scored: 1995.0–1998.0 s moved too much to be a still canvas (spread 6.18), so something was likely there that the masks missed.

Not scored: 2006.0–2010.0 s moved too much to be a still canvas (spread 6.02), so something was likely there that the masks missed.

Not scored: 2034.0–2037.0 s moved too much to be a still canvas (spread 6.33), so something was likely there that the masks missed.

Not scored: 2086.0–2090.0 s moved too much to be a still canvas (spread 6.1), so something was likely there that the masks missed.

Not scored: 2527.0–2584.0 s moved too much to be a still canvas (spread 6.57), so something was likely there that the masks missed.

Not scored: 2593.0–2605.0 s moved too much to be a still canvas (spread 6.71), so something was likely there that the masks missed.

Not scored: 2611.0–2636.0 s moved too much to be a still canvas (spread 6.19), so something was likely there that the masks missed.

Not scored: 2749.0–2808.0 s moved too much to be a still canvas (spread 6.05), so something was likely there that the masks missed.

Not scored: 2832.0–2835.0 s moved too much to be a still canvas (spread 6.3), so something was likely there that the masks missed.

Not scored: 2843.0–2920.0 s moved too much to be a still canvas (spread 6.38), so something was likely there that the masks missed.

Not scored: 2948.0–2952.0 s moved too much to be a still canvas (spread 6.06), so something was likely there that the masks missed.

Not scored: 3005.0–3008.0 s moved too much to be a still canvas (spread 6.02), so something was likely there that the masks missed.

Not scored: 3046.0–3051.0 s moved too much to be a still canvas (spread 7.95), so something was likely there that the masks missed.

Not scored: 3060.0–3080.0 s moved too much to be a still canvas (spread 6.0), so something was likely there that the masks missed.

Not scored: 3136.0–3147.0 s moved too much to be a still canvas (spread 6.32), so something was likely there that the masks missed.

Not scored: 3179.0–3183.0 s moved too much to be a still canvas (spread 6.21), so something was likely there that the masks missed.

Not scored: 3457.0–3467.0 s moved too much to be a still canvas (spread 6.15), so something was likely there that the masks missed.

Not scored: 3494.0–3508.0 s moved too much to be a still canvas (spread 6.4), so something was likely there that the masks missed.

Not scored: 3531.0–3534.0 s moved too much to be a still canvas (spread 6.02), so something was likely there that the masks missed.

Not scored: 3555.0–3600.0 s moved too much to be a still canvas (spread 6.1), so something was likely there that the masks missed.

Meta calls for this run: 11, about $3.69 (see the run's ledger.jsonl).
