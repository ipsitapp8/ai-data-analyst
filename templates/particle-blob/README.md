# Particle blob template

One file, no libraries, no internet needed: `index.html`. Thousands of drifting points gather into rings and then
into a glowing sphere, dissolve, and repeat (18 s loop). The mouse tilts the blob and pushes nearby points aside.

## Use it
- Open `index.html` in a browser, or
- Put it in any project as a background: `<iframe src="index.html" style="position:fixed;inset:0;width:100%;height:100%;border:0"></iframe>`,
  or copy the `<canvas>` and the `<script>` into your own page.

## Change it (top of the script, the `CONFIG` block)
| Setting | What it does |
|---|---|
| `size` | blob radius as a fraction of the smaller window side (`.30` default) |
| `colors` | the three point colours as `[r, g, b]` |
| `captions` | `false` hides the three labels at the bottom |
| `captionText` | the label wording |

Background colour: `background:#0b0a1a` in the `<style>` line. Loop length: `T = 18` (seconds; the stage timings are fixed fractions of it).

Respects the OS "reduce motion" setting by drawing one still frame of the finished blob.
Extracted from Silt's Home page scene (`frontend/hero_scene.py`).
