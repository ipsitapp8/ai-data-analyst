"""Home hero: a canvas particle scene that shows what Silt does in one 18 s loop.

noise  -> thousands of unlabelled points drifting in a turbulent flow field (raw data)
scan   -> a wave sweeps out; points it touches light up and fall into three orbit rings (evaluation)
sphere -> the rings tighten into one rotating, depth-shaded sphere with a core pulse (verified insight)
dissolve, repeat.

The scene is a self-contained HTML page for `st.iframe` (no network, no external assets, no data from
the app is passed in, so there is nothing to escape). The mouse tilts the sphere and pushes loose
points aside. `prefers-reduced-motion` shows a single still frame of the finished sphere.
"""

SCENE_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="color-scheme" content="dark"><style>
html,body{margin:0;height:100%;background:transparent;overflow:hidden;color-scheme:dark}
canvas{display:block;width:100%;height:100%}
</style></head><body><canvas id="c"></canvas><script>
(() => {
const cv = document.getElementById('c'), ctx = cv.getContext('2d');
const REDUCE = matchMedia('(prefers-reduced-motion: reduce)').matches;
const TAU = Math.PI * 2, T = 18, GOLD = Math.PI * (3 - Math.sqrt(5));
const PAL = [[168, 85, 247], [255, 196, 108], [255, 122, 184]];   // violet, amber, pink
const DIM = [122, 112, 196];
let W = 0, H = 0, DPR = 1, cx = 0, cy = 0, R = 0, N = 0, P = [], sprites = [], dimSprite;
let mx = .5, my = .5, tmx = .5, tmy = .5, mouseIn = false;

const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const ss = (a, b, x) => { x = clamp((x - a) / (b - a)); return x * x * (3 - 2 * x); };
const ease = x => x * x * (3 - 2 * x);
const rnd = (a, b) => a + Math.random() * (b - a);

function sprite(rgb) {
  const s = 64, c = document.createElement('canvas'); c.width = c.height = s;
  const g = c.getContext('2d'), gr = g.createRadialGradient(s/2, s/2, 0, s/2, s/2, s/2);
  gr.addColorStop(0, `rgba(255,255,255,1)`);
  gr.addColorStop(.18, `rgba(${rgb[0]},${rgb[1]},${rgb[2]},.95)`);
  gr.addColorStop(.5, `rgba(${rgb[0]},${rgb[1]},${rgb[2]},.22)`);
  gr.addColorStop(1, `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0)`);
  g.fillStyle = gr; g.fillRect(0, 0, s, s); return c;
}
sprites = PAL.map(sprite); dimSprite = sprite(DIM);

function build() {
  N = Math.round(clamp(W * H / 380, 1400, 3200));
  P = [];
  for (let i = 0; i < N; i++) P.push({
    x: rnd(0, W), y: rnd(0, H), vx: 0, vy: 0,
    g: i % 3, th: rnd(0, TAU), d: Math.random(), sw: Math.random() < .5 ? -1 : 1,
    sz: rnd(.7, 1.7), jit: rnd(-1, 1), rs: rnd(.7, 1.3)
  });
}
function resize() {
  DPR = Math.min(window.devicePixelRatio || 1, 2);
  W = cv.clientWidth; H = cv.clientHeight;
  cv.width = Math.round(W * DPR); cv.height = Math.round(H * DPR);
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  anchor();
  build(); fade(1);
}
// the canvas is transparent and fixed behind the whole page: old frames fade out instead of being painted over
function fade(a) {
  ctx.globalCompositeOperation = 'destination-out'; ctx.fillStyle = `rgba(0,0,0,${a})`; ctx.fillRect(0, 0, W, H);
  ctx.globalCompositeOperation = 'source-over';
}
// the sphere, captions and cards are anchored to the hero stage in the page (same-origin parent), so they scroll with it
let sx = 0, sy = 0, sw = 0, sh = 0;
function anchor() {
  let r = null;
  try { const el = window.parent.document.querySelector('[class*="st-key-home_stage"]'); if (el) r = el.getBoundingClientRect(); } catch (e) {}
  if (r && r.width > 50) { sx = r.left; sy = r.top; sw = r.width; sh = r.height; }
  else { sx = W * .55; sy = H * .14; sw = W * .38; sh = H * .6; }
  cx = sx + sw * .36; cy = sy + sh * .53; R = Math.min(sw, sh) * .23;
}

// divergence-free (curl) flow: particles keep an even density instead of piling into filaments
function flow(x, y, t, out) {
  const kx = .0105, ky = .0125, k1 = .0062, k2 = .0083;
  const A = x * kx + t * .3, B = y * ky - t * .2, C = (x + y) * k1 + t * .15, D = (x - y) * k2 - t * .1;
  const cA = Math.cos(A), sA = Math.sin(A), cB = Math.cos(B), sB = Math.sin(B);
  const cC = Math.cos(C), sC = Math.sin(C), cD = Math.cos(D), sD = Math.sin(D);
  const dx = kx * cA * cB + .6 * (k1 * cC * cD - k2 * sC * sD);
  const dy = -ky * sA * sB + .6 * (k1 * cC * cD + k2 * sC * sD);
  out[0] = dy * 3800; out[1] = -dx * 3800;
}
const FV = [0, 0];

const TILT = [-.55, .4, 1.25];
function ringPos(p, t) {
  const rr = R * (.62 + .2 * p.g) * (1 + p.jit * .035);
  const th = p.th + t * (.55 + p.g * .12) * (p.g === 1 ? -1 : 1) * p.rs;
  const ex = rr * Math.cos(th), ey = rr * .34 * Math.sin(th), a = TILT[p.g] + (my - .5) * .35;
  return [cx + ex * Math.cos(a) - ey * Math.sin(a), cy + ex * Math.sin(a) + ey * Math.cos(a), Math.sin(th)];
}
let rotY = 0, tiltA = .42;
function spherePos(p, i) {
  const y3 = 1 - 2 * (i + .5) / N, r = Math.sqrt(1 - y3 * y3), ph = i * GOLD + rotY;
  const x3 = Math.cos(ph) * r, z3 = Math.sin(ph) * r;
  const ca = Math.cos(tiltA), sa = Math.sin(tiltA);
  const yy = y3 * ca - z3 * sa, zz = y3 * sa + z3 * ca, per = 1 / (1 - zz * .2);
  return [cx + R * x3 * per, cy + R * yy * per, zz, y3];
}

function stroke(c, w) { ctx.strokeStyle = c; ctx.lineWidth = w; ctx.stroke(); }
function caption(t) {
  const items = [
    ['RAW DATA', 'unlabelled, messy, no meaning yet', 1 - ss(3.4, 4.4, t)],
    ['EVALUATING', 'profiling, testing, cross-checking', ss(4.2, 5.2, t) * (1 - ss(12.2, 13, t))],
    ['VERIFIED INSIGHT', 'one answer you can trust', ss(12.8, 14, t) * (1 - ss(16.6, 17.6, t))]];
  ctx.globalCompositeOperation = 'source-over'; ctx.textBaseline = 'alphabetic';
  for (const [h, s, a] of items) {
    if (a < .01) continue;
    ctx.globalAlpha = a; ctx.fillStyle = '#fff'; ctx.font = '600 12px system-ui,Segoe UI,sans-serif';
    if ('letterSpacing' in ctx) ctx.letterSpacing = '1.6px'; ctx.fillText(h, sx + 24, sy + 34);
    if ('letterSpacing' in ctx) ctx.letterSpacing = '0px';
    ctx.fillStyle = 'rgba(255,255,255,.55)'; ctx.font = '400 11.5px system-ui,Segoe UI,sans-serif'; ctx.fillText(s, sx + 24, sy + 52);
  }
  ctx.globalAlpha = 1;
}

let last = 0, clock = 0;
function frame(now) {
  const dt = Math.min(.05, (now - last) / 1000 || .016); last = now; clock += dt;
  if (!REDUCE) requestAnimationFrame(frame);          // schedule first: one bad frame must never stop the loop
  try { draw(REDUCE ? 15.2 : clock % T, dt); }
  catch (e) { ctx.globalAlpha = 1; ctx.globalCompositeOperation = 'source-over'; if (window.console) console.error('hero frame', e); }
}

function draw(t, dt) {
  tmx += (mx - tmx) * .06; tmy += (my - tmy) * .06;
  rotY += dt * .5 + (tmx - .5) * dt * 1.6; tiltA = .42 + (tmy - .5) * .6;

  // sphere weight, drives glow and wireframe
  const sw = t < 10 ? 0 : t < 14 ? ss(10, 14, t) : t < 16.5 ? 1 : 1 - ss(16.5, 18, t);
  anchor(); fade(REDUCE ? 1 : .3);

  // advance the loose points every frame so every transition starts from where they are
  for (const p of P) {
    flow(p.x, p.y, clock, FV); const sp = .7 + p.sz * .22;
    p.vx += (FV[0] * sp - p.vx) * .06; p.vy += (FV[1] * sp - p.vy) * .06;
    if (mouseIn) { const dx = p.x - tmx * W, dy = p.y - tmy * H, d2 = dx * dx + dy * dy;
      if (d2 < 9000) { const f = (1 - d2 / 9000) * 160; const d = Math.sqrt(d2) + .1; p.vx += dx / d * f * dt * 6; p.vy += dy / d * f * dt * 6; } }
    p.x += p.vx * dt; p.y += p.vy * dt;
    if (p.x < -10) p.x += W + 20; else if (p.x > W + 10) p.x -= W + 20;
    if (p.y < -10) p.y += H + 20; else if (p.y > H + 10) p.y -= H + 20;
  }

  ctx.globalCompositeOperation = 'lighter';
  // core glow under the sphere
  if (sw > .01) {
    const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, R * 1.5);
    g.addColorStop(0, `rgba(170,120,255,${.22 * sw})`); g.addColorStop(.45, `rgba(110,60,210,${.12 * sw})`); g.addColorStop(1, 'rgba(60,30,140,0)');
    ctx.fillStyle = g; ctx.fillRect(cx - R * 1.6, cy - R * 1.6, R * 3.2, R * 3.2);
  }
  // scan wave
  const waveOn = t > 4 && t < 8.6, waveR = ss(4, 8.2, t) * Math.hypot(W, H) * .62;
  if (waveOn) {
    ctx.beginPath(); ctx.arc(cx, cy, waveR, 0, TAU);
    stroke(`rgba(214,190,255,${.16 * (1 - ss(7.6, 8.6, t))})`, 1.2);
    ctx.beginPath(); ctx.arc(cx, cy, Math.max(0, waveR - 7), 0, TAU);
    stroke(`rgba(168,85,247,${.05 * (1 - ss(7.6, 8.6, t))})`, 5);
  }

  // particles
  const lp = (t0, t1, d) => ease(clamp((t - t0 - d * .35 * (t1 - t0)) / ((t1 - t0) * .65)));
  for (let i = 0; i < N; i++) {
    const p = P[i]; let x = p.x, y = p.y, col = 0, df = 0, zr = 0, al = .5, rz = 0;
    if (t >= 4) {
      const rg = ringPos(p, t); zr = rg[2];
      if (t < 8) { const k = lp(4, 8, p.d); x = p.x + (rg[0] - p.x) * k + (rg[1] - p.y) * .22 * p.sw * Math.sin(Math.PI * k);
        y = p.y + (rg[1] - p.y) * k - (rg[0] - p.x) * .22 * p.sw * Math.sin(Math.PI * k); col = k; }
      else if (t < 10) { x = rg[0]; y = rg[1]; col = 1; }
      else if (t < 14) { const k = lp(10, 14, p.d), s = spherePos(p, i);
        x = rg[0] + (s[0] - rg[0]) * k + (s[1] - rg[1]) * .18 * p.sw * Math.sin(Math.PI * k);
        y = rg[1] + (s[1] - rg[1]) * k - (s[0] - rg[0]) * .18 * p.sw * Math.sin(Math.PI * k); col = 1; df = k; rz = s[2]; p._y3 = s[3]; }
      else if (t < 16.5) { const s = spherePos(p, i); x = s[0]; y = s[1]; col = 1; df = 1; rz = s[2]; p._y3 = s[3]; }
      else { const k = lp(16.5, 18, p.d), s = spherePos(p, i);
        x = s[0] + (p.x - s[0]) * k + (p.y - s[1]) * .2 * p.sw * Math.sin(Math.PI * k);
        y = s[1] + (p.y - s[1]) * k - (p.x - s[0]) * .2 * p.sw * Math.sin(Math.PI * k); col = 1 - k; df = 1 - k; rz = s[2]; p._y3 = s[3]; }
    }
    // alpha and size
    let size = p.sz * 2.1, a = .26;
    if (col > 0) a = .38 + .3 * (.5 + .5 * zr) * (1 - df);
    if (df > 0) { const depth = .5 + .5 * rz; a = a * (1 - df) + (.2 + .6 * depth) * df; size *= (1 - df) + (.65 + .7 * depth) * df; }
    if (waveOn) { const dd = Math.abs(Math.hypot(p.x - cx, p.y - cy) - waveR); if (dd < 60) { const b = 1 - dd / 60; a += b * .6; size *= 1 + b * .9; } }
    // colour: dim while loose, group colour on rings, vertical gradient on the sphere
    const spr = df > .5 ? sprites[p._y3 > .33 ? 1 : p._y3 < -.33 ? 0 : 2] : sprites[p.g];
    const s = size * 2.6;
    if (col < .98) { ctx.globalAlpha = clamp(a * (1 - col)); ctx.drawImage(dimSprite, x - s / 2, y - s / 2, s, s); }
    if (col > .02) { ctx.globalAlpha = clamp(a * col); ctx.drawImage(spr, x - s / 2, y - s / 2, s, s); }
  }
  ctx.globalAlpha = 1;

  // sphere wireframe and verification ripples
  if (sw > .02) {
    ctx.globalCompositeOperation = 'lighter';
    const ca = Math.cos(tiltA), sa = Math.sin(tiltA);
    for (let k = 0; k < 6; k++) {
      ctx.beginPath();
      const a0 = rotY * 1 + k * Math.PI / 6;
      for (let j = 0; j <= 48; j++) {
        const u = j / 48 * TAU, x3 = Math.cos(a0) * Math.sin(u), z3 = Math.sin(a0) * Math.sin(u), y3 = Math.cos(u);
        const yy = y3 * ca - z3 * sa, zz = y3 * sa + z3 * ca, per = 1 / (1 - zz * .2);
        const X = cx + R * x3 * per, Y = cy + R * yy * per; j ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
      }
      stroke(`rgba(206,176,255,${.1 * sw})`, .8);
    }
    for (const lat of [-.6, -.2, .2, .6]) {
      ctx.beginPath(); const rr = Math.sqrt(1 - lat * lat);
      for (let j = 0; j <= 48; j++) {
        const u = j / 48 * TAU, x3 = Math.cos(u) * rr, z3 = Math.sin(u) * rr;
        const yy = lat * ca - z3 * sa, zz = lat * sa + z3 * ca, per = 1 / (1 - zz * .2);
        const X = cx + R * x3 * per, Y = cy + R * yy * per; j ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
      }
      stroke(`rgba(206,176,255,${.08 * sw})`, .8);
    }
  }
  if (t > 13.6 && t < 16.8) for (const off of [0, .5]) {
    const k = clamp((t - 13.6 - off) / 2.6); if (k <= 0 || k >= 1) continue;
    ctx.beginPath(); ctx.arc(cx, cy, R * (1.02 + k * .85), 0, TAU); stroke(`rgba(225,205,255,${.34 * (1 - k)})`, 1.4);
  }
  caption(t);
}

addEventListener('resize', resize);
try {
  window.parent.addEventListener('mousemove', e => { mx = e.clientX / W; my = e.clientY / H; mouseIn = true; });
  window.parent.document.addEventListener('mouseleave', () => { mouseIn = false; mx = .5; my = .5; });
} catch (e) {}
resize();
if (REDUCE) { for (let k = 0; k < 4; k++) draw(15.2, .016); } else requestAnimationFrame(frame);
})();
</script></body></html>
"""
