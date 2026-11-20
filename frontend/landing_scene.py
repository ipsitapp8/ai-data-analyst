"""Landing page: the Silt intelligence core.

A transparent, page-fixed scene (for `st.iframe`), anchored to the `#lp-core` element of the landing page:

* a smooth, glossy, translucent form rendered live by a WebGL shader: a slowly morphing blob with light moving inside it,
  fresnel rim light, soft specular, and fine filaments radiating from a bright centre;
* three orbital paths (Planner, Analyst, Critic) drawn as bundles of thin strands carrying glass beads with motion streaks;
  the beads pass behind and in front of the form (three stacked canvases: back, WebGL, front);
* a pulse of light from the centre when the Critic completes, and thin threads from each agent panel to its orbit;
* far-away dust, and cursor-driven light and parallax written to CSS variables on the parent page.

The loop is a *demonstration* of how Silt works. It is not driven by, and does not show, any live analysis; the page labels it as illustrative.
Self-contained: no network, no libraries, no data from the app. If WebGL is unavailable a soft 2D glow replaces the shader.
`prefers-reduced-motion` draws one still frame.
"""

LANDING_SCENE_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="color-scheme" content="dark"><style>
html,body{margin:0;height:100%;background:transparent;overflow:hidden;color-scheme:dark}
canvas{position:absolute;left:0;top:0;display:block}
#back,#front{width:100%;height:100%}
</style></head><body><canvas id="back"></canvas><canvas id="gl"></canvas><canvas id="front"></canvas><script>
(() => {
const cb = document.getElementById('back'), cf = document.getElementById('front'), cg = document.getElementById('gl');
const bx = cb.getContext('2d'), fx = cf.getContext('2d');
const REDUCE = matchMedia('(prefers-reduced-motion: reduce)').matches;
const TAU = Math.PI * 2, T = 15;
const VIO = [139, 92, 246], LAV = [196, 181, 253], PINK = [244, 114, 182], WHITE = [255, 255, 255];
const AGENT_COL = [LAV, VIO, PINK];                       // Planner, Analyst, Critic
const WIN = [[1.0, 4.6], [4.8, 8.6], [8.8, 12.0]];        // when each agent is active within the 15 s loop
const PULSE_AT = 12.0;                                    // the core pulses when the Critic completes
let W = 0, H = 0, DPR = 1, dust = [], mobile = false;
let stage = { x: 0, y: 0, w: 0, h: 0, ok: false }, cx = 0, cy = 0, R = 0;
let mx = .5, my = .4, sx = .5, sy = .4, rotY = 0, tiltA = .3, active = -1;

const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const ss = (a, b, x) => { x = clamp((x - a) / (b - a)); return x * x * (3 - 2 * x); };
const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
const rnd = (a, b) => a + Math.random() * (b - a);

function makeSprite(c) {
  const s = 64, k = document.createElement('canvas'); k.width = k.height = s;
  const g = k.getContext('2d'), r = g.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
  r.addColorStop(0, 'rgba(255,255,255,1)'); r.addColorStop(.16, rgba(c, .9)); r.addColorStop(.5, rgba(c, .18)); r.addColorStop(1, rgba(c, 0));
  g.fillStyle = r; g.fillRect(0, 0, s, s); return k;
}
const SP = [makeSprite(LAV), makeSprite(VIO), makeSprite(PINK)];

// ---------------------------------------------------------------- the glossy translucent form (WebGL) ----
const VS = 'attribute vec2 p; void main(){ gl_Position = vec4(p, 0., 1.); }';
const FS = `
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
uniform vec2 uRes; uniform float uTime; uniform mat3 uRot; uniform vec3 uAct; uniform float uSweep; uniform float uPulse;
float hash(vec3 p){ p = fract(p * 0.3183099 + .1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }
float noise(vec3 x){
  vec3 i = floor(x), f = fract(x); f = f * f * (3. - 2. * f);
  return mix(mix(mix(hash(i), hash(i + vec3(1,0,0)), f.x), mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),
             mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x), mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y), f.z);
}
float map(vec3 p){
  vec3 q = uRot * p; float t = uTime * .16;
  vec3 w = q * 1.05 + vec3(t, -t * .7, t * .45);
  float d = length(q) - 1.0 - .26 * (noise(w) - .5) * 2. - .09 * (noise(w * 2.1 + 4.) - .5) * 2.;
  return d * .72;
}
vec3 nrm(vec3 p){ vec2 e = vec2(.004, -.004); return normalize(e.xyy * map(p + e.xyy) + e.yyx * map(p + e.yyx) + e.yxy * map(p + e.yxy) + e.xxx * map(p + e.xxx)); }
void main(){
  vec2 uv = (gl_FragCoord.xy - .5 * uRes) / (.5 * uRes.y);
  vec3 ro = vec3(uv * 1.55, -3.), rd = vec3(0., 0., 1.);
  float t = 0., minD = 10., tmin = 0.; bool hit = false;
  for (int i = 0; i < 38; i++) { vec3 p = ro + rd * t; float d = map(p); if (d < minD) { minD = d; tmin = t; } if (d < .004) { hit = true; break; } t += d; if (t > 5.) break; }
  float cover = 1.;
  if (!hit && minD < .035) { hit = true; t = tmin; cover = 1. - smoothstep(.004, .035, minD); }
  vec3 lav = vec3(.78, .70, 1.), vio = vec3(.55, .36, 1.), pnk = vec3(1., .46, .78);
  float energy = max(uAct.x, max(uAct.y, uAct.z));
  if (!hit) {
    float h = exp(-max(minD, 0.) * 3.6) * (1. - smoothstep(.45, .98, length(uv)));
    vec3 g = mix(vec3(.42, .16, .86), vec3(.95, .35, .85), .35) * h * (.30 + .22 * energy + .45 * uPulse);
    gl_FragColor = vec4(g, clamp(max(g.r, max(g.g, g.b)) * .9, 0., 1.)); return;
  }
  vec3 p = ro + rd * t, n = nrm(p);
  float nv = max(dot(n, -rd), 0.), fres = pow(1. - nv, 2.4);
  float core = 0., fil = 0.; vec3 tint = vec3(0.);
  float st = .17;
  for (int i = 0; i < 14; i++) {
    vec3 sp = p + rd * (float(i) * st + .05);
    if (map(sp) > .02) break;
    vec3 q = uRot * sp; float r = length(q);
    vec3 dir = q / max(r, .001);
    float f = noise(dir * 7. + vec3(0., uTime * .12, 0.));
    f = smoothstep(.5, .82, f) * .95 + smoothstep(.62, .95, noise(dir * 15. + 3.)) * .35;
    float band = pow(.5 + .5 * cos(atan(dir.z, dir.x) * 2. + uTime * 1.3 + dir.y * 3.), 5.);
    float sweep = exp(-pow((dir.y - uSweep) * 3.2, 2.));
    float chk = .55 + .45 * sin(uTime * 5. + r * 9.);
    float a = uAct.x * sweep + uAct.y * band + uAct.z * chk;
    float w = (1. - r * .62);
    fil += f * w * (1. + a * 2.2) * st;
    tint += (lav * uAct.x * sweep + vio * uAct.y * band + pnk * uAct.z * chk) * f * w * st;
    core += exp(-r * 3.3) * st;
  }
  vec3 body = mix(vec3(.16, .04, .40), vec3(.48, .14, .82), nv);
  vec3 col = body * (.42 + .42 * nv);
  col += mix(lav, pnk, .35 + .3 * dot(n, vec3(.3, -.6, .2))) * fil * .95;
  col += tint * 2.1;
  col += vec3(1., .93, 1.) * core * (1.5 + 3.2 * uPulse + .9 * energy);
  col += mix(pnk, lav, nv) * fres * (.95 + .8 * uPulse);
  vec3 L = normalize(vec3(-.45, .65, -.7)), h = normalize(L - rd);
  col += vec3(1.) * pow(max(dot(n, h), 0.), 60.) * .95 + vec3(.9, .8, 1.) * pow(max(dot(n, normalize(vec3(.5, -.4, -.8) - rd)), 0.), 28.) * .22;
  col += .05 * vec3(.6 + .4 * sin(n.x * 7.), .15, .7 + .3 * sin(n.z * 7. + 2.)) * fres;
  col = col / (1. + col * .35);
  float alpha = clamp(.80 + fres * .2 + core * .3, 0., 1.) * cover;
  gl_FragColor = vec4(col * cover * clamp(.80 + fres * .2 + core * .3, 0., 1.), alpha);
}`;
let gl = null, prog = null, U = {}, glScale = 1.05, glOK = false;
function initGL() {
  try { gl = cg.getContext('webgl', { alpha: true, premultipliedAlpha: true, antialias: false, powerPreference: 'low-power' }); } catch (e) { gl = null; }
  if (!gl) return;
  const sh = (t, s) => { const o = gl.createShader(t); gl.shaderSource(o, s); gl.compileShader(o); return gl.getShaderParameter(o, gl.COMPILE_STATUS) ? o : null; };
  const v = sh(gl.VERTEX_SHADER, VS), f = sh(gl.FRAGMENT_SHADER, FS); if (!v || !f) { gl = null; return; }
  prog = gl.createProgram(); gl.attachShader(prog, v); gl.attachShader(prog, f); gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) { gl = null; return; }
  gl.useProgram(prog);
  const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, 'p'); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  for (const n of ['uRes', 'uTime', 'uRot', 'uAct', 'uSweep', 'uPulse']) U[n] = gl.getUniformLocation(prog, n);
  gl.clearColor(0, 0, 0, 0); glOK = true;
}
function rotMat() {                                          // Rx(pitch) * Ry(yaw), column-major
  const cy_ = Math.cos(rotY), sy_ = Math.sin(rotY), cx_ = Math.cos(tiltA), sx_ = Math.sin(tiltA);
  return new Float32Array([cy_, sx_ * sy_, -cx_ * sy_, 0, cx_, sx_, sy_, -sx_ * cy_, cx_ * cy_]);
}
function drawGL(t, act, sweep, pulse) {
  const S = Math.round(R * 3.1);                              // canvas side in CSS px; the form is ~R across
  const px = Math.max(96, Math.round(S * Math.min(DPR, 1.5) * glScale));
  if (cg.width !== px) { cg.width = px; cg.height = px; }
  cg.style.width = S + 'px'; cg.style.height = S + 'px';
  cg.style.transform = `translate(${(cx - S / 2).toFixed(1)}px, ${(cy - S / 2).toFixed(1)}px)`;
  gl.viewport(0, 0, px, px); gl.clear(gl.COLOR_BUFFER_BIT);
  gl.uniform2f(U.uRes, px, px); gl.uniform1f(U.uTime, t); gl.uniformMatrix3fv(U.uRot, false, rotMat());
  gl.uniform3f(U.uAct, act[0], act[1], act[2]); gl.uniform1f(U.uSweep, sweep); gl.uniform1f(U.uPulse, pulse);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
}

// ---------------------------------------------------------------- page, geometry ----
function buildDust() {
  const n = mobile ? 70 : 140; dust = [];
  for (let i = 0; i < n; i++) dust.push({ x: Math.random(), y: Math.random(), z: rnd(.15, 1), tw: rnd(0, TAU), v: rnd(.004, .012) });
}
function resize() {
  DPR = Math.min(window.devicePixelRatio || 1, 2);
  W = cb.clientWidth; H = cb.clientHeight; mobile = W < 900;
  for (const c of [cb, cf]) { c.width = Math.round(W * DPR); c.height = Math.round(H * DPR); }
  bx.setTransform(DPR, 0, 0, DPR, 0, 0); fx.setTransform(DPR, 0, 0, DPR, 0, 0);
  buildDust();
}
function par() { try { return window.parent.document; } catch (e) { return null; } }
function anchor() {
  const d = par(), el = d && d.getElementById('lp-core');
  if (el) { const r = el.getBoundingClientRect(); stage = { x: r.left, y: r.top, w: r.width, h: r.height, ok: r.bottom > -80 && r.top < H + 80 && r.width > 60 }; }
  else stage = { x: W * .55, y: H * .16, w: W * .4, h: H * .64, ok: true };
  if (mobile) { cx = stage.x + stage.w * .5; cy = stage.y + Math.min(175, stage.h * .45); R = Math.min(stage.w * .27, 118); }
  else { cx = stage.x + stage.w * .42; cy = stage.y + stage.h * .5; R = Math.min(stage.w * .30, stage.h * .31); }
  journey();
}
// As the page scrolls the form leaves the hero and drifts across the background: it wanders left and right behind each
// section, swells, softens and shifts hue, and spins faster while the page is moving. The page itself is never touched.
let scrollEl = null, so = 0, lastSo = 0, spin = 0, jk = 0;
function journey() {
  if (REDUCE) return;
  let s = 0;
  try { const d = par(); scrollEl = scrollEl || (d && (d.querySelector('[data-testid="stMain"]') || d.scrollingElement)); s = scrollEl ? scrollEl.scrollTop : 0; } catch (e) {}
  so += (s - so) * .1;
  spin += (Math.min(6, Math.abs(so - lastSo) * .12) - spin) * .15; lastSo = so;
  const u = so / Math.max(1, H); jk = ss(.1, .9, u);
  if (jk > 0) {
    stage.ok = true;                                          // keep drawing after the hero has scrolled away
    const px = W * (.5 + .3 * Math.sin(u * 1.25 + .6)), py = H * (.47 + .08 * Math.cos(u * 1.6));
    cx += (px - cx) * jk; cy += (py - cy) * jk;
    R *= 1 + jk * (.35 + .2 * Math.sin(u * .9));
  }
  try { const fe = window.frameElement; if (fe) { fe.style.opacity = (1 - .5 * jk).toFixed(3); fe.style.filter = 'hue-rotate(' + (Math.sin(u * .5) * 32).toFixed(1) + 'deg) blur(' + (jk * 6).toFixed(1) + 'px)'; } } catch (e) {}
}
function panels() { const d = par(); return d ? [...d.querySelectorAll('.lp-panel')].map(p => p.getBoundingClientRect()) : []; }
function setVars() {
  const d = par(); if (!d) return; const s = d.documentElement.style;
  s.setProperty('--lp-px', ((sx - .5) * 2).toFixed(3)); s.setProperty('--lp-py', ((sy - .5) * 2).toFixed(3));
  s.setProperty('--lp-mx', (sx * W).toFixed(0) + 'px'); s.setProperty('--lp-my', (sy * H).toFixed(0) + 'px');
}
function setActive(a) {
  if (a === active) return; active = a; const d = par(); if (!d) return;
  d.querySelectorAll('.lp-panel').forEach((p, i) => p.classList.toggle('on', i === a));
  const st = d.getElementById('lp-core'); if (st) st.setAttribute('data-agent', a < 0 ? '' : ['planner', 'analyst', 'critic'][a]);
}

// ---------------------------------------------------------------- orbits and glass beads (2D) ----
const RINGS = [ { r: 1.52, k: .27, tilt: -.50, spin: .50, n: 9 }, { r: 1.80, k: .24, tilt: .46, spin: -.40, n: 8 }, { r: 2.08, k: .30, tilt: 1.12, spin: .33, n: 9 } ];
const STRANDS = [-1.5, -.5, .5, 1.5];                       // each orbit is a small bundle of fine strands, like a glass tube
function ringPoint(i, u, off) {
  const g = RINGS[i], a = g.tilt + (sy - .5) * .28, rr = g.r * R * (1 + (off || 0) * .006);
  const ex = rr * Math.cos(u), ey = rr * g.k * Math.sin(u);
  return [cx + ex * Math.cos(a) - ey * Math.sin(a), cy + ex * Math.sin(a) + ey * Math.cos(a), Math.sin(u)];
}
function strands(ctx, i, act, front) {
  const col = AGENT_COL[i], N = 90;
  for (const off of STRANDS) {
    ctx.beginPath(); let on = false;
    for (let k = 0; k <= N; k++) {
      const q = ringPoint(i, k / N * TAU, off), isFront = q[2] > 0;
      if (isFront !== front) { on = false; continue; }
      if (!on) { ctx.moveTo(q[0], q[1]); on = true; } else ctx.lineTo(q[0], q[1]);
    }
    ctx.strokeStyle = rgba(col, (front ? .30 : .12) + .5 * act * (front ? 1 : .5) - Math.abs(off) * .05); ctx.lineWidth = .7 + .5 * act; ctx.stroke();
  }
}
function bead(ctx, x, y, r, col, act, depth) {
  const k = .55 + .45 * depth;
  ctx.globalCompositeOperation = 'lighter';
  const sp = SP[AGENT_COL.indexOf(col)], gz = r * (3 + 4 * act);
  ctx.globalAlpha = (.35 + .5 * act) * k; ctx.drawImage(sp, x - gz, y - gz, gz * 2, gz * 2);
  ctx.globalCompositeOperation = 'source-over'; ctx.globalAlpha = 1;
  let g = ctx.createRadialGradient(x - r * .3, y - r * .35, r * .05, x, y, r);
  g.addColorStop(0, 'rgba(255,255,255,.98)'); g.addColorStop(.22, rgba([236, 228, 255], .85)); g.addColorStop(.6, rgba(col, .45 * k + .2)); g.addColorStop(1, rgba([40, 20, 90], .85));
  ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, r, 0, TAU); ctx.fill();
  ctx.strokeStyle = `rgba(255,255,255,${.35 * k + .25 * act})`; ctx.lineWidth = Math.max(.6, r * .09); ctx.beginPath(); ctx.arc(x, y, r - ctx.lineWidth / 2, 0, TAU); ctx.stroke();
  ctx.fillStyle = rgba(col, .5 * k); ctx.beginPath(); ctx.ellipse(x + r * .32, y + r * .4, r * .38, r * .2, -.6, 0, TAU); ctx.fill();
}
function beads(i, t, act) {
  const g = RINGS[i], col = AGENT_COL[i], spin = g.spin * (1 + 1.6 * act);
  for (let k = 0; k < g.n; k++) {
    const u = t * spin + k * TAU / g.n + Math.sin(k * 7.3) * .25, q = ringPoint(i, u, 0), front = q[2] > 0, ctx = front ? fx : bx;
    const size = (4.6 + 6.4 * ((k * 53) % 10) / 10) * (.8 + .4 * q[2]) * (mobile ? .8 : 1) * (1 + .35 * act);
    const dir = Math.sign(spin) || 1;
    for (let s = 1; s <= 9; s++) {
      const a = ringPoint(i, u - dir * s * .055, 0), b2 = ringPoint(i, u - dir * (s + 1) * .055, 0), c2 = a[2] > 0 ? fx : bx;
      c2.strokeStyle = rgba(col, (.5 - s * .05) * (.5 + .7 * act)); c2.lineWidth = Math.max(.4, size * (.55 - s * .05)); c2.beginPath(); c2.moveTo(a[0], a[1]); c2.lineTo(b2[0], b2[1]); c2.stroke();
    }
    bead(ctx, q[0], q[1], size, col, act, .5 + .5 * q[2]);
  }
}

let last = 0, clock = 0, perf = 0;
function frame(now) {
  const dt = Math.min(.05, (now - last) / 1000 || .016); last = now; clock += dt;
  if (!REDUCE) requestAnimationFrame(frame);
  try {
    draw(REDUCE ? 12.4 : clock % T, dt);
    if (glOK && !REDUCE) {                                    // keep it light: step the shader's resolution down if frames run long
      perf = perf * .94 + dt * .06;
      if (perf > .026 && glScale > .45) { glScale *= .85; perf = .016; }
    }
  } catch (e) { for (const c of [bx, fx]) { c.globalAlpha = 1; c.globalCompositeOperation = 'source-over'; } if (window.console) console.error('landing scene', e); }
}

function draw(t, dt) {
  sx += (mx - sx) * .06; sy += (my - sy) * .06; setVars();
  for (const c of [bx, fx]) { c.clearRect(0, 0, W, H); c.globalCompositeOperation = 'source-over'; c.globalAlpha = 1; }

  for (const p of dust) {
    p.y -= p.v * dt * (REDUCE ? 0 : 1); if (p.y < -.02) { p.y = 1.02; p.x = Math.random(); }
    const x = p.x * W + (sx - .5) * -40 * p.z, y = p.y * H + (sy - .5) * -26 * p.z, tw = .45 + .55 * Math.sin(p.tw + t * .8);
    bx.fillStyle = `rgba(214,200,255,${(.12 + .3 * p.z) * tw})`; bx.fillRect(x, y, .8 + p.z * 1.1, .8 + p.z * 1.1);
  }

  anchor();
  const act = WIN.map(w => ss(w[0], w[0] + .5, t) * (1 - ss(w[1] - .3, w[1] + .5, t)));
  const which = act.reduce((b, v, i) => v > .5 ? i : b, -1);
  setActive(REDUCE ? 2 : which);
  if (!stage.ok) { if (glOK) cg.style.transform = 'translate(-9999px,-9999px)'; return; }

  const pulse = t > PULSE_AT && t < PULSE_AT + 1.9 ? (t - PULSE_AT) / 1.9 : -1, pf = pulse >= 0 ? Math.sin(Math.PI * pulse) : 0;
  const energy = Math.max(act[0], act[1], act[2]) * .6 + pf;
  rotY += dt * (.14 + spin * .9 + (sx - .5) * .5); tiltA = .3 + (sy - .5) * .45;

  bx.globalCompositeOperation = 'lighter';
  let g = bx.createRadialGradient(cx, cy, 0, cx, cy, R * 3.2);
  g.addColorStop(0, rgba(VIO, .20 + .18 * energy)); g.addColorStop(.4, rgba(PINK, .05 + .06 * energy)); g.addColorStop(1, rgba(VIO, 0));
  bx.fillStyle = g; bx.fillRect(cx - R * 3.2, cy - R * 3.2, R * 6.4, R * 6.4); bx.globalCompositeOperation = 'source-over';

  for (let i = 0; i < 3; i++) { strands(bx, i, act[i], false); strands(fx, i, act[i], true); }
  for (let i = 0; i < 3; i++) beads(i, t, act[i]);

  if (glOK) drawGL(t, act, ss(WIN[0][0], WIN[0][1] - .4, t) * 2.2 - 1.1, pf);
  else {
    const o = fx.createRadialGradient(cx - R * .3, cy - R * .35, R * .1, cx, cy, R * 1.1);
    o.addColorStop(0, 'rgba(255,240,255,.95)'); o.addColorStop(.3, rgba(PINK, .55)); o.addColorStop(.7, rgba(VIO, .5)); o.addColorStop(1, rgba(VIO, 0));
    bx.fillStyle = o; bx.beginPath(); bx.arc(cx, cy, R * 1.1, 0, TAU); bx.fill();
  }

  const pr = jk > .05 ? [] : panels();                          // the cards have scrolled away: no connector lines
  for (let i = 0; i < Math.min(3, pr.length); i++) {
    const b = pr[i], py = b.top + b.height / 2, from = [(b.left + b.width / 2) < cx ? b.right : b.left, py], port = ringPoint(i, [-2.1, .25, 2.4][i], 0), a = act[i];
    const gr = fx.createLinearGradient(from[0], from[1], port[0], port[1]);
    gr.addColorStop(0, rgba(AGENT_COL[i], .06 + .5 * a)); gr.addColorStop(1, rgba(AGENT_COL[i], .02 + .35 * a));
    fx.strokeStyle = gr; fx.lineWidth = 1; fx.beginPath(); fx.moveTo(from[0], from[1]);
    fx.quadraticCurveTo((from[0] + port[0]) / 2, from[1] * .35 + port[1] * .65, port[0], port[1]); fx.stroke();
    if (a > .05) { const k = (t * .7) % 1, qx = from[0] + (port[0] - from[0]) * k, qy = from[1] + (port[1] - from[1]) * k, z = 14 * a; fx.globalCompositeOperation = 'lighter'; fx.globalAlpha = a; fx.drawImage(SP[i], qx - z, qy - z, z * 2, z * 2); fx.globalAlpha = 1; fx.globalCompositeOperation = 'source-over'; }
  }
  if (pulse >= 0) {
    fx.globalCompositeOperation = 'lighter';
    for (const off of [0, .18]) { const k = clamp(pulse - off); if (k <= 0 || k >= 1) continue; fx.beginPath(); fx.arc(cx, cy, R * (.7 + k * 2.4), 0, TAU); fx.strokeStyle = rgba(WHITE, .5 * (1 - k)); fx.lineWidth = 1.4 * (1 - k) + .4; fx.stroke(); }
    fx.globalCompositeOperation = 'source-over';
  }
}

try {
  const w = window.parent;
  w.addEventListener('mousemove', e => { mx = e.clientX / W; my = e.clientY / H; });
  w.document.addEventListener('mouseleave', () => { mx = .5; my = .4; });
  w.document.addEventListener('mousemove', e => {
    const b = e.target && e.target.closest ? e.target.closest('.lp-btn') : null; if (!b) return; const r = b.getBoundingClientRect();
    b.style.transform = `translate(${((e.clientX - (r.left + r.width / 2)) * .16).toFixed(1)}px, ${((e.clientY - (r.top + r.height / 2)) * .28).toFixed(1)}px)`;
  });
  w.document.addEventListener('mouseout', e => { const b = e.target && e.target.closest ? e.target.closest('.lp-btn') : null; if (b && !b.contains(e.relatedTarget)) b.style.transform = ''; });
} catch (e) {}
addEventListener('resize', resize);
initGL(); resize();
if (REDUCE) { for (let k = 0; k < 3; k++) draw(12.4, .016); } else requestAnimationFrame(frame);
})();
</script></body></html>
"""
