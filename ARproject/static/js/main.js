// ── 粒子背景 ──────────────────────────────────────────
const bg = document.getElementById('bg');
const bx = bg.getContext('2d');
let W, H, pts = [];

function resize() {
  W = bg.width = window.innerWidth;
  H = bg.height = window.innerHeight;
}

function mkPts(n) {
  pts = [];
  for (let i = 0; i < n; i++) pts.push({
    x: Math.random() * W, y: Math.random() * H,
    vx: (Math.random() - .5) * .35, vy: (Math.random() - .5) * .35,
    r: Math.random() * 1.4 + .4
  });
}

function drawBg() {
  bx.clearRect(0, 0, W, H);
  const gold = 'rgba(201,168,76,';
  pts.forEach(p => {
    p.x += p.vx; p.y += p.vy;
    if (p.x < 0 || p.x > W) p.vx *= -1;
    if (p.y < 0 || p.y > H) p.vy *= -1;
    bx.beginPath();
    bx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
    bx.fillStyle = gold + '.55)';
    bx.fill();
  });
  for (let i = 0; i < pts.length; i++) {
    for (let j = i + 1; j < pts.length; j++) {
      const dx = pts[i].x - pts[j].x, dy = pts[i].y - pts[j].y;
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d < 130) {
        bx.beginPath();
        bx.moveTo(pts[i].x, pts[i].y);
        bx.lineTo(pts[j].x, pts[j].y);
        bx.strokeStyle = gold + (1 - d / 130) * .18 + ')';
        bx.lineWidth = .6;
        bx.stroke();
      }
    }
  }
  requestAnimationFrame(drawBg);
}

resize();
mkPts(72);
drawBg();
window.addEventListener('resize', () => { resize(); mkPts(72); });

// ── 手势演示动画 ──────────────────────────────────────
const gc = document.getElementById('gc');
const gx = gc.getContext('2d');
const inds = [
  document.getElementById('i0'), document.getElementById('i1'),
  document.getElementById('i2'), document.getElementById('i3'),
  document.getElementById('i4')
];
const fpsEl = document.getElementById('gfps');

let gW, gH;
function resizeGc() {
  gW = gc.width = gc.offsetWidth;
  gH = gc.height = gc.offsetHeight;
}
resizeGc();
window.addEventListener('resize', resizeGc);

const CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17]
];

const POSES = {
  fist: (() => {
    const p = Array(21).fill(null).map(() => ({x:.5,y:.5}));
    p[0]={x:.50,y:.82}; p[1]={x:.46,y:.72}; p[2]={x:.42,y:.64}; p[3]={x:.40,y:.58}; p[4]={x:.39,y:.54};
    p[5]={x:.44,y:.60}; p[6]={x:.43,y:.54}; p[7]={x:.43,y:.50}; p[8]={x:.43,y:.47};
    p[9]={x:.50,y:.59}; p[10]={x:.50,y:.53}; p[11]={x:.50,y:.49}; p[12]={x:.50,y:.46};
    p[13]={x:.56,y:.60}; p[14]={x:.57,y:.54}; p[15]={x:.57,y:.50}; p[16]={x:.57,y:.47};
    p[17]={x:.62,y:.63}; p[18]={x:.63,y:.58}; p[19]={x:.63,y:.55}; p[20]={x:.63,y:.52};
    return p;
  })(),
  open: (() => {
    const p = Array(21).fill(null).map(() => ({x:.5,y:.5}));
    p[0]={x:.50,y:.88}; p[1]={x:.46,y:.78}; p[2]={x:.42,y:.68}; p[3]={x:.39,y:.60}; p[4]={x:.36,y:.54};
    p[5]={x:.44,y:.65}; p[6]={x:.43,y:.52}; p[7]={x:.42,y:.42}; p[8]={x:.42,y:.33};
    p[9]={x:.50,y:.63}; p[10]={x:.50,y:.50}; p[11]={x:.50,y:.39}; p[12]={x:.50,y:.30};
    p[13]={x:.56,y:.65}; p[14]={x:.57,y:.52}; p[15]={x:.57,y:.41}; p[16]={x:.57,y:.32};
    p[17]={x:.62,y:.68}; p[18]={x:.63,y:.57}; p[19]={x:.63,y:.48}; p[20]={x:.63,y:.40};
    return p;
  })(),
  pinch: (() => {
    const p = Array(21).fill(null).map(() => ({x:.5,y:.5}));
    p[0]={x:.50,y:.85}; p[1]={x:.46,y:.75}; p[2]={x:.42,y:.65}; p[3]={x:.40,y:.57}; p[4]={x:.44,y:.50};
    p[5]={x:.44,y:.63}; p[6]={x:.43,y:.52}; p[7]={x:.43,y:.44}; p[8]={x:.44,y:.50};
    p[9]={x:.50,y:.62}; p[10]={x:.50,y:.52}; p[11]={x:.50,y:.44}; p[12]={x:.50,y:.38};
    p[13]={x:.56,y:.63}; p[14]={x:.57,y:.53}; p[15]={x:.57,y:.45}; p[16]={x:.57,y:.38};
    p[17]={x:.62,y:.66}; p[18]={x:.63,y:.58}; p[19]={x:.63,y:.52}; p[20]={x:.63,y:.46};
    return p;
  })(),
  point: (() => {
    const p = Array(21).fill(null).map(() => ({x:.5,y:.5}));
    p[0]={x:.50,y:.88}; p[1]={x:.46,y:.78}; p[2]={x:.42,y:.68}; p[3]={x:.40,y:.60}; p[4]={x:.39,y:.55};
    p[5]={x:.44,y:.65}; p[6]={x:.43,y:.52}; p[7]={x:.42,y:.42}; p[8]={x:.42,y:.33};
    p[9]={x:.50,y:.64}; p[10]={x:.50,y:.56}; p[11]={x:.50,y:.52}; p[12]={x:.50,y:.49};
    p[13]={x:.56,y:.65}; p[14]={x:.57,y:.57}; p[15]={x:.57,y:.53}; p[16]={x:.57,y:.50};
    p[17]={x:.62,y:.68}; p[18]={x:.63,y:.61}; p[19]={x:.63,y:.57}; p[20]={x:.63,y:.54};
    return p;
  })()
};

const SEQUENCE = [
  { pose: 'fist',  ind: 0, label: '握拳检测中...' },
  { pose: 'open',  ind: 1, label: '张掌 → 菜单打开' },
  { pose: 'pinch', ind: 2, label: '捏合确认选择' },
  { pose: 'point', ind: 3, label: '食指光标移动' },
  { pose: 'open',  ind: 4, label: '菜单激活' }
];

let cur = POSES.fist.map(p => ({ x: p.x, y: p.y }));
let target = POSES.fist;
let seqIdx = 0, progress = 0, lastTime = 0;
let fps = 0, fpsCount = 0, fpsTimer = 0;

function lerpPts(from, to, t) {
  return from.map((p, i) => ({
    x: p.x + (to[i].x - p.x) * t,
    y: p.y + (to[i].y - p.y) * t
  }));
}

function drawHand(pts, alpha) {
  CONNECTIONS.forEach(([a, b]) => {
    gx.beginPath();
    gx.moveTo(pts[a].x * gW, pts[a].y * gH);
    gx.lineTo(pts[b].x * gW, pts[b].y * gH);
    gx.strokeStyle = `rgba(201,168,76,${alpha * .55})`;
    gx.lineWidth = 1.5;
    gx.stroke();
  });
  pts.forEach((p, i) => {
    const isTip = [4, 8, 12, 16, 20].includes(i);
    gx.beginPath();
    gx.arc(p.x * gW, p.y * gH, isTip ? 4 : 2.5, 0, Math.PI * 2);
    gx.fillStyle = isTip ? `rgba(201,168,76,${alpha})` : `rgba(255,255,255,${alpha * .6})`;
    gx.fill();
    if (isTip) {
      gx.beginPath();
      gx.arc(p.x * gW, p.y * gH, 7, 0, Math.PI * 2);
      gx.strokeStyle = `rgba(201,168,76,${alpha * .3})`;
      gx.lineWidth = 1;
      gx.stroke();
    }
  });
}

function animGesture(ts) {
  const dt = ts - lastTime;
  lastTime = ts;
  fpsTimer += dt;
  fpsCount++;
  if (fpsTimer >= 600) {
    fps = Math.round(fpsCount / (fpsTimer / 1000));
    fpsEl.textContent = fps + ' fps';
    fpsCount = 0;
    fpsTimer = 0;
  }

  progress += dt / 1800;
  if (progress >= 1) {
    progress = 0;
    seqIdx = (seqIdx + 1) % SEQUENCE.length;
    inds.forEach(el => el.classList.remove('on'));
    inds[SEQUENCE[seqIdx].ind].classList.add('on');
    target = POSES[SEQUENCE[seqIdx].pose];
  }

  cur = lerpPts(cur, target, 0.08);

  gx.clearRect(0, 0, gW, gH);

  // 扫描线效果
  const scanY = (ts * .04) % gH;
  gx.fillStyle = 'rgba(201,168,76,.03)';
  gx.fillRect(0, scanY, gW, 2);

  drawHand(cur, .9);

  gx.font = '10px monospace';
  gx.fillStyle = 'rgba(201,168,76,.6)';
  gx.fillText('▶ ' + SEQUENCE[seqIdx].label, 10, gH - 10);

  requestAnimationFrame(animGesture);
}

inds[0].classList.add('on');
requestAnimationFrame(animGesture);

// ── 数字滚动 ──────────────────────────────────────────
function countUp(el, target, duration) {
  let startTime = null;
  function step(ts) {
    if (!startTime) startTime = ts;
    const p = Math.min((ts - startTime) / duration, 1);
    const ease = 1 - Math.pow(1 - p, 3);
    el.textContent = Math.floor(ease * target);
    if (p < 1) requestAnimationFrame(step);
    else el.textContent = target;
  }
  requestAnimationFrame(step);
}

const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.querySelectorAll('[data-t]').forEach(el => {
        countUp(el, parseInt(el.dataset.t), 1800);
      });
      observer.unobserve(entry.target);
    }
  });
}, { threshold: 0.3 });

const statsEl = document.querySelector('.sin');
if (statsEl) observer.observe(statsEl);

// ── 滚动淡入 ──────────────────────────────────────────
const fadeObserver = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.style.opacity = '1';
      entry.target.style.transform = 'translateY(0)';
      fadeObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.1 });

document.querySelectorAll('.card, .step, .gitem, .ti').forEach(el => {
  el.style.opacity = '0';
  el.style.transform = 'translateY(20px)';
  el.style.transition = 'opacity .5s ease, transform .5s ease';
  fadeObserver.observe(el);
});
