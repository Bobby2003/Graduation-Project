/**
 * AR REALM — 浏览器端手势驱动（MediaPipe Hand Landmarker）
 * 本地：摄像头 → WASM → 立即更新光标 / 捏合 / 滚动
 * WebSocket：将识别结果中继给其他标签页/客户端（本机已本地处理，不重复应用下行数据）
 */

const MP_TASKS_VERSION = '0.10.14';
const WASM_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const TASKS_ESM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm';
const HAND_MODEL_URL =
    'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

const cursorStyles = `
    #ar-cursor {
        position: fixed;
        left: 0;
        top: 0;
        width: 18px;
        height: 18px;
        border-radius: 50%;
        background-color: #00f2ff;
        box-shadow: 0 0 15px #00f2ff, 0 0 30px #00f2ff;
        pointer-events: none;
        z-index: 2147483647;
        transform: translate(-50%, -50%);
        transition: width 0.15s cubic-bezier(0.175, 0.885, 0.32, 1.275),
                    height 0.15s cubic-bezier(0.175, 0.885, 0.32, 1.275),
                    background-color 0.15s, box-shadow 0.15s, border 0.15s;
        box-sizing: border-box;
        border: 2px solid transparent;
        display: none;
    }
    #ar-cursor.cursor-locked {
        width: 48px !important;
        height: 48px !important;
        background-color: transparent !important;
        border: 3px solid #fff !important;
        box-shadow: 0 0 20px rgba(255,255,255,0.8), inset 0 0 15px rgba(255,255,255,0.5) !important;
    }
`;

const styleSheet = document.createElement('style');
styleSheet.innerText = cursorStyles;
document.head.appendChild(styleSheet);

function lmHypot(dx, dy) {
    return Math.hypot(dx, dy);
}

/** @param {{x:number,y:number,z?:number}[]} lm */
function isPinch(lm) {
    const ps = lmHypot(lm[0].x - lm[9].x, lm[0].y - lm[9].y);
    return ps > 1e-5 && lmHypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y) < ps * 0.25;
}

function isPointing(lm) {
    const indexUp = lm[8].y < lm[6].y;
    const othersBent = [ [12, 10], [16, 14], [20, 18] ].every(
        ([t, m]) => lm[t].y > lm[m].y
    );
    return indexUp && othersBent;
}

function getPointerNorm(lm) {
    return { x: 1 - lm[8].x, y: lm[8].y };
}

/** 张开手掌：至少 4 指伸展（兼容镜像与不同摄像头角度） */
function isOpenPalm(lm) {
    const tips = [4, 8, 12, 16, 20];
    const bases = [2, 5, 9, 13, 17];
    let extended = 0;
    for (let i = 0; i < 5; i++) {
        const tip = tips[i];
        const base = bases[i];
        const dTip = lmHypot(lm[tip].x - lm[0].x, lm[tip].y - lm[0].y);
        const dBase = lmHypot(lm[base].x - lm[0].x, lm[base].y - lm[0].y);
        if (dTip > dBase * 1.08) extended++;
    }
    return extended >= 4;
}

function isFist(lm) {
    const tips = [8, 12, 16, 20];
    const mcps = [5, 9, 13, 17];
    let bent = 0;
    for (let i = 0; i < 4; i++) {
        if (lm[tips[i]].y > lm[mcps[i]].y) bent++;
    }
    return bent >= 4;
}

class ARGestureController {
    constructor() {
        this.statusText =
            document.getElementById('gesture-status-text') ||
            document.getElementById('realm-ar-gesture-status');
        this.gestureWidget = document.getElementById('ar-gesture-widget');
        this._detectLogFrame = 0;

        this.ws = null;
        this.arActive = false;
        this.lastX = 0;
        this.lastY = 0;
        this.lastClickTime = 0;
        this.lastHoveredElement = null;

        this.prevPalmY = null;
        this.handLandmarker = null;
        this.visionModule = null;
        this.rafId = null;
        this.stream = null;
        this.video = null;

        this._lastWsSend = 0;
        this._openPalmFrames = 0;
        this._palmMenuOpen = false;

        this.selectors =
            'a, button, input, .realm-card, .market-card, .btn-action, .loot-item, .rank-item, .btn-link, ' +
            '#realm-ar-menu button, [data-ar-action], #realm-vr-menu-btn, .mode-btn';

        this.createCursor();
        this.wireToggle();

        window.__arGesture = this;
        window.toggleGesture = () => this.toggle();

        console.log('[AR System] 浏览器手势驱动就绪（需摄像头权限）');
    }

    createCursor() {
        const oldCursor = document.getElementById('ar-cursor');
        if (oldCursor) oldCursor.remove();

        this.cursor = document.createElement('div');
        this.cursor.id = 'ar-cursor';
        document.body.appendChild(this.cursor);
    }

    wireToggle() {
        window.addEventListener('realm-action', (ev) => {
            const d = ev.detail;
            if (!d || d.action !== 'control-mode-changed') return;
            if (d.mode === 'gesture') {
                this.startAR().catch((e) => console.error('[AR System] 手势模式启动失败:', e));
            } else {
                this.stopAR();
                this._palmMenuOpen = false;
                this._openPalmFrames = 0;
            }
        });

        const hasModeSwitcher = document.querySelector('.mode-btn');
        const autostart = document.body?.dataset?.arGestureAutostart === '1';
        const stored = (typeof localStorage !== 'undefined' && localStorage.getItem('ar_realm_control_mode')) || 'desktop';

        if (hasModeSwitcher) {
            if (this.statusText) {
                const m = stored;
                this.statusText.innerText = m === 'gesture' ? 'MODE: GESTURE' : 'MODE: ' + String(m).toUpperCase();
                this.statusText.style.color = '#8892b0';
            }
            queueMicrotask(() => {
                const m =
                    (window.ARRealmControls && window.ARRealmControls.getMode()) ||
                    localStorage.getItem('ar_realm_control_mode') ||
                    'desktop';
                if (m === 'gesture') {
                    this.startAR().catch((e) => {
                        console.error('[AR System] 手势模式启动失败:', e);
                        this.showRealmGestureToast('摄像头启动失败，请允许权限');
                    });
                }
            });
            return;
        }

        if (autostart || stored === 'gesture') {
            queueMicrotask(() => {
                this.startAR().catch((e) => console.error('[AR System] 自动启动失败:', e));
            });
        } else if (this.statusText) {
            this.statusText.innerText = 'CONTROL: DESKTOP';
            this.statusText.style.color = '#8892b0';
        }
    }

    toggle() {
        if (this.arActive) {
            this.stopAR();
        } else {
            this.startAR().catch((e) => console.error('[AR System] 启动失败:', e));
        }
    }

    async startAR() {
        if (this.arActive) return;
        console.log('[AR System] startAR begin', { wasm: WASM_BASE, esm: TASKS_ESM });

        if (window.ARRealmControls && window.ARRealmControls.getMode() !== 'gesture') {
            window.ARRealmControls.setMode('gesture');
        }

        this.arActive = true;

        if (this.gestureWidget) {
            this.gestureWidget.style.display = 'block';
        }

        if (this.statusText) {
            this.statusText.innerText = 'AR: 正在启动摄像头…';
            this.statusText.style.color = '#ffd080';
        }

        try {
            this.ensureVideoElement();
            await this.openCamera();
            await this.loadLandmarker();
            this.connectWebSocket();
            this.loopDetect();
            console.log('[AR System] startAR success');
            this.showRealmGestureToast('AR 就绪 · 任意手张开掌打开菜单 · M 键备用');
        } catch (e) {
            console.error('[AR System] startAR fail', e);
            this.stopAR();
            const msg =
                (e && e.name === 'NotAllowedError')
                    ? '请允许摄像头权限后重试'
                    : 'AR 启动失败: ' + (e && e.message ? e.message : String(e));
            this.showRealmGestureToast(msg);
            throw e;
        }
    }

    stopAR() {
        this.arActive = false;

        if (this.rafId != null) {
            cancelAnimationFrame(this.rafId);
            this.rafId = null;
        }

        if (this.stream) {
            this.stream.getTracks().forEach((t) => t.stop());
            this.stream = null;
        }

        if (this.video) {
            this.video.srcObject = null;
        }

        if (this.handLandmarker) {
            try {
                this.handLandmarker.close();
            } catch (_) { /* noop */ }
            this.handLandmarker = null;
        }

        if (this.ws) {
            this.ws.onclose = null;
            this.ws.close();
            this.ws = null;
        }

        this.prevPalmY = null;

        if (this.cursor && this.cursor.style.display !== 'none') {
            this.cursor.style.display = 'none';
            this.cursor.style.transform = 'translate(-50%, -50%) scale(1)';
            this.clearHover();
        }

        if (this.gestureWidget) {
            this.gestureWidget.style.display = 'none';
        }
        if (this.statusText) {
            const m =
                (typeof localStorage !== 'undefined' && localStorage.getItem('ar_realm_control_mode')) || 'desktop';
            this.statusText.innerText = 'MODE: ' + String(m).toUpperCase();
            this.statusText.style.color = '#8892b0';
        }
    }

    ensureVideoElement() {
        let v = document.getElementById('ar-gesture-video');
        if (!v) {
            v = document.createElement('video');
            v.id = 'ar-gesture-video';
            v.setAttribute('playsinline', '');
            v.setAttribute('muted', '');
            v.muted = true;
            Object.assign(v.style, {
                position: 'fixed',
                opacity: '0',
                width: '1px',
                height: '1px',
                pointerEvents: 'none',
            });
            document.body.appendChild(v);
        }
        this.video = v;
    }

    async openCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
                audio: false,
            });
            this.stream = stream;
            this.video.srcObject = stream;
            await this.video.play();

            await new Promise((resolve) => {
                const ok = () => {
                    if (this.video.videoWidth > 0) {
                        this.video.removeEventListener('loadeddata', ok);
                        resolve();
                    }
                };
                this.video.addEventListener('loadeddata', ok);
                ok();
            });
            console.log('[AR System] camera success', {
                width: this.video.videoWidth,
                height: this.video.videoHeight,
            });
        } catch (e) {
            console.error('[AR System] camera fail', e);
            throw e;
        }
    }

    async loadLandmarker() {
        try {
            if (!this.visionModule) {
                console.log('[AR System] mediapipe import', TASKS_ESM);
                this.visionModule = await import(TASKS_ESM);
                console.log('[AR System] mediapipe import success');
            }
        } catch (e) {
            console.error('[AR System] mediapipe import fail', e);
            throw e;
        }
        const { HandLandmarker, FilesetResolver } = this.visionModule;

        const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);

        const baseOpts = {
            modelAssetPath: HAND_MODEL_URL,
            delegate: 'GPU',
        };

        const opts = {
            baseOptions: baseOpts,
            runningMode: 'VIDEO',
            numHands: 2,
            minHandDetectionConfidence: 0.7,
            minHandPresenceConfidence: 0.7,
            minTrackingConfidence: 0.7,
        };

        try {
            this.handLandmarker = await HandLandmarker.createFromOptions(fileset, opts);
            console.log('[AR System] HandLandmarker ready (GPU)');
        } catch (e) {
            console.warn('[AR System] GPU delegate 不可用，回退 CPU', e);
            opts.baseOptions = { ...baseOpts, delegate: 'CPU' };
            this.handLandmarker = await HandLandmarker.createFromOptions(fileset, opts);
            console.log('[AR System] HandLandmarker ready (CPU)');
        }
    }

    connectWebSocket() {
        const wsProtocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
        const wsUrl = wsProtocol + window.location.host + '/ws/gestures/';

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            if (this.statusText) {
                this.statusText.innerText = 'LINK: ONLINE (LOCAL)';
                this.statusText.style.color = '#00f2ff';
            }
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                window.dispatchEvent(new CustomEvent('ar-gesture-remote', { detail: data }));
            } catch (_) { /* ignore */ }
        };

        this.ws.onclose = () => {
            this.ws = null;
            if (!this.arActive) return;
            if (this.statusText) {
                this.statusText.innerText = 'LINK: LOST';
                this.statusText.style.color = '#ff3e3e';
            }
            setTimeout(() => {
                if (this.arActive && !this.ws) {
                    this.connectWebSocket();
                }
            }, 3000);
        };
    }

    buildPayload(result) {
        let rightGesture = 'none';
        let leftGesture = 'none';
        let scroll = null;
        let pointer = null;
        let palmOpen = false;
        let palmFist = false;

        const n = result.landmarks?.length || 0;

        for (let i = 0; i < n; i++) {
            const lm = result.landmarks[i];
            if (isOpenPalm(lm)) palmOpen = true;
            if (isFist(lm)) palmFist = true;
        }

        for (let i = 0; i < n; i++) {
            const lm = result.landmarks[i];
            const cat = result.handednesses?.[i]?.[0];
            const label = cat?.categoryName || cat?.displayName || '';

            const palmThisHand = isOpenPalm(lm);
            const fistThisHand = isFist(lm);
            const treatAsMenuHand = palmThisHand || (palmOpen && !isPointing(lm) && !isPinch(lm));

            if (treatAsMenuHand) {
                leftGesture = 'open';
            } else if (fistThisHand) {
                leftGesture = 'fist';
            }

            const useForPointer =
                !palmThisHand &&
                (label === 'Right' || (label !== 'Left' && rightGesture === 'none'));

            if (useForPointer) {
                if (isPinch(lm)) {
                    rightGesture = 'pinch';
                } else if (isPointing(lm)) {
                    rightGesture = 'point';
                    pointer = getPointerNorm(lm);
                }
            }
        }

        if (palmOpen) leftGesture = 'open';
        if (palmFist && !palmOpen) leftGesture = 'fist';

        if (rightGesture === 'none' && n > 0 && !palmOpen) {
            const lm0 = result.landmarks[0];
            if (isPinch(lm0)) rightGesture = 'pinch';
            else if (isPointing(lm0)) {
                rightGesture = 'point';
                pointer = getPointerNorm(lm0);
            }
        }

        return { right: rightGesture, left: leftGesture, scroll, pointer, palmOpen, palmFist };
    }

    maybeSendWs(payload) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;

        const now = performance.now();
        if (now - this._lastWsSend < 33) return;
        this._lastWsSend = now;
        this.ws.send(JSON.stringify(payload));
    }

    loopDetect() {
        if (!this.arActive || !this.handLandmarker || !this.video) return;
        console.log('[AR System] loopDetect start');

        const step = () => {
            if (!this.arActive) return;

            if (this.video.videoWidth > 0) {
                const result = this.handLandmarker.detectForVideo(this.video, performance.now());
                const payload = this.buildPayload(result);
                this._detectLogFrame += 1;
                if (this._detectLogFrame % 45 === 0) {
                    console.log('[AR System] detect', {
                        handCount: result.landmarks?.length || 0,
                        palmOpen: payload.palmOpen,
                        left: payload.left,
                    });
                }
                this.handleData(payload);
                this.maybeSendWs(payload);
            }

            this.rafId = requestAnimationFrame(step);
        };

        this.rafId = requestAnimationFrame(step);
    }

    handleData(data) {
        if (data.right === 'point' && data.pointer) {
            const x = Math.round(data.pointer.x * window.innerWidth);
            const y = Math.round(data.pointer.y * window.innerHeight);

            this.cursor.style.display = 'block';
            this.cursor.style.left = x + 'px';
            this.cursor.style.top = y + 'px';

            this.lastX = x;
            this.lastY = y;

            this.updateHover(x, y);
        } else if (data.right === 'pinch' && this.lastX) {
            this.cursor.style.transform = 'translate(-50%, -50%) scale(0.6)';
            this.cursor.style.backgroundColor = '#ff3e3e';
            this.doClick(this.lastX, this.lastY);

            window.dispatchEvent(
                new CustomEvent('ar-gesture', { detail: { action: 'grab', message: 'pinch' } })
            );
        } else {
            if (this.cursor.style.display !== 'none') {
                this.cursor.style.display = 'none';
                this.cursor.style.transform = 'translate(-50%, -50%) scale(1)';
                this.cursor.style.backgroundColor = '#00f2ff';
                this.clearHover();
            }
        }

        this.handlePalmMenuGestures(data);
    }

    handlePalmMenuGestures(data) {
        const mode =
            (window.ARRealmControls && window.ARRealmControls.getMode()) ||
            (typeof localStorage !== 'undefined' && localStorage.getItem('ar_realm_control_mode')) ||
            'desktop';
        if (mode !== 'gesture') {
            this._openPalmFrames = 0;
            return;
        }

        const palmOpen = !!(data.palmOpen || data.left === 'open');
        const palmFist = !!(data.palmFist || data.left === 'fist');

        if (palmOpen) {
            this._openPalmFrames += 1;
            if (!this._palmMenuOpen && this._openPalmFrames >= 3) {
                this._palmMenuOpen = true;
                console.log('[AR System] dispatch open-palm-menu', {
                    palmOpen: palmOpen,
                    frames: this._openPalmFrames,
                });
                window.dispatchEvent(
                    new CustomEvent('ar-gesture', { detail: { action: 'open-palm-menu' } })
                );
                this.showRealmGestureToast('已打开管理菜单');
            }
        } else {
            this._openPalmFrames = Math.max(0, this._openPalmFrames - 1);
        }

        if (palmFist && this._palmMenuOpen) {
            this._palmMenuOpen = false;
            this._openPalmFrames = 0;
            window.dispatchEvent(
                new CustomEvent('ar-gesture', { detail: { action: 'close-palm-menu' } })
            );
            this.showRealmGestureToast('菜单已关闭');
        }
    }

    showRealmGestureToast(msg) {
        let el = document.getElementById('realm-ar-gesture-status');
        if (!el) {
            el = document.getElementById('gesture-status-text');
        }
        if (el) {
            el.textContent = msg;
            el.style.color = '#7ee8ff';
        }
    }

    updateHover(x, y) {
        const target = document.elementFromPoint(x, y);
        const interactive = target ? target.closest(this.selectors) : null;

        if (this.lastHoveredElement !== interactive) {
            this.clearHover();
            if (interactive) {
                interactive.classList.add('ar-hover');
                this.cursor.classList.add('cursor-locked');
                this.lastHoveredElement = interactive;
            }
        }
    }

    clearHover() {
        if (this.lastHoveredElement) {
            this.lastHoveredElement.classList.remove('ar-hover');
            this.lastHoveredElement = null;
        }
        this.cursor.classList.remove('cursor-locked');
    }

    doClick(x, y) {
        if (Date.now() - this.lastClickTime < 600) return;
        this.lastClickTime = Date.now();

        const target = document.elementFromPoint(x, y);
        if (target) {
            const interact = target.closest(this.selectors);
            (interact || target).click();
            console.log('[AR] 点击了:', interact || target);
        }
    }
}

function boot() {
    new ARGestureController();
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
} else {
    boot();
}
