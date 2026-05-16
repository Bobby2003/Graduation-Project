/**
 * AR REALM — 统一输入层：桌面键鼠 / AR 手势 / VR（占位）
 * 3D 沉浸行为仅作用于 realm / training（#realm-canvas-root 或 body.ar-immersive-3d）
 */
(function () {
    const LS_KEY = 'ar_realm_control_mode';

    const state = {
        mode: localStorage.getItem(LS_KEY) || 'desktop',
        keys: {},
        mouseLocked: false,
        yaw: 0,
        pitch: 0,
    };

    if (!localStorage.getItem(LS_KEY)) {
        localStorage.setItem(LS_KEY, 'desktop');
        state.mode = 'desktop';
    }

    function isHomePage() {
        if (document.body && document.body.dataset.arHomePage === '1') {
            return true;
        }
        var p = (window.location.pathname || '/').replace(/\/+$/, '') || '/';
        return p === '/' || p === '';
    }

    function forceDesktopOnHomePage() {
        if (!isHomePage()) return;
        state.mode = 'desktop';
        try {
            localStorage.setItem(LS_KEY, 'desktop');
        } catch (e) {}
    }

    forceDesktopOnHomePage();

    function isImmersive3DPage() {
        return !!(
            document.getElementById('realm-canvas-root') ||
            (document.body && document.body.classList.contains('ar-immersive-3d'))
        );
    }

    function normalizeMode(mode) {
        if (mode !== 'desktop' && mode !== 'gesture' && mode !== 'vr') {
            return 'desktop';
        }
        return mode;
    }

    function releaseDesktopInput() {
        state.keys = {};
        state.mouseLocked = false;
        if (document.pointerLockElement) {
            document.exitPointerLock?.();
        }
    }

    function syncRealmBodyMode(mode) {
        if (!document.body) return;
        mode = normalizeMode(mode);
        document.body.dataset.controlMode = mode;
        document.body.classList.remove(
            'realm-mode-desktop',
            'realm-mode-gesture',
            'realm-mode-vr'
        );
        document.body.classList.add('realm-mode-' + mode);
    }

    window.syncRealmBodyMode = syncRealmBodyMode;
    window.ARRealmIsImmersive3DPage = isImmersive3DPage;

    function emitAction(action, payload) {
        window.dispatchEvent(
            new CustomEvent('realm-action', {
                detail: {
                    action,
                    mode: state.mode,
                    ...payload,
                },
            })
        );
    }

    function applyModeToDocument(mode) {
        if (!document.body || !isImmersive3DPage()) return;
        mode = normalizeMode(mode);
        document.body.dataset.controlMode = mode;
        document.body.classList.remove('realm-mode-desktop', 'realm-mode-gesture', 'realm-mode-vr');
        document.body.classList.add('realm-mode-' + mode);
    }

    function setMode(mode) {
        if (mode !== 'desktop' && mode !== 'gesture' && mode !== 'vr') return;

        const prev = state.mode;
        state.mode = mode;

        try {
            localStorage.setItem(LS_KEY, mode);
        } catch (e) {}

        syncRealmBodyMode(mode);
        if (isImmersive3DPage()) {
            applyModeToDocument(mode);
        } else {
            releaseDesktopInput();
        }

        document.querySelectorAll('.mode-btn').forEach((btn) => {
            btn.classList.toggle('active', btn.dataset.mode === mode);
        });

        if (prev === 'desktop' && mode !== 'desktop') {
            releaseDesktopInput();
        }

        emitAction('control-mode-changed', { mode });
    }

    window.ARRealmControls = {
        setMode,
        getMode: () => state.mode,
        emitAction,
        isImmersive3DPage,
        isHomePage,
        forceDesktopOnHomePage,
    };

    document.addEventListener('DOMContentLoaded', function () {
        forceDesktopOnHomePage();
        syncRealmBodyMode(state.mode);
        if (isImmersive3DPage()) {
            applyModeToDocument(state.mode);
        }

        document.querySelectorAll('.mode-btn').forEach((btn) => {
            btn.classList.toggle('active', btn.dataset.mode === state.mode);
            btn.addEventListener('click', function (e) {
                const mode = this.dataset.mode;
                if (mode === 'desktop' || mode === 'gesture' || mode === 'vr') {
                    setMode(mode);
                }
                if (e.currentTarget && typeof e.currentTarget.blur === 'function') {
                    e.currentTarget.blur();
                }
            });
        });

        setMode(state.mode);

        if (!isImmersive3DPage()) {
            return;
        }

        document.addEventListener('keydown', function (e) {
            if (state.mode !== 'desktop') return;

            state.keys[e.code] = true;

            switch (e.code) {
                case 'KeyW':
                    emitAction('move-forward');
                    break;
                case 'KeyS':
                    emitAction('move-backward');
                    break;
                case 'KeyA':
                    emitAction('move-left');
                    break;
                case 'KeyD':
                    emitAction('move-right');
                    break;
                case 'Space':
                    emitAction('jump');
                    break;
                case 'ShiftLeft':
                case 'ShiftRight':
                    emitAction('sprint');
                    break;
                case 'KeyE':
                    emitAction('interact');
                    break;
                case 'KeyF':
                    emitAction('use-skill');
                    break;
                case 'Tab':
                    e.preventDefault();
                    emitAction('open-inventory');
                    break;
                case 'Escape':
                    emitAction('open-menu');
                    break;
                default:
                    break;
            }
        });

        document.addEventListener('keyup', function (e) {
            if (state.mode !== 'desktop') return;

            state.keys[e.code] = false;

            switch (e.code) {
                case 'KeyW':
                    emitAction('stop-forward');
                    break;
                case 'KeyS':
                    emitAction('stop-backward');
                    break;
                case 'KeyA':
                    emitAction('stop-left');
                    break;
                case 'KeyD':
                    emitAction('stop-right');
                    break;
                case 'ShiftLeft':
                case 'ShiftRight':
                    emitAction('stop-sprint');
                    break;
                default:
                    break;
            }
        });

        document.addEventListener('mousemove', function (e) {
            if (state.mode !== 'desktop') return;
            if (!state.mouseLocked) return;

            state.yaw += e.movementX * 0.1;
            state.pitch -= e.movementY * 0.1;
            state.pitch = Math.max(-89, Math.min(89, state.pitch));

            emitAction('look', {
                yaw: state.yaw,
                pitch: state.pitch,
                dx: e.movementX,
                dy: e.movementY,
            });
        });

        document.addEventListener('pointerlockchange', function () {
            var canvas = document.querySelector('#realm-canvas-root canvas');
            var lockedEl = document.pointerLockElement;
            state.mouseLocked =
                lockedEl === document.body || !!(canvas && lockedEl === canvas);
            emitAction('mouse-lock-changed', { locked: state.mouseLocked });
        });
    });

    window.addEventListener('realm-action', function (e) {
        const d = e.detail || {};
        if (d.action === 'control-mode-changed' && d.mode) {
            syncRealmBodyMode(d.mode);
        }
    });

    window.addEventListener('storage', function (e) {
        if (e.key === LS_KEY) {
            syncRealmBodyMode(e.newValue || 'desktop');
        }
    });
})();
