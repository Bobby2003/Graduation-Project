/**
 * AR REALM — 统一输入层：桌面键鼠 / AR 手势 / VR（占位）
 * 与 gesture.js 通过 localStorage + realm-action 事件协同
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

    function setMode(mode) {
        if (!['desktop', 'gesture', 'vr'].includes(mode)) return;
        state.mode = mode;
        localStorage.setItem(LS_KEY, mode);
        document.body.dataset.controlMode = mode;

        document.querySelectorAll('.mode-btn').forEach((btn) => {
            btn.classList.toggle('active', btn.dataset.mode === mode);
        });

        emitAction('control-mode-changed', { mode });
    }

    window.ARRealmControls = {
        setMode,
        getMode: () => state.mode,
        emitAction,
    };

    document.addEventListener('DOMContentLoaded', function () {
        document.body.dataset.controlMode = state.mode;

        document.querySelectorAll('.mode-btn').forEach((btn) => {
            btn.classList.toggle('active', btn.dataset.mode === state.mode);
            btn.addEventListener('click', function () {
                setMode(this.dataset.mode);
            });
        });

        setMode(state.mode);

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

        document.addEventListener('click', function () {
            if (state.mode !== 'desktop') return;
            if (!document.pointerLockElement) {
                document.body.requestPointerLock?.();
            }
        });

        document.addEventListener('pointerlockchange', function () {
            state.mouseLocked = document.pointerLockElement === document.body;
            emitAction('mouse-lock-changed', { locked: state.mouseLocked });
        });
    });
})();
