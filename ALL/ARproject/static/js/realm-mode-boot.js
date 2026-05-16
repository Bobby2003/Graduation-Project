/**
 * Realm 模式同步兜底（在 controls.js 之后加载）
 * 解决浏览器缓存旧 controls.js 时 body class 与 getMode() 不一致。
 */
(function () {
    'use strict';

    var BOOT_VERSION = '2026-05-16c';
    window.__REALM_MODE_BOOT_VERSION = BOOT_VERSION;

    function normalizeMode(mode) {
        if (mode === 'desktop' || mode === 'gesture' || mode === 'vr') return mode;
        return 'desktop';
    }

    function syncRealmBodyMode(mode) {
        mode = normalizeMode(mode);
        if (!document.body) return;
        document.body.dataset.controlMode = mode;
        document.body.classList.remove(
            'realm-mode-desktop',
            'realm-mode-gesture',
            'realm-mode-vr'
        );
        document.body.classList.add('realm-mode-' + mode);
        console.log('[REALM MODE BOOT]', {
            version: BOOT_VERSION,
            mode: mode,
            bodyMode: document.body.dataset.controlMode,
            bodyClass: document.body.className,
        });
    }

    if (typeof window.syncRealmBodyMode !== 'function') {
        window.syncRealmBodyMode = syncRealmBodyMode;
        console.warn('[REALM MODE BOOT] installed syncRealmBodyMode (cached controls.js)');
    }

    function resolveMode() {
        if (window.ARRealmControls && typeof window.ARRealmControls.getMode === 'function') {
            return window.ARRealmControls.getMode();
        }
        try {
            return localStorage.getItem('ar_realm_control_mode') || 'desktop';
        } catch (e) {
            return 'desktop';
        }
    }

    function bootSync() {
        syncRealmBodyMode(resolveMode());
    }

    document.addEventListener(
        'click',
        function (e) {
            var btn = e.target && e.target.closest ? e.target.closest('.mode-btn') : null;
            if (!btn || !btn.dataset || !btn.dataset.mode) return;
            console.warn('[REALM MODE BTN]', {
                dataMode: btn.dataset.mode,
                label: (btn.textContent || '').trim(),
            });
        },
        true
    );

    window.addEventListener('realm-action', function (e) {
        var d = e.detail;
        if (d && d.action === 'control-mode-changed' && d.mode) {
            syncRealmBodyMode(d.mode);
        }
    });

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', bootSync);
    } else {
        bootSync();
    }
})();
