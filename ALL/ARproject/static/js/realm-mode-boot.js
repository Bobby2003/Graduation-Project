/**
 * Realm 模式同步兜底（在 controls.js 之后加载）
 */
(function () {
    'use strict';

    var BOOT_VERSION = '2026-05-16h';
    window.__REALM_MODE_BOOT_VERSION = BOOT_VERSION;

    function isImmersive3DPage() {
        if (typeof window.ARRealmIsImmersive3DPage === 'function') {
            return window.ARRealmIsImmersive3DPage();
        }
        return !!(
            document.getElementById('realm-canvas-root') ||
            (document.body && document.body.classList.contains('ar-immersive-3d'))
        );
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
        var mode = resolveMode();
        if (typeof window.syncRealmBodyMode === 'function') {
            window.syncRealmBodyMode(mode);
        }
        if (
            window.RealmStereo &&
            typeof window.RealmStereo.applyStereoLayout === 'function'
        ) {
            window.RealmStereo.applyStereoLayout(mode);
        }
        if (
            window.PortalStereo &&
            typeof window.PortalStereo.applyPortalStereo === 'function'
        ) {
            window.PortalStereo.applyPortalStereo(mode);
        }
    }

    document.addEventListener(
        'click',
        function (e) {
            if (!isImmersive3DPage()) return;
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
            bootSync();
        }
    });

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', bootSync);
    } else {
        bootSync();
    }
})();
