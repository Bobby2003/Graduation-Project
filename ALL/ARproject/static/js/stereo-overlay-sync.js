/**
 * AR/VR 立体分屏：同步左右眼 HTML 弹窗镜像（右眼仅显示，交互在左眼）
 */
(function (global) {
    'use strict';

    function stripMirrorInteractivity(root) {
        if (!root) return;
        root.querySelectorAll('[id]').forEach(function (node) {
            node.removeAttribute('id');
        });
        root.querySelectorAll('a, button, input, select, textarea, label').forEach(function (node) {
            node.setAttribute('tabindex', '-1');
            node.setAttribute('aria-hidden', 'true');
        });
    }

    function syncOverlayRoot(root) {
        if (!root) return;
        var src = root.querySelector('[data-overlay-content="source"]');
        var mirror = root.querySelector('[data-overlay-content="mirror"]');
        if (!src || !mirror) return;
        mirror.innerHTML = src.innerHTML;
        stripMirrorInteractivity(mirror);
    }

    function syncById(rootId) {
        syncOverlayRoot(document.getElementById(rootId));
    }

    function isPortalStereoActive() {
        return global.PortalStereo && typeof global.PortalStereo.isActive === 'function' && global.PortalStereo.isActive();
    }

    function isStereoUiMode() {
        var mode = 'desktop';
        if (global.ARRealmControls && typeof global.ARRealmControls.getMode === 'function') {
            mode = global.ARRealmControls.getMode();
        } else {
            try {
                mode = localStorage.getItem('ar_realm_control_mode') || 'desktop';
            } catch (e) { /* ignore */ }
        }
        return mode === 'gesture' || mode === 'vr';
    }

    global.ARStereoOverlaySync = {
        syncOverlayRoot: syncOverlayRoot,
        syncById: syncById,
        isPortalStereoActive: isPortalStereoActive,
        isStereoUiMode: isStereoUiMode,
    };
})(window);
