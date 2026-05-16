/**
 * 门户页（非 3D 位面）AR/VR 立体分屏 · 左右眼 SBS + 视差
 * 3D 位面由 realm-stereo.js + realm-main 处理
 */
(function () {
    'use strict';

    var host = null;
    var sourceEl = null;
    var leftPane = null;
    var rightPane = null;
    var onSourceScroll = null;

    function getControlMode() {
        if (window.ARRealmControls && typeof window.ARRealmControls.getMode === 'function') {
            return window.ARRealmControls.getMode();
        }
        try {
            return localStorage.getItem('ar_realm_control_mode') || 'desktop';
        } catch (e) {
            return 'desktop';
        }
    }

    function isStereoMode(mode) {
        mode = mode || getControlMode();
        return mode === 'gesture' || mode === 'vr';
    }

    function isHomePage() {
        if (window.ARRealmControls && typeof window.ARRealmControls.isHomePage === 'function') {
            return window.ARRealmControls.isHomePage();
        }
        if (document.body && document.body.dataset.arHomePage === '1') {
            return true;
        }
        var p = (window.location.pathname || '/').replace(/\/+$/, '') || '/';
        return p === '/' || p === '';
    }

    function isImmersive3DPage() {
        if (typeof window.ARRealmIsImmersive3DPage === 'function') {
            return window.ARRealmIsImmersive3DPage();
        }
        return !!(
            document.getElementById('realm-canvas-root') ||
            (document.body && document.body.classList.contains('ar-immersive-3d'))
        );
    }

    function findContentRoot() {
        if (isImmersive3DPage()) return null;
        var selectors = [
            'main',
            '.settings-container',
            '.app-shell',
            '.about-container',
            '.doc-main',
            '.gateway-box',
        ];
        for (var i = 0; i < selectors.length; i++) {
            var el = document.querySelector(selectors[i]);
            if (el) return el;
        }
        return null;
    }

    function stripDuplicateIds(node) {
        if (!node || node.nodeType !== 1) return;
        if (node.id) node.removeAttribute('id');
        var ch = node.children;
        for (var i = 0; i < ch.length; i++) {
            stripDuplicateIds(ch[i]);
        }
    }

    function createPane(side) {
        var pane = document.createElement('div');
        pane.className = 'ar-portal-stereo-pane ar-portal-stereo-pane--' + side;
        pane.setAttribute('aria-hidden', 'true');

        var inner = document.createElement('div');
        inner.className = 'ar-portal-stereo-inner ar-portal-stereo-parallax--' + side;
        pane.appendChild(inner);
        return pane;
    }

    function syncScrollFromSource() {
        if (!sourceEl || !leftPane || !rightPane) return;
        var st = sourceEl.scrollTop;
        var sl = sourceEl.scrollLeft;
        leftPane.scrollTop = st;
        leftPane.scrollLeft = sl;
        rightPane.scrollTop = st;
        rightPane.scrollLeft = sl;
    }

    function unmountPortalStereo() {
        if (onSourceScroll && sourceEl) {
            sourceEl.removeEventListener('scroll', onSourceScroll);
        }
        onSourceScroll = null;

        if (host && host.parentNode) {
            host.parentNode.removeChild(host);
        }
        host = null;
        leftPane = null;
        rightPane = null;

        if (sourceEl) {
            sourceEl.classList.remove('ar-portal-stereo-source-active');
            sourceEl = null;
        }
    }

    function mountPortalStereo() {
        unmountPortalStereo();

        if (!isStereoMode() || isImmersive3DPage()) {
            return;
        }

        var source = findContentRoot();
        if (!source) return;

        sourceEl = source;

        host = document.createElement('div');
        host.id = 'ar-portal-stereo-host';
        host.className = 'ar-portal-stereo-host';
        host.setAttribute('role', 'presentation');

        leftPane = createPane('left');
        rightPane = createPane('right');

        var cloneL = source.cloneNode(true);
        var cloneR = source.cloneNode(true);
        stripDuplicateIds(cloneL);
        stripDuplicateIds(cloneR);

        leftPane.querySelector('.ar-portal-stereo-inner').appendChild(cloneL);
        rightPane.querySelector('.ar-portal-stereo-inner').appendChild(cloneR);

        host.appendChild(leftPane);
        host.appendChild(rightPane);
        document.body.appendChild(host);

        sourceEl.classList.add('ar-portal-stereo-source-active');

        onSourceScroll = function () {
            syncScrollFromSource();
        };
        sourceEl.addEventListener('scroll', onSourceScroll, { passive: true });
        syncScrollFromSource();
    }

    function applyPortalStereo(mode) {
        mode = mode || getControlMode();
        if (isHomePage() || !isStereoMode(mode) || isImmersive3DPage()) {
            unmountPortalStereo();
            return;
        }
        mountPortalStereo();
    }

    window.addEventListener('realm-action', function (e) {
        var d = e.detail;
        if (d && d.action === 'control-mode-changed') {
            applyPortalStereo(d.mode);
        }
    });

    function boot() {
        applyPortalStereo();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', boot);
    } else {
        boot();
    }

    window.PortalStereo = {
        applyPortalStereo: applyPortalStereo,
        mountPortalStereo: mountPortalStereo,
        unmountPortalStereo: unmountPortalStereo,
    };
})();
