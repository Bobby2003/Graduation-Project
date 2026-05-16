/**

 * 门户页 AR/VR 立体分屏：左栏为真实 DOM（可交互），右栏为同步镜像（Edge 等兼容）

 * 3D 位面由 realm-stereo.js + realm-main 处理

 */

(function () {

    'use strict';



    var EYE_SCALE = 0.5;

    var host = null;

    var stage = null;

    var row = null;

    var leftEye = null;

    var rightEye = null;

    var scaler = null;

    var mirrorScaler = null;

    var sourceEl = null;

    var sourceParent = null;

    var sourceNext = null;

    var mirrorObserver = null;

    var mirrorRefreshTimer = null;



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



    function disconnectMirrorObserver() {

        if (mirrorObserver) {

            mirrorObserver.disconnect();

            mirrorObserver = null;

        }

        if (mirrorRefreshTimer) {

            clearTimeout(mirrorRefreshTimer);

            mirrorRefreshTimer = null;

        }

    }



    function scheduleMirrorRefresh() {

        if (!sourceEl || !mirrorScaler) return;

        if (mirrorRefreshTimer) clearTimeout(mirrorRefreshTimer);

        mirrorRefreshTimer = setTimeout(refreshMirror, 120);

    }



    function refreshMirror() {

        mirrorRefreshTimer = null;

        if (!sourceEl || !mirrorScaler) return;

        var clone = sourceEl.cloneNode(true);

        stripDuplicateIds(clone);

        mirrorScaler.replaceChildren(clone);

    }



    function connectMirrorObserver() {

        disconnectMirrorObserver();

        if (!sourceEl || typeof MutationObserver === 'undefined') return;

        mirrorObserver = new MutationObserver(scheduleMirrorRefresh);

        mirrorObserver.observe(sourceEl, {

            childList: true,

            subtree: true,

            characterData: true,

            attributes: true,

        });

    }



    function restoreSource() {

        disconnectMirrorObserver();

        if (!sourceEl || !sourceParent) return;

        sourceEl.classList.remove('ar-portal-stereo-source-active');

        if (sourceNext && sourceNext.parentNode === sourceParent) {

            sourceParent.insertBefore(sourceEl, sourceNext);

        } else {

            sourceParent.appendChild(sourceEl);

        }

        sourceEl = null;

        sourceParent = null;

        sourceNext = null;

    }



    function unmountPortalStereo() {

        restoreSource();

        if (host && host.parentNode) {

            host.parentNode.removeChild(host);

        }

        host = null;

        stage = null;

        row = null;

        leftEye = null;

        rightEye = null;

        scaler = null;

        mirrorScaler = null;

    }



    function mountPortalStereo() {

        if (!isStereoMode() || isImmersive3DPage()) {

            unmountPortalStereo();

            return;

        }



        var source = findContentRoot();

        if (!source) return;



        if (host && sourceEl === source) {

            return;

        }



        unmountPortalStereo();



        sourceParent = source.parentNode;

        sourceNext = source.nextSibling;

        sourceEl = source;



        host = document.createElement('div');

        host.id = 'ar-portal-stereo-host';

        host.className = 'ar-portal-stereo-host';

        host.setAttribute('role', 'presentation');



        stage = document.createElement('div');

        stage.className = 'ar-portal-stereo-stage';



        row = document.createElement('div');

        row.className = 'ar-portal-stereo-row';



        leftEye = document.createElement('div');

        leftEye.className = 'ar-portal-stereo-eye ar-portal-stereo-eye--left';



        rightEye = document.createElement('div');

        rightEye.className = 'ar-portal-stereo-eye ar-portal-stereo-eye--right';



        scaler = document.createElement('div');

        scaler.className = 'ar-portal-stereo-scaler';

        scaler.style.setProperty('--portal-eye-scale', String(EYE_SCALE));



        mirrorScaler = document.createElement('div');

        mirrorScaler.className = 'ar-portal-stereo-scaler ar-portal-stereo-scaler--mirror';

        mirrorScaler.setAttribute('aria-hidden', 'true');

        mirrorScaler.style.setProperty('--portal-eye-scale', String(EYE_SCALE));



        scaler.appendChild(source);

        leftEye.appendChild(scaler);

        rightEye.appendChild(mirrorScaler);



        row.appendChild(leftEye);

        row.appendChild(rightEye);

        stage.appendChild(row);

        host.appendChild(stage);

        document.body.appendChild(host);



        refreshMirror();

        connectMirrorObserver();

    }



    function applyPortalStereo(mode) {

        mode = mode || getControlMode();

        if (isHomePage() || !isStereoMode(mode) || isImmersive3DPage()) {

            unmountPortalStereo();

            return;

        }

        mountPortalStereo();

    }



    function isActive() {

        return !!host && !!scaler && !!mirrorScaler;

    }



    function mapScreenToContentPoint(screenX, screenY) {

        var vw = window.innerWidth;

        var vh = window.innerHeight;

        var eyeW = vw * 0.5;

        var relInEye = screenX >= eyeW ? (screenX - eyeW) / eyeW : screenX / eyeW;

        relInEye = Math.max(0, Math.min(1, relInEye));

        return {

            x: relInEye * eyeW,

            y: screenY,

            relX: relInEye,

            relY: screenY / vh,

        };

    }



    function mapRelToStereoScreens(relX, relY) {

        var vw = window.innerWidth;

        var vh = window.innerHeight;

        var eyeW = vw * 0.5;

        var y = relY * vh;

        return {

            left: { x: relX * eyeW, y: y },

            right: { x: eyeW + relX * eyeW, y: y },

        };

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

        refreshMirror: refreshMirror,

        isActive: isActive,

        getEyeScale: function () {

            return EYE_SCALE;

        },

        mapScreenToContentPoint: mapScreenToContentPoint,

        mapRelToStereoScreens: mapRelToStereoScreens,

    };

})();


