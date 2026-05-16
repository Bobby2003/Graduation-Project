/**
 * VR / AR 立体分屏（Side-by-Side）· 左右眼视差
 */
(function () {
    'use strict';

    var EYE_SEPARATION = 0.065;
    var stereoCamL = null;
    var stereoCamR = null;
    var _offVec = null;
    var _bufSize = null;

    /** Three r160 使用 getDrawingBufferSize，无 getDrawingBufferWidth */
    function getBufferSize(renderer) {
        if (!renderer) return { w: 0, h: 0 };
        if (typeof renderer.getDrawingBufferSize === 'function' && typeof THREE !== 'undefined') {
            if (!_bufSize) _bufSize = new THREE.Vector2();
            renderer.getDrawingBufferSize(_bufSize);
            return { w: _bufSize.x, h: _bufSize.y };
        }
        var el = renderer.domElement;
        return { w: el ? el.width : 0, h: el ? el.height : 0 };
    }

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

    function isMobileViewport() {
        var coarse =
            typeof window.matchMedia === 'function' &&
            window.matchMedia('(pointer: coarse)').matches;
        var narrow = Math.min(window.innerWidth, window.innerHeight) < 720;
        var ua = /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent || '');
        return (coarse && narrow) || (ua && narrow);
    }

    function ensureStereoCameras(baseCamera) {
        if (typeof THREE === 'undefined' || !baseCamera) return false;
        if (!stereoCamL) {
            stereoCamL = new THREE.PerspectiveCamera();
            stereoCamR = new THREE.PerspectiveCamera();
            _offVec = new THREE.Vector3();
        }
        return true;
    }

    function syncStereoCameras(camera, renderer) {
        if (!ensureStereoCameras(camera) || !renderer) return false;

        var buf = getBufferSize(renderer);
        var w = buf.w;
        var h = buf.h;
        if (w < 4 || h < 4) return false;

        var halfAspect = (w * 0.5) / h;

        [stereoCamL, stereoCamR].forEach(function (cam) {
            cam.fov = camera.fov;
            cam.near = camera.near;
            cam.far = camera.far;
            cam.aspect = halfAspect;
            cam.updateProjectionMatrix();
        });

        _offVec.set(EYE_SEPARATION * 0.5, 0, 0);
        _offVec.applyQuaternion(camera.quaternion);

        stereoCamL.position.copy(camera.position).sub(_offVec);
        stereoCamR.position.copy(camera.position).add(_offVec);
        stereoCamL.quaternion.copy(camera.quaternion);
        stereoCamR.quaternion.copy(camera.quaternion);
        stereoCamL.updateMatrixWorld(true);
        stereoCamR.updateMatrixWorld(true);
        return true;
    }

    function renderStereo(renderer, scene, camera) {
        if (!isStereoMode() || !renderer || !scene || !camera) {
            return false;
        }
        if (!syncStereoCameras(camera, renderer)) {
            return false;
        }

        var buf2 = getBufferSize(renderer);
        var w = buf2.w;
        var h = buf2.h;
        var halfW = Math.floor(w / 2);
        if (halfW < 2) return false;

        var prevAutoClear = renderer.autoClear;
        renderer.autoClear = false;
        renderer.setScissorTest(false);
        renderer.setViewport(0, 0, w, h);
        renderer.clear(true, true, true);

        renderer.setScissorTest(true);

        renderer.setViewport(0, 0, halfW, h);
        renderer.setScissor(0, 0, halfW, h);
        renderer.render(scene, stereoCamL);

        renderer.setViewport(halfW, 0, w - halfW, h);
        renderer.setScissor(halfW, 0, w - halfW, h);
        renderer.render(scene, stereoCamR);

        renderer.setScissorTest(false);
        renderer.setViewport(0, 0, w, h);
        renderer.autoClear = prevAutoClear;
        return true;
    }

    function unlockOrientation() {
        try {
            if (screen.orientation && screen.orientation.unlock) {
                screen.orientation.unlock();
            }
        } catch (e) {}
    }

    function lockLandscape() {
        if (!isMobileViewport()) return;
        try {
            if (screen.orientation && screen.orientation.lock) {
                screen.orientation.lock('landscape').catch(function () {});
            }
        } catch (e) {}
    }

    function applyStereoLayout(mode) {
        mode = mode || getControlMode();
        var active = isStereoMode(mode);
        var mobile = active && isMobileViewport();

        document.body.classList.toggle('realm-stereo-active', active);
        document.body.classList.toggle('realm-stereo-mobile', mobile);

        if (active) {
            lockLandscape();
        } else {
            unlockOrientation();
        }
    }

    window.addEventListener('realm-action', function (e) {
        var d = e.detail;
        if (d && d.action === 'control-mode-changed') {
            applyStereoLayout(d.mode);
        }
    });

    window.addEventListener('resize', function () {
        if (isStereoMode()) {
            document.body.classList.toggle('realm-stereo-mobile', isMobileViewport());
        }
    });

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function () {
            applyStereoLayout();
        });
    } else {
        applyStereoLayout();
    }

    window.RealmStereo = {
        isStereoMode: isStereoMode,
        isMobileViewport: isMobileViewport,
        renderStereo: renderStereo,
        applyStereoLayout: applyStereoLayout,
        getBufferSize: getBufferSize,
        EYE_SEPARATION: EYE_SEPARATION,
    };
})();
