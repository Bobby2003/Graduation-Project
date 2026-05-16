/**
 * VR / AR 立体分屏（Side-by-Side）· 平行相机 + 离轴投影（非 toe-in）
 *
 * 左/右眼：rotation 与主相机完全一致，仅沿局部 X 平移 ±IPD/2；
 * 投影矩阵按汇聚距离做水平离轴偏移，避免内八旋转带来的垂直视差。
 */
(function () {
    'use strict';

    /** 普通室内 VR 场景推荐 IPD（米），见项目立体参数说明 */
    var EYE_SEPARATION = 0.064;
    /** 汇聚距离（与位面场景单位一致，≈米）；略远可减轻近处过大视差 */
    var STEREO_CONVERGENCE = 5.0;

    var stereoCamL = null;
    var stereoCamR = null;
    var _offVec = null;
    var _bufSize = null;

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

    /**
     * 平行立体离轴投影（Paul Bourke / WebXR 常用做法，非 toe-in）
     * @param {THREE.PerspectiveCamera} stereoCam
     * @param {number} fovDeg
     * @param {number} aspect 单眼视口宽高比
     * @param {number} near
     * @param {number} far
     * @param {boolean} isLeftEye
     */
    function applyParallelOffAxisProjection(stereoCam, fovDeg, aspect, near, far, isLeftEye) {
        var fovRad = THREE.MathUtils.degToRad(fovDeg);
        var ymax = near * Math.tan(fovRad * 0.5);
        var ymin = -ymax;
        var xextent = ymax * aspect;
        var z = Math.max(STEREO_CONVERGENCE, near * 1.01);
        var shift = (EYE_SEPARATION * 0.5) * (near / z);

        var left;
        var right;
        if (isLeftEye) {
            left = -xextent - shift;
            right = xextent - shift;
        } else {
            left = -xextent + shift;
            right = xextent + shift;
        }

        stereoCam.projectionMatrix.makePerspective(left, right, ymax, ymin, near, far);
        stereoCam.projectionMatrixInverse.copy(stereoCam.projectionMatrix).invert();
    }

    function syncStereoCameras(camera, renderer) {
        if (!ensureStereoCameras(camera) || !renderer) return false;

        var buf = getBufferSize(renderer);
        var w = buf.w;
        var h = buf.h;
        if (w < 4 || h < 4) return false;

        var halfAspect = (w * 0.5) / h;
        var halfSep = EYE_SEPARATION * 0.5;

        [stereoCamL, stereoCamR].forEach(function (cam) {
            cam.fov = camera.fov;
            cam.near = camera.near;
            cam.far = camera.far;
            cam.aspect = halfAspect;
        });

        _offVec.set(halfSep, 0, 0);
        _offVec.applyQuaternion(camera.quaternion);

        stereoCamL.position.copy(camera.position).sub(_offVec);
        stereoCamR.position.copy(camera.position).add(_offVec);

        stereoCamL.quaternion.copy(camera.quaternion);
        stereoCamR.quaternion.copy(camera.quaternion);

        applyParallelOffAxisProjection(stereoCamL, camera.fov, halfAspect, camera.near, camera.far, true);
        applyParallelOffAxisProjection(stereoCamR, camera.fov, halfAspect, camera.near, camera.far, false);

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

    function setStereoProfile(profile) {
        if (profile === 'comfort') {
            EYE_SEPARATION = 0.05;
            STEREO_CONVERGENCE = 5.5;
        } else if (profile === 'strong') {
            EYE_SEPARATION = 0.072;
            STEREO_CONVERGENCE = 4.5;
        } else {
            EYE_SEPARATION = 0.064;
            STEREO_CONVERGENCE = 5.0;
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

    setStereoProfile('normal');

    window.RealmStereo = {
        isStereoMode: isStereoMode,
        isMobileViewport: isMobileViewport,
        renderStereo: renderStereo,
        applyStereoLayout: applyStereoLayout,
        getBufferSize: getBufferSize,
        setStereoProfile: setStereoProfile,
        getEyeSeparation: function () {
            return EYE_SEPARATION;
        },
        getConvergence: function () {
            return STEREO_CONVERGENCE;
        },
        EYE_SEPARATION: EYE_SEPARATION,
    };
})();
