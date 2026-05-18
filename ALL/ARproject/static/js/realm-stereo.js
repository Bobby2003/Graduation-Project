/**
 * VR / AR 立体分屏（Side-by-Side）· 左右眼视差
 */
(function () {
    'use strict';

    /** 左右眼间距（米）。偏弱 0.04~0.05；默认 0.048；偏强 0.064~0.08 */
    var EYE_SEPARATION = 0.048;
    /** 零视差平面距离（米）。越大视差越弱、越易融合；过小会显得左右眼「分得太开」 */
    var STEREO_CONVERGENCE = 5.5;
    var stereoCamL = null;
    var stereoCamR = null;
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
        }
        return true;
    }

    /**
     * 平行立体：相机同位姿 + 离轴投影（非对称视锥）。
     * 勿同时平移相机又用对称 updateProjectionMatrix，否则视差被放大、难以融合。
     */
    function applyOffAxisStereoFrustum(stereoCam, mainCam, eyeSign, eyeSeparation) {
        var near = mainCam.near;
        var far = mainCam.far;
        var fovRad = THREE.MathUtils.degToRad(mainCam.fov);
        var top = near * Math.tan(fovRad * 0.5);
        var bottom = -top;
        var halfW = top * stereoCam.aspect;
        var left = -halfW;
        var right = halfW;
        var conv = Math.max(near * 2, STEREO_CONVERGENCE);
        var shift = eyeSign * (eyeSeparation * 0.5) * near / conv;
        left += shift;
        right += shift;
        stereoCam.projectionMatrix.makePerspective(left, right, top, bottom, near, far);
        if (stereoCam.projectionMatrixInverse) {
            stereoCam.projectionMatrixInverse.copy(stereoCam.projectionMatrix).invert();
        }
    }

    function syncStereoCameras(camera, renderer) {
        if (!ensureStereoCameras(camera) || !renderer) return false;

        var buf = getBufferSize(renderer);
        var w = buf.w;
        var h = buf.h;
        if (w < 4 || h < 4) return false;

        var halfAspect = (w * 0.5) / h;

        stereoCamL.fov = camera.fov;
        stereoCamR.fov = camera.fov;
        stereoCamL.near = camera.near;
        stereoCamR.near = camera.near;
        stereoCamL.far = camera.far;
        stereoCamR.far = camera.far;
        stereoCamL.aspect = halfAspect;
        stereoCamR.aspect = halfAspect;

        stereoCamL.position.copy(camera.position);
        stereoCamR.position.copy(camera.position);
        stereoCamL.quaternion.copy(camera.quaternion);
        stereoCamR.quaternion.copy(camera.quaternion);

        applyOffAxisStereoFrustum(stereoCamL, camera, -1, EYE_SEPARATION);
        applyOffAxisStereoFrustum(stereoCamR, camera, 1, EYE_SEPARATION);

        stereoCamL.updateMatrixWorld(true);
        stereoCamR.updateMatrixWorld(true);
        return true;
    }

    function cameraPoseIsFinite(camera) {
        if (!camera) return false;
        var p = camera.position;
        var q = camera.quaternion;
        return (
            Number.isFinite(p.x) &&
            Number.isFinite(p.y) &&
            Number.isFinite(p.z) &&
            Number.isFinite(q.x) &&
            Number.isFinite(q.y) &&
            Number.isFinite(q.z) &&
            Number.isFinite(q.w)
        );
    }

    /** 每帧结束恢复 WebGL 视口/裁剪，避免 AR/VR 切模式后单眼路径黑屏 */
    function resetRendererFrame(renderer) {
        if (!renderer) return;
        var buf = getBufferSize(renderer);
        renderer.setScissorTest(false);
        if (buf.w > 0 && buf.h > 0) {
            renderer.setViewport(0, 0, buf.w, buf.h);
        }
        renderer.autoClear = true;
    }

    function renderStereo(renderer, scene, camera) {
        if (!isStereoMode() || !renderer || !scene || !camera) {
            return false;
        }
        if (!cameraPoseIsFinite(camera)) {
            return false;
        }
        if (!syncStereoCameras(camera, renderer)) {
            return false;
        }

        var buf2 = getBufferSize(renderer);
        var w = Math.floor(buf2.w);
        var h = Math.floor(buf2.h);
        if (w < 8 || h < 8) {
            return false;
        }

        var halfW = Math.floor(w / 2);
        if (halfW < 4) {
            return false;
        }

        if (scene.background && scene.background.isColor) {
            renderer.setClearColor(scene.background, 1);
        }

        var prevAutoClear = renderer.autoClear;
        renderer.autoClear = true;
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
            requestAnimationFrame(function () {
                requestAnimationFrame(function () {
                    window.dispatchEvent(new Event('resize'));
                });
            });
        } else {
            unlockOrientation();
            requestAnimationFrame(function () {
                window.dispatchEvent(new Event('resize'));
            });
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

    var CURSOR_INSET = 0.06;

    function clampEyeRel(v) {
        var n = Math.max(0, Math.min(1, Number(v) || 0));
        return CURSOR_INSET + n * (1 - 2 * CURSOR_INSET);
    }

    /** 当前 #realm-stereo-frame 在屏幕上的左右半屏矩形（与 67% 画框对齐） */
    function getStereoLayout() {
        var frame = document.getElementById('realm-stereo-frame');
        if (!frame || !isStereoMode()) return null;
        var fr = frame.getBoundingClientRect();
        if (fr.width < 8 || fr.height < 8) return null;
        var halfW = fr.width * 0.5;
        return {
            frame: fr,
            left: { left: fr.left, top: fr.top, width: halfW, height: fr.height },
            right: { left: fr.left + halfW, top: fr.top, width: halfW, height: fr.height },
        };
    }

    function pointInEyeFromRel(layout, relX, relY, eye) {
        var t = clampEyeRel(relX);
        var u = clampEyeRel(relY);
        var panel = eye === 'right' ? layout.right : layout.left;
        return {
            x: panel.left + t * panel.width,
            y: panel.top + u * panel.height,
            relX: t,
            relY: u,
        };
    }

    /** 立体分屏：归一化坐标 → 左右眼屏幕像素（基于画框，非整窗 50%） */
    function mapRelToStereoScreens(relX, relY) {
        var layout = getStereoLayout();
        if (!layout) {
            var vw = window.innerWidth;
            var vh = window.innerHeight;
            var eyeW = vw * 0.5;
            var t = clampEyeRel(relX);
            var u = clampEyeRel(relY);
            return {
                left: { x: t * eyeW, y: u * vh },
                right: { x: eyeW + t * eyeW, y: u * vh },
            };
        }
        var l = pointInEyeFromRel(layout, relX, relY, 'left');
        var r = pointInEyeFromRel(layout, relX, relY, 'right');
        return { left: { x: l.x, y: l.y }, right: { x: r.x, y: r.y } };
    }

    /** 点击命中：映射到左眼半屏内容坐标 */
    function mapScreenToContentPoint(screenX, screenY) {
        var layout = getStereoLayout();
        if (layout) {
            var mid = layout.left.left + layout.left.width;
            var panel = screenX >= mid ? layout.right : layout.left;
            var relInEye = (screenX - panel.left) / panel.width;
            var relY = (screenY - panel.top) / panel.height;
            relInEye = Math.max(0, Math.min(1, relInEye));
            relY = Math.max(0, Math.min(1, relY));
            return {
                x: screenX,
                y: screenY,
                relX: relInEye,
                relY: relY,
            };
        }
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

    window.RealmStereo = {
        isStereoMode: isStereoMode,
        isMobileViewport: isMobileViewport,
        renderStereo: renderStereo,
        resetRendererFrame: resetRendererFrame,
        cameraPoseIsFinite: cameraPoseIsFinite,
        applyStereoLayout: applyStereoLayout,
        getBufferSize: getBufferSize,
        getStereoLayout: getStereoLayout,
        clampEyeRel: clampEyeRel,
        mapRelToStereoScreens: mapRelToStereoScreens,
        mapScreenToContentPoint: mapScreenToContentPoint,
        EYE_SEPARATION: EYE_SEPARATION,
        STEREO_CONVERGENCE: STEREO_CONVERGENCE,
    };
})();
