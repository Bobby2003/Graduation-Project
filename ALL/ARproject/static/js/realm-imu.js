/**
 * AR/VR 沉浸模式：设备位姿驱动视角
 * - Center get_pose_latest()：扫描相机在 reconstruction_world 中的位移（参考 Center_pipeline）
 * - DeviceOrientation：手机/平板陀螺仪兜底
 */
(function () {
    'use strict';

    var POSE_POLL_MS = 100;
    var POSE_SCENE_SCALE = 0.42;
    var DEVICE_SENSITIVITY = 0.85;

    var pollTimer = null;
    var pollInFlight = false;
    var refPoseInv = null;
    var arCameraBase = null;
    var lastCenterTracking = false;
    var deviceHandler = null;

    var urls = (window.REALM_BOOTSTRAP && window.REALM_BOOTSTRAP.urls) || {};

    function apiUrl(name, fallback) {
        return urls[name] || fallback;
    }

    function getMode() {
        if (window.ARRealmControls && window.ARRealmControls.getMode) {
            return window.ARRealmControls.getMode();
        }
        try {
            return localStorage.getItem('ar_realm_control_mode') || 'desktop';
        } catch (e) {
            return 'desktop';
        }
    }

    function isArImmersiveMode() {
        var m = getMode();
        return m === 'gesture' || m === 'vr';
    }

    function getCamera() {
        return window.camera || null;
    }

    function matrix4FromNested(rows) {
        if (!rows || rows.length < 4) return null;
        var m = new THREE.Matrix4();
        m.set(
            rows[0][0],
            rows[1][0],
            rows[2][0],
            rows[3][0],
            rows[0][1],
            rows[1][1],
            rows[2][1],
            rows[3][1],
            rows[0][2],
            rows[1][2],
            rows[2][2],
            rows[3][2],
            rows[0][3],
            rows[1][3],
            rows[2][3],
            rows[3][3]
        );
        return m;
    }

    function resetPoseAnchor() {
        refPoseInv = null;
        arCameraBase = null;
        lastCenterTracking = false;
    }

    function ensureCameraBase(cam) {
        if (!arCameraBase) {
            arCameraBase = {
                position: cam.position.clone(),
                quaternion: cam.quaternion.clone(),
            };
        }
    }

    function applyCenterPose(c2wMatrix) {
        var cam = getCamera();
        if (!cam || !c2wMatrix) return;

        if (!refPoseInv) {
            refPoseInv = c2wMatrix.clone().invert();
            ensureCameraBase(cam);
            return;
        }

        var delta = refPoseInv.clone().multiply(c2wMatrix);
        var pos = new THREE.Vector3();
        var quat = new THREE.Quaternion();
        var scl = new THREE.Vector3();
        delta.decompose(pos, quat, scl);

        pos.multiplyScalar(POSE_SCENE_SCALE);
        ensureCameraBase(cam);

        cam.position.copy(arCameraBase.position).add(pos);
        cam.quaternion.copy(arCameraBase.quaternion).multiply(quat);

        clampCameraToBounds(cam);
    }

    function clampCameraToBounds(cam) {
        var bounds = window.realmCameraMoveBounds;
        if (!bounds) return;
        cam.position.x = Math.max(bounds.xMin, Math.min(bounds.xMax, cam.position.x));
        cam.position.z = Math.max(bounds.zMin, Math.min(bounds.zMax, cam.position.z));
        cam.position.y = 1.7;
    }

    function applyDeviceOrientation(alpha, beta, gamma) {
        if (lastCenterTracking) return;
        var cam = getCamera();
        if (!cam) return;
        if (alpha == null && beta == null && gamma == null) return;

        ensureCameraBase(cam);

        var euler = new THREE.Euler(
            THREE.MathUtils.degToRad(beta || 0) * DEVICE_SENSITIVITY,
            THREE.MathUtils.degToRad(alpha || 0) * DEVICE_SENSITIVITY,
            THREE.MathUtils.degToRad(-(gamma || 0)) * DEVICE_SENSITIVITY,
            'YXZ'
        );
        var qDevice = new THREE.Quaternion().setFromEuler(euler);
        cam.quaternion.copy(arCameraBase.quaternion).multiply(qDevice);
        cam.position.copy(arCameraBase.position);
        clampCameraToBounds(cam);
    }

    function pollCenterPose() {
        if (!isArImmersiveMode() || pollInFlight) return;
        pollInFlight = true;
        fetch(apiUrl('centerLatestPose', '/api/center/latest-pose/'), { credentials: 'same-origin' })
            .then(function (res) {
                return res.json();
            })
            .then(function (data) {
                if (!data || data.ok === false) return;
                if (!data.tracking_success || !data.camera_to_world) {
                    lastCenterTracking = false;
                    return;
                }
                lastCenterTracking = true;
                var m = matrix4FromNested(data.camera_to_world);
                if (m) applyCenterPose(m);
            })
            .catch(function () {
                /* 静默 */
            })
            .finally(function () {
                pollInFlight = false;
            });
    }

    function startDeviceOrientation() {
        if (deviceHandler || typeof DeviceOrientationEvent === 'undefined') return;

        deviceHandler = function (ev) {
            if (!isArImmersiveMode()) return;
            applyDeviceOrientation(ev.alpha, ev.beta, ev.gamma);
        };
        window.addEventListener('deviceorientation', deviceHandler, true);

        if (
            typeof DeviceOrientationEvent !== 'undefined' &&
            typeof DeviceOrientationEvent.requestPermission === 'function'
        ) {
            DeviceOrientationEvent.requestPermission().catch(function () {});
        }
    }

    function stopDeviceOrientation() {
        if (deviceHandler) {
            window.removeEventListener('deviceorientation', deviceHandler, true);
            deviceHandler = null;
        }
    }

    function startImu() {
        if (pollTimer) return;
        resetPoseAnchor();
        startDeviceOrientation();
        pollCenterPose();
        pollTimer = setInterval(pollCenterPose, POSE_POLL_MS);
    }

    function stopImu() {
        if (pollTimer) {
            clearInterval(pollTimer);
            pollTimer = null;
        }
        stopDeviceOrientation();
        resetPoseAnchor();
    }

    function onModeChanged(mode) {
        if (mode === 'gesture' || mode === 'vr') {
            startImu();
        } else {
            stopImu();
        }
    }

    window.addEventListener('realm-action', function (e) {
        var d = e.detail;
        if (d && d.action === 'control-mode-changed') {
            onModeChanged(d.mode);
        }
    });

    window.RealmImu = {
        start: startImu,
        stop: stopImu,
        resetAnchor: resetPoseAnchor,
        updateMovement: function () {
            /* 由轮询驱动；保留钩子供 realm-main 调用 */
        },
    };

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function () {
            if (isArImmersiveMode()) startImu();
        });
    } else if (isArImmersiveMode()) {
        startImu();
    }
})();
