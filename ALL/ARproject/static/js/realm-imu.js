/**
 * AR/VR 沉浸模式：用 Center_pipeline 官方接口驱动虚拟相机
 *
 * 浏览器侧只调已有 Django 路由（内部对应 Center_pipeline Python API）：
 * - GET /api/center/status/  → get_pipeline_status() + get_pose_latest()
 *
 * 见 Center_API说明.md：模型保持在 reconstruction_world，通过 pose 更新相机，勿把模型绑在相机前。
 */
(function () {
    'use strict';

    var POSE_POLL_MS = 100;
    /** reconstruction_world（米）→ 位面场景单位 */
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

    /** Center_pipeline.get_pose_latest() → camera_to_world 4×4（行主序嵌套数组） */
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

    /**
     * 解析 get_pipeline_status() 响应中的 pose_latest（Center_pipeline.get_pose_latest）
     */
    function applyPoseFromCenterStatus(data) {
        if (!data || data.ok === false) {
            lastCenterTracking = false;
            return;
        }

        var pose = data.pose_latest;
        if (!pose || pose.status === 'not_running' || pose.status === 'not_ready') {
            lastCenterTracking = false;
            return;
        }

        if (!pose.tracking_success || !pose.camera_to_world) {
            lastCenterTracking = false;
            return;
        }

        if (pose.coordinate_space && pose.coordinate_space !== 'reconstruction_world') {
            console.warn('[RealmImu] unexpected coordinate_space', pose.coordinate_space);
        }

        lastCenterTracking = true;
        var m = matrix4FromNested(pose.camera_to_world);
        if (m) applyCenterPose(m);
    }

    /** GET /api/center/status/ → Center_pipeline.get_pipeline_status + get_pose_latest */
    function pollCenterStatus() {
        if (!isArImmersiveMode() || pollInFlight) return;
        pollInFlight = true;
        fetch(apiUrl('centerStatus', '/api/center/status/'), { credentials: 'same-origin' })
            .then(function (res) {
                return res.json();
            })
            .then(function (data) {
                applyPoseFromCenterStatus(data);
            })
            .catch(function () {
                lastCenterTracking = false;
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
        pollCenterStatus();
        pollTimer = setInterval(pollCenterStatus, POSE_POLL_MS);
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
    };

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function () {
            if (isArImmersiveMode()) startImu();
        });
    } else if (isArImmersiveMode()) {
        startImu();
    }
})();
