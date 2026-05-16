/**
 * AR/VR 沉浸位面：轮询 Center get_pose_latest()，用 IMU（优先）/相机位姿驱动 Three 相机。
 * 模型保持在场景内，通过相对位姿 delta 实现「可移动设备」漫游（对齐 Center_API 说明）。
 */
(function () {
    'use strict';

    var POLL_MS = 80;
    var POSITION_GAIN = 1.0;
    var SMOOTH = 0.35;

    var pollTimer = null;
    var pollInFlight = false;
    var baselinePose = null;
    var refCameraMatrix = null;
    var lastStatus = '';
    var smoothedPos = null;
    var smoothedQuat = null;

    var urls = (window.REALM_BOOTSTRAP && window.REALM_BOOTSTRAP.urls) || {};

    function apiUrl(name, fallback) {
        return urls[name] || fallback;
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

    function isImmersive3DPage() {
        if (typeof window.ARRealmIsImmersive3DPage === 'function') {
            return window.ARRealmIsImmersive3DPage();
        }
        return !!(
            document.getElementById('realm-canvas-root') ||
            (document.body && document.body.classList.contains('ar-immersive-3d'))
        );
    }

    function shouldTrackImu() {
        if (!isImmersive3DPage()) return false;
        var m = getControlMode();
        return m === 'gesture' || m === 'vr';
    }

    function mat4FromRows(rows) {
        if (!rows || rows.length < 4) return null;
        var m = new THREE.Matrix4();
        m.set(
            rows[0][0],
            rows[0][1],
            rows[0][2],
            rows[0][3],
            rows[1][0],
            rows[1][1],
            rows[1][2],
            rows[1][3],
            rows[2][0],
            rows[2][1],
            rows[2][2],
            rows[2][3],
            rows[3][0],
            rows[3][1],
            rows[3][2],
            rows[3][3]
        );
        return m;
    }

    function pickPoseMatrix(data) {
        if (!data) return null;
        if (data.imu_to_world) return mat4FromRows(data.imu_to_world);
        if (data.camera_to_world) return mat4FromRows(data.camera_to_world);
        return null;
    }

    function resetTracking() {
        baselinePose = null;
        refCameraMatrix = null;
        smoothedPos = null;
        smoothedQuat = null;
        lastStatus = '';
    }

    function setStatusHint(text) {
        var el =
            document.getElementById('realm-ar-gesture-status') ||
            document.getElementById('gesture-status-text');
        if (!el || !shouldTrackImu()) return;
        if (text === lastStatus) return;
        lastStatus = text;
        el.textContent = text;
        el.style.color = '#7ee8ff';
    }

    function lerpVec3(out, a, b, t) {
        out.x = a.x + (b.x - a.x) * t;
        out.y = a.y + (b.y - a.y) * t;
        out.z = a.z + (b.z - a.z) * t;
    }

    function applyRelativePose(currentPose) {
        var cam = window.camera;
        if (!cam || !currentPose) return;

        if (!baselinePose || !refCameraMatrix) {
            baselinePose = currentPose.clone();
            cam.updateMatrixWorld(true);
            refCameraMatrix = cam.matrixWorld.clone();
            smoothedPos = cam.position.clone();
            smoothedQuat = cam.quaternion.clone();
            setStatusHint('IMU 已校准 · 移动设备/相机可漫游场景');
            return;
        }

        var delta = new THREE.Matrix4().copy(baselinePose).invert().multiply(currentPose);
        var pos = new THREE.Vector3();
        var quat = new THREE.Quaternion();
        var scl = new THREE.Vector3();
        delta.decompose(pos, quat, scl);
        pos.multiplyScalar(POSITION_GAIN);

        var deltaPure = new THREE.Matrix4().compose(pos, quat, new THREE.Vector3(1, 1, 1));
        var target = new THREE.Matrix4().copy(refCameraMatrix).multiply(deltaPure);

        var targetPos = new THREE.Vector3();
        var targetQuat = new THREE.Quaternion();
        var targetScl = new THREE.Vector3();
        target.decompose(targetPos, targetQuat, targetScl);

        if (!smoothedPos) smoothedPos = targetPos.clone();
        if (!smoothedQuat) smoothedQuat = targetQuat.clone();

        lerpVec3(smoothedPos, smoothedPos, targetPos, SMOOTH);
        smoothedQuat.slerp(targetQuat, SMOOTH);

        cam.position.copy(smoothedPos);
        cam.quaternion.copy(smoothedQuat);
        cam.updateMatrixWorld(true);
    }

    function handlePosePayload(data) {
        if (!data || data.ok === false) {
            if (data && data.error) {
                setStatusHint('IMU: ' + String(data.error).slice(0, 80));
            }
            return;
        }

        if (data.status === 'not_running') {
            setStatusHint('IMU: 请先开始扫描 (Center START)');
            return;
        }
        if (data.status === 'not_ready' || !data.tracking_success) {
            setStatusHint('IMU: 等待跟踪…');
            return;
        }

        var pose = pickPoseMatrix(data);
        if (!pose) {
            setStatusHint('IMU: 无位姿数据');
            return;
        }

        applyRelativePose(pose);
    }

    function pollPose() {
        if (!shouldTrackImu()) return;
        if (pollInFlight) return;
        if (typeof THREE === 'undefined' || !window.camera) return;

        pollInFlight = true;
        fetch(apiUrl('centerPoseLatest', '/api/center/pose-latest/'), { credentials: 'same-origin' })
            .then(function (res) {
                return res.json();
            })
            .then(handlePosePayload)
            .catch(function () {
                setStatusHint('IMU: 无法连接 pose-latest');
            })
            .finally(function () {
                pollInFlight = false;
            });
    }

    function startPolling() {
        if (pollTimer) return;
        pollPose();
        pollTimer = setInterval(pollPose, POLL_MS);
    }

    function stopPolling() {
        if (!pollTimer) return;
        clearInterval(pollTimer);
        pollTimer = null;
    }

    function syncTrackingState() {
        if (shouldTrackImu()) {
            startPolling();
        } else {
            stopPolling();
            resetTracking();
        }
    }

    window.addEventListener('realm-action', function (e) {
        var d = e.detail;
        if (!d) return;
        if (d.action === 'control-mode-changed') {
            resetTracking();
            syncTrackingState();
        }
        if (d.action === 'scan-start' || d.action === 'scan-stop') {
            resetTracking();
        }
    });

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', syncTrackingState);
    } else {
        syncTrackingState();
    }

    window.RealmImuTracker = {
        start: startPolling,
        stop: stopPolling,
        reset: resetTracking,
        isActive: function () {
            return !!pollTimer;
        },
    };
})();
