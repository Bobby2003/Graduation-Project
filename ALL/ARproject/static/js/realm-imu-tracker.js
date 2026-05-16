/**
 * AR/VR 沉浸位面：进入 gesture/vr 即拉起 Center 跟踪并轮询 get_pose_latest()，
 * 用 IMU（优先）/相机位姿驱动 Three 相机（无需先点「开始扫描」）。
 * 页面临时显示 IMU 连接与倾斜/移动调试信息（Center_pipeline.get_pose_latest）。
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

    var debugPanel = null;
    var debugBody = null;
    var lastPollAt = 0;
    var lastHttpOk = false;
    var lastPayload = null;
    var lastNetworkError = '';

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

    function ensureDebugPanel() {
        if (debugPanel) return;
        var host = document.querySelector('.realm-viewport') || document.body;
        debugPanel = document.createElement('div');
        debugPanel.id = 'realm-imu-debug-panel';
        debugPanel.className = 'realm-immersive-keep';
        debugPanel.setAttribute('aria-live', 'polite');
        debugPanel.hidden = true;

        var title = document.createElement('div');
        title.className = 'realm-imu-debug-title';
        title.textContent = 'IMU / 设备位姿 · 调试（临时）';

        debugBody = document.createElement('pre');
        debugBody.id = 'realm-imu-debug-body';
        debugBody.className = 'realm-imu-debug-body';
        debugBody.textContent = '等待 AR/VR 模式…';

        debugPanel.appendChild(title);
        debugPanel.appendChild(debugBody);
        host.appendChild(debugPanel);
    }

    function fmtNum(n, digits) {
        if (n == null || !Number.isFinite(n)) return '—';
        return n.toFixed(digits);
    }

    function translationFromRows(rows) {
        if (!rows || rows.length < 4) return null;
        return {
            x: rows[0][3],
            y: rows[1][3],
            z: rows[2][3],
        };
    }

    function eulerDegFromRows(rows) {
        if (!rows || typeof THREE === 'undefined') return null;
        var m = mat4FromRows(rows);
        if (!m) return null;
        var e = new THREE.Euler().setFromRotationMatrix(m, 'YXZ');
        return {
            pitch: THREE.MathUtils.radToDeg(e.x),
            yaw: THREE.MathUtils.radToDeg(e.y),
            roll: THREE.MathUtils.radToDeg(e.z),
        };
    }

    function deltaFromBaseline(currentRows) {
        if (!baselinePose || !currentRows || typeof THREE === 'undefined') return null;
        var cur = mat4FromRows(currentRows);
        if (!cur) return null;
        var delta = new THREE.Matrix4().copy(baselinePose).invert().multiply(cur);
        var pos = new THREE.Vector3();
        var quat = new THREE.Quaternion();
        var scl = new THREE.Vector3();
        delta.decompose(pos, quat, scl);
        var e = new THREE.Euler().setFromQuaternion(quat, 'YXZ');
        return {
            dx: pos.x,
            dy: pos.y,
            dz: pos.z,
            pitch: THREE.MathUtils.radToDeg(e.x),
            yaw: THREE.MathUtils.radToDeg(e.y),
            roll: THREE.MathUtils.radToDeg(e.z),
        };
    }

    function buildDebugText(data, errMsg) {
        var lines = [];
        var now = Date.now();
        var ageMs = lastPollAt ? now - lastPollAt : null;

        lines.push('【接口】 GET /api/center/pose-latest/');
        if (errMsg) {
            lines.push('  连接: ✗ ' + errMsg);
        } else if (lastHttpOk) {
            lines.push('  连接: ✓ HTTP 200' + (ageMs != null ? ' · ' + ageMs + 'ms 前' : ''));
        } else {
            lines.push('  连接: … 等待首次响应');
        }

        if (!data) {
            lines.push('【数据】 无 payload');
            return lines.join('\n');
        }

        if (data.ok === false) {
            lines.push('【数据】 ✗ ' + (data.error || 'ok=false'));
            return lines.join('\n');
        }

        var st = data.status || 'ok';
        lines.push('【管线】 status=' + st);
        if (data.imu_only) {
            lines.push('  模式: 仅 IMU（深度相机未启动）');
        }
        if (st === 'not_running') {
            lines.push('  → 正在自动启动 IMU 跟踪（AR/VR 进入即启，不启深度相机）');
        } else if (st === 'not_ready') {
            lines.push('  → 管线已启但跟踪/外参未就绪');
        } else {
            lines.push(
                '  tracking=' +
                    (data.tracking_success ? '✓ 成功' : '✗ 失败') +
                    ' · mode=' +
                    (data.tracking_mode != null ? data.tracking_mode : '—') +
                    ' · frame=' +
                    (data.frame_id != null ? data.frame_id : '—')
            );
        }

        var hasImu = !!(data.imu_to_world && data.imu_to_world.length >= 4);
        var hasCam = !!(data.camera_to_world && data.camera_to_world.length >= 4);
        lines.push('【IMU 字段】');
        lines.push('  imu_to_world: ' + (hasImu ? '✓ 有（Center 推导）' : '✗ 无'));
        lines.push('  camera_to_world: ' + (hasCam ? '✓ 有' : '✗ 无'));
        lines.push('  坐标系: ' + (data.coordinate_space || 'reconstruction_world'));

        var poseRows = hasImu ? data.imu_to_world : hasCam ? data.camera_to_world : null;
        var src = hasImu ? 'imu_to_world' : hasCam ? 'camera_to_world' : null;

        if (poseRows) {
            var t = translationFromRows(poseRows);
            var rpy = eulerDegFromRows(poseRows);
            lines.push('【当前位姿 · ' + src + '】');
            if (t) {
                lines.push(
                    '  平移(m): x=' +
                        fmtNum(t.x, 3) +
                        '  y=' +
                        fmtNum(t.y, 3) +
                        '  z=' +
                        fmtNum(t.z, 3)
                );
            }
            if (rpy) {
                lines.push(
                    '  倾斜(°): pitch=' +
                        fmtNum(rpy.pitch, 2) +
                        '  yaw=' +
                        fmtNum(rpy.yaw, 2) +
                        '  roll=' +
                        fmtNum(rpy.roll, 2)
                );
                lines.push('  （pitch≈俯仰  yaw≈偏航  roll≈横滚）');
            }

            var d = deltaFromBaseline(poseRows);
            if (d) {
                lines.push('【相对基准 · 设备移动量】');
                lines.push(
                    '  Δ平移(m): x=' +
                        fmtNum(d.dx, 3) +
                        '  y=' +
                        fmtNum(d.dy, 3) +
                        '  z=' +
                        fmtNum(d.dz, 3)
                );
                lines.push(
                    '  Δ倾斜(°): pitch=' +
                        fmtNum(d.pitch, 2) +
                        '  yaw=' +
                        fmtNum(d.yaw, 2) +
                        '  roll=' +
                        fmtNum(d.roll, 2)
                );
            } else if (data.tracking_success) {
                lines.push('【相对基准】 等待首帧校准…');
            }
        } else {
            lines.push('【位姿】 无 4×4 矩阵可解析');
        }

        lines.push('【AR 相机驱动】');
        lines.push('  基准已锁定: ' + (baselinePose ? '✓' : '✗'));
        lines.push('  轮询: ' + (pollTimer ? '进行中 ' + POLL_MS + 'ms' : '已停'));

        return lines.join('\n');
    }

    function refreshDebugPanel(data, errMsg) {
        ensureDebugPanel();
        if (!debugPanel || !debugBody) return;

        if (!shouldTrackImu()) {
            debugPanel.hidden = true;
            return;
        }

        debugPanel.hidden = false;
        debugBody.textContent = buildDebugText(data || lastPayload, errMsg || lastNetworkError);
    }

    function mat4FromRows(rows) {
        if (!rows || rows.length < 4 || typeof THREE === 'undefined') return null;
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
        lastPayload = null;
        refreshDebugPanel(null, lastNetworkError);
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
            refreshDebugPanel(lastPayload, '');
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
        lastPayload = data;
        lastPollAt = Date.now();
        lastHttpOk = true;
        lastNetworkError = '';
        refreshDebugPanel(data, '');

        if (!data || data.ok === false) {
            if (data && data.error) {
                setStatusHint('IMU: ' + String(data.error).slice(0, 80));
            }
            return;
        }

        if (data.status === 'not_running') {
            requestImuPipeline();
            setStatusHint('IMU: 正在启动跟踪…');
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
        refreshDebugPanel(data, '');
    }

    function pollPose() {
        if (!shouldTrackImu()) {
            refreshDebugPanel(null, '');
            return;
        }
        if (pollInFlight) return;

        ensureDebugPanel();
        refreshDebugPanel(lastPayload, lastNetworkError);

        if (typeof THREE === 'undefined' || !window.camera) {
            refreshDebugPanel(
                lastPayload,
                'Three.js 或 camera 未就绪（请稍候 realm-main 加载）'
            );
            return;
        }

        pollInFlight = true;
        fetch(apiUrl('centerPoseLatest', '/api/center/pose-latest/'), { credentials: 'same-origin' })
            .then(function (res) {
                lastHttpOk = res.ok;
                if (!res.ok) {
                    throw new Error('HTTP ' + res.status);
                }
                return res.json();
            })
            .then(handlePosePayload)
            .catch(function (err) {
                lastNetworkError =
                    err && err.message ? err.message : '无法连接 pose-latest';
                setStatusHint('IMU: ' + lastNetworkError);
                refreshDebugPanel(lastPayload, lastNetworkError);
            })
            .finally(function () {
                pollInFlight = false;
            });
    }

    function startPolling() {
        if (pollTimer) return;
        ensureDebugPanel();
        pollPose();
        pollTimer = setInterval(pollPose, POLL_MS);
    }

    function stopPolling() {
        if (!pollTimer) return;
        clearInterval(pollTimer);
        pollTimer = null;
    }

    function requestImuPipeline() {
        if (
            window.CenterRealtimeMesh &&
            typeof window.CenterRealtimeMesh.ensurePipelineForImu === 'function'
        ) {
            window.CenterRealtimeMesh.ensurePipelineForImu().catch(function () {});
        }
    }

    function syncTrackingState() {
        ensureDebugPanel();
        if (shouldTrackImu()) {
            requestImuPipeline();
            if (
                window.CenterRealtimeMesh &&
                typeof window.CenterRealtimeMesh.syncForControlMode === 'function'
            ) {
                window.CenterRealtimeMesh.syncForControlMode();
            }
            startPolling();
        } else {
            stopPolling();
            resetTracking();
            if (debugPanel) debugPanel.hidden = true;
            if (
                window.CenterRealtimeMesh &&
                typeof window.CenterRealtimeMesh.stopImuOnly === 'function'
            ) {
                window.CenterRealtimeMesh.stopImuOnly().catch(function () {});
            }
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
        getLastPayload: function () {
            return lastPayload;
        },
    };
})();
