/**
 * AR/VR 沉浸位面：POST start-imu + 轮询 pose-latest。
 * 使用 Center 下发的 camera_to_world（OpenCV 相机轴，已含 R_cam_imu），
 * 转成 Three 相机系后做世界系相对旋转 Δ·ref 驱动相机；API 矩阵按行填入 Three.Matrix4。
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
    var debugPanelVisible = false;
    var lastPollAt = 0;
    var lastHttpOk = false;
    var lastPayload = null;
    var lastNetworkError = '';

    var urls = (window.REALM_BOOTSTRAP && window.REALM_BOOTSTRAP.urls) || {};

    function readEyeHeightM() {
        if (typeof window.readRealmEyeHeightM === 'function') {
            return window.readRealmEyeHeightM();
        }
        var b = window.REALM_BOOTSTRAP || {};
        var h = b.eyeHeightM;
        if (typeof h === 'number' && Number.isFinite(h)) return h;
        return 1.7;
    }

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
        return eulerDegFromMatrix(m);
    }

    function eulerDegFromMatrix(m) {
        if (!m || typeof THREE === 'undefined') return null;
        var e = new THREE.Euler().setFromRotationMatrix(m, 'YXZ');
        return {
            pitch: THREE.MathUtils.radToDeg(e.x),
            yaw: THREE.MathUtils.radToDeg(e.y),
            roll: THREE.MathUtils.radToDeg(e.z),
        };
    }

    function deltaFromBaselinePose(currentMat) {
        if (!baselinePose || !currentMat || typeof THREE === 'undefined') return null;
        var delta = new THREE.Matrix4().copy(currentMat).multiply(baselinePose.clone().invert());
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
        lines.push('【位姿字段】');
        lines.push('  camera_to_world: ' + (hasCam ? '✓（AR 驱动）' : '✗ 无'));
        lines.push('  imu_to_world: ' + (hasImu ? '✓' : '✗ 无'));
        lines.push('  坐标系: ' + (data.coordinate_space || 'reconstruction_world'));

        var poseRows = hasCam ? data.camera_to_world : hasImu ? data.imu_to_world : null;
        var src = hasCam ? 'camera_to_world' : hasImu ? 'imu_to_world' : null;

        if (poseRows) {
            var curPose = pickPoseMatrix(data);
            var t = translationFromRows(poseRows);
            var rpy = curPose ? eulerDegFromMatrix(curPose) : eulerDegFromRows(poseRows);
            lines.push('【当前位姿 · ' + src + (curPose ? ' → Three' : '') + '】');
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

            var d = deltaFromBaselinePose(curPose);
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

    function applyDebugPanelVisibility() {
        if (!debugPanel) return;
        debugPanel.hidden = !(debugPanelVisible && shouldTrackImu());
    }

    function toggleDebugPanel(forceOpen) {
        ensureDebugPanel();
        if (!shouldTrackImu()) {
            debugPanelVisible = false;
            applyDebugPanelVisibility();
            return false;
        }
        debugPanelVisible =
            typeof forceOpen === 'boolean' ? forceOpen : !debugPanelVisible;
        applyDebugPanelVisibility();
        if (debugPanelVisible && debugBody) {
            debugBody.textContent = buildDebugText(
                lastPayload,
                lastNetworkError || null
            );
        }
        return debugPanelVisible;
    }

    function refreshDebugPanel(data, errMsg) {
        ensureDebugPanel();
        if (!debugPanel || !debugBody) return;

        if (!shouldTrackImu()) {
            debugPanel.hidden = true;
            return;
        }

        applyDebugPanelVisibility();
        if (!debugPanelVisible) return;

        debugBody.textContent = buildDebugText(data || lastPayload, errMsg || lastNetworkError);
    }

    /** API 下发的 4×4 按行数组（NumPy/Python 等与 rows[i][j] 一致）；Three.Matrix4.set 按矩阵行填入 n11–n44。 */
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

    /** OpenCV 相机 (x右 y下 z前) → Three 相机 (x右 y上 z后)：绕 X 转 180° */
    var opencvCamToThree = null;

    function getOpencvCamToThree() {
        if (!opencvCamToThree) {
            opencvCamToThree = new THREE.Matrix4().makeRotationX(Math.PI);
        }
        return opencvCamToThree;
    }

    /** T_three = T_opencv @ C，与后端 OpenCV camera_to_world 对齐 Three 默认相机 */
    function cameraToWorldForThree(rows) {
        var m = mat4FromRows(rows);
        if (!m) return null;
        return new THREE.Matrix4().multiplyMatrices(m, getOpencvCamToThree());
    }

    function pickPoseMatrix(data) {
        if (!data) return null;
        if (data.camera_to_world) return cameraToWorldForThree(data.camera_to_world);
        // imu 体轴系与相机 OpenCV 系不同，不可套 cameraToWorldForThree
        if (data.imu_to_world) return mat4FromRows(data.imu_to_world);
        return null;
    }

    function isValidQuaternion(q) {
        if (!q) return false;
        return (
            Number.isFinite(q.x) &&
            Number.isFinite(q.y) &&
            Number.isFinite(q.z) &&
            Number.isFinite(q.w)
        );
    }

    function getCenterCsrfToken() {
        var form = document.getElementById('realm-csrf-form');
        if (!form) return '';
        var el = form.querySelector('[name=csrfmiddlewaretoken]');
        return el ? el.value : '';
    }

    function postStartImuApi() {
        return fetch(apiUrl('centerStartImu', '/api/center/start-imu/'), {
            method: 'POST',
            headers: { 'X-CSRFToken': getCenterCsrfToken() },
            credentials: 'same-origin',
        }).then(function (res) {
            return res.json();
        });
    }

    function readDefaultCameraPose() {
        var cfg =
            window.REALM_BOOTSTRAP &&
            window.REALM_BOOTSTRAP.imuRecenter;
        var eyeY = readEyeHeightM();
        var pos = (cfg && cfg.cameraPosition) || [0, eyeY, 4.2];
        var rot = (cfg && cfg.cameraRotation) || [0, 0, 0];
        return {
            x: Number(pos[0]) || 0,
            y: Number(pos[1]) || eyeY,
            z: Number(pos[2]) || 4.2,
            rx: Number(rot[0]) || 0,
            ry: Number(rot[1]) || 0,
            rz: Number(rot[2]) || 0,
        };
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

    function recenterImuView() {
        resetTracking();
        var cam = window.camera;
        var pose = readDefaultCameraPose();
        if (cam && typeof THREE !== 'undefined') {
            cam.position.set(pose.x, pose.y, pose.z);
            cam.rotation.set(pose.rx, pose.ry, pose.rz, 'YXZ');
            cam.quaternion.setFromEuler(cam.rotation);
            cam.updateMatrixWorld(true);
        }
        setStatusHint('✌ 视角已回正 · 等待 IMU 重新校准…');
        refreshDebugPanel(lastPayload, lastNetworkError);
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
            // 不要用 IMU 绝对姿态覆盖相机。currentPose 经 OpenCV→Three 后直接赋给 quat 会让默认相机（-Z 前向、Y 上）
            // 与矩阵语义错位，出现「一上来朝后、上下颠倒」。只锁基线矩阵，画面仍以当前场景朝向为 q_ref。
            cam.updateMatrixWorld(true);
            refCameraMatrix = cam.matrixWorld.clone();
            smoothedPos = cam.position.clone();
            smoothedQuat = cam.quaternion.clone();
            setStatusHint('IMU 已校准 · 移动设备/相机可漫游场景');
            refreshDebugPanel(lastPayload, '');
            return;
        }

        var baselineInv = new THREE.Matrix4().copy(baselinePose).invert();
        // Δ = T_curr · T_base⁻¹；目标画面 q = q_delta · q_ref
        var delta = new THREE.Matrix4().copy(currentPose).multiply(baselineInv);
        var pos = new THREE.Vector3();
        var quat = new THREE.Quaternion();
        var scl = new THREE.Vector3();
        delta.decompose(pos, quat, scl);

        var refPos = new THREE.Vector3();
        var refQuat = new THREE.Quaternion();
        var refScl = new THREE.Vector3();
        refCameraMatrix.decompose(refPos, refQuat, refScl);

        if (!isValidQuaternion(quat) || !isValidQuaternion(refQuat)) {
            return;
        }

        var e = new THREE.Euler().setFromQuaternion(quat, 'YXZ');
        e.y = -e.y;
        quat.setFromEuler(e);

        var targetQuat = quat.clone().multiply(refQuat);

        if (!smoothedQuat) {
            smoothedQuat = cam.quaternion.clone();
        }

        smoothedQuat.slerp(targetQuat, SMOOTH);

        if (!isValidQuaternion(smoothedQuat)) {
            console.warn('[REALM IMU] invalid quaternion, skip frame');
            return;
        }

        cam.quaternion.copy(smoothedQuat);
        cam.rotation.order = 'YXZ';
        cam.rotation.setFromQuaternion(cam.quaternion);
        cam.position.y = readEyeHeightM();
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
        if (data.status === 'not_ready') {
            setStatusHint('IMU: 等待首帧…');
            return;
        }
        if (!data.tracking_success && !data.imu_only) {
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

    var imuStartInFlight = false;

    function requestImuPipeline() {
        if (imuStartInFlight) return;
        if (
            window.CenterRealtimeMesh &&
            typeof window.CenterRealtimeMesh.ensurePipelineForImu === 'function'
        ) {
            imuStartInFlight = true;
            window.CenterRealtimeMesh.ensurePipelineForImu()
                .catch(function () {
                    return postStartImuApi();
                })
                .finally(function () {
                    imuStartInFlight = false;
                });
            return;
        }
        imuStartInFlight = true;
        postStartImuApi().finally(function () {
            imuStartInFlight = false;
        });
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
            applyDebugPanelVisibility();
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
            if (shouldTrackImu() && window.camera) {
                var pose = readDefaultCameraPose();
                window.camera.position.set(pose.x, pose.y, pose.z);
            }
            syncTrackingState();
        }
        if (d.action === 'scan-start' || d.action === 'scan-stop') {
            resetTracking();
        }
    });

    window.addEventListener('ar-gesture', function (e) {
        var d = e.detail;
        if (!d || d.action !== 'imu-recenter') return;
        if (!shouldTrackImu()) return;
        recenterImuView();
    });

    window.addEventListener(
        'keydown',
        function (e) {
            if (e.code !== 'KeyN' || e.repeat) return;
            if (!isImmersive3DPage() || !shouldTrackImu()) return;
            var tag = (e.target && e.target.tagName) || '';
            if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
            e.preventDefault();
            toggleDebugPanel();
        },
        true
    );

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
        isDebugPanelVisible: function () {
            return debugPanelVisible;
        },
        toggleDebugPanel: toggleDebugPanel,
        recenterView: recenterImuView,
    };
})();
