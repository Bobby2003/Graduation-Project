/**
 * REALITY OVERRIDE — 实验模式（摄像头 + 模拟扫描 + overlay + 保存）
 */
(function () {
    'use strict';

    var boot = window.REALITY_OVERRIDE_BOOTSTRAP || {};
    var urls = boot.urls || {};

    var ERROR_LABELS = {
        INVALID_JSON: 'JSON 格式无效',
        INVALID_PAYLOAD: '请求体无效',
        INVALID_SURFACES: '扫描表面数据无效',
        INVALID_MATERIAL_PACK: '材质协议无效',
        INVALID_SCAN: '扫描对象无效',
        UNKNOWN: '保存失败',
    };

    function labelSaveError(code) {
        if (!code) return ERROR_LABELS.UNKNOWN;
        return ERROR_LABELS[code] || code;
    }

    var overrideState = {
        stream: null,
        scanning: false,
        progress: 0,
        surfaces: [],
        materialPack: 'cyber_neon',
    };

    function getCsrfToken() {
        var inp = document.querySelector('input[name="csrfmiddlewaretoken"]');
        if (inp && inp.value) return inp.value;
        var m = document.cookie.match(/csrftoken=([^;]+)/);
        return m ? decodeURIComponent(m[1]) : '';
    }

    function showOverrideToast(msg, isErr) {
        var stack = document.getElementById('ro-toast-stack');
        if (!stack) {
            if (isErr) window.alert(msg);
            return;
        }
        var el = document.createElement('div');
        el.className = 'ro-toast' + (isErr ? ' err' : '');
        el.textContent = msg;
        stack.appendChild(el);
        setTimeout(function () {
            el.remove();
        }, 4200);
    }

    function setDeviceStatus(t) {
        var el = document.getElementById('device-status');
        if (el) el.textContent = t;
    }

    function setScanStatus(t) {
        var el = document.getElementById('scan-status');
        if (el) el.textContent = t;
    }

    function updateScanProgress() {
        var fill = document.getElementById('scan-progress-fill');
        if (fill) fill.style.width = overrideState.progress + '%';
    }

    function handleProgressResponse(data) {
        if (!data || !data.ok) return;
        if (Array.isArray(data.newly_completed_missions)) {
            data.newly_completed_missions.forEach(function (mission) {
                showOverrideToast(
                    '任务完成: ' + mission.title + ' · +' + mission.reward_exp + ' EXP',
                    false
                );
            });
        }
        if (Array.isArray(data.newly_unlocked_achievements)) {
            data.newly_unlocked_achievements.forEach(function (ach) {
                var tag = ach.icon_key ? '[' + ach.icon_key + '] ' : '';
                showOverrideToast(tag + ach.name + ' — ' + (ach.description || ''), false);
            });
        }
    }

    function reportProgressEvent(event, payload) {
        var url = urls.progressEvent;
        if (!url) return;
        fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCsrfToken(),
            },
            body: JSON.stringify({ event: event, payload: payload || {} }),
        })
            .then(function (r) {
                return r.json();
            })
            .then(function (data) {
                if (!data || !data.ok) {
                    if (data) console.warn('progress event failed:', data);
                    return;
                }
                handleProgressResponse(data);
            })
            .catch(function (err) {
                console.error('progress event error:', err);
            });
    }

    function resizeOverlay() {
        var video = document.getElementById('override-video');
        var canvas = document.getElementById('override-overlay');
        if (!video || !canvas) return;
        canvas.width = video.clientWidth;
        canvas.height = video.clientHeight;
        drawOverlay();
    }

    function drawOverlay() {
        var canvas = document.getElementById('override-overlay');
        if (!canvas) return;
        var ctx = canvas.getContext('2d');
        if (!ctx) return;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        overrideState.surfaces.forEach(function (surface) {
            var b = surface.bounds;
            if (!b || typeof b.x !== 'number') return;
            var x = b.x * canvas.width;
            var y = b.y * canvas.height;
            var w = b.w * canvas.width;
            var h = b.h * canvas.height;

            var color =
                surface.type === 'floor'
                    ? '#00ff88'
                    : surface.type === 'wall'
                      ? '#00f2ff'
                      : '#bc00ff';

            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(x, y, w, h);

            ctx.fillStyle = color;
            ctx.font = '12px Syncopate, sans-serif';
            var conf = typeof surface.confidence === 'number' ? surface.confidence : 0;
            ctx.fillText(
                surface.type.toUpperCase() + ' ' + (conf * 100).toFixed(0) + '%',
                x + 8,
                y + 18
            );
        });
    }

    function enableCamera() {
        var video = document.getElementById('override-video');
        if (!video) return;

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            setDeviceStatus('DEVICE: UNSUPPORTED');
            showOverrideToast('当前浏览器不支持摄像头 API', true);
            return;
        }

        navigator.mediaDevices
            .getUserMedia({
                video: { facingMode: 'environment' },
                audio: false,
            })
            .then(function (stream) {
                overrideState.stream = stream;
                video.srcObject = stream;
                setDeviceStatus('DEVICE: CAMERA ONLINE');
                var scanBtn = document.getElementById('start-scan-btn');
                if (scanBtn) scanBtn.disabled = false;
                video.addEventListener('loadedmetadata', resizeOverlay, { once: true });
                setTimeout(resizeOverlay, 200);
            })
            .catch(function () {
                setDeviceStatus('DEVICE: CAMERA BLOCKED');
                showOverrideToast('摄像头权限被拒绝或设备不可用', true);
            });
    }

    function completeScan() {
        overrideState.surfaces = [
            {
                id: 'floor_001',
                type: 'floor',
                confidence: 0.94,
                bounds: { x: 0.08, y: 0.66, w: 0.84, h: 0.26 },
            },
            {
                id: 'wall_001',
                type: 'wall',
                confidence: 0.89,
                bounds: { x: 0.12, y: 0.12, w: 0.76, h: 0.38 },
            },
            {
                id: 'object_001',
                type: 'object_zone',
                confidence: 0.78,
                bounds: { x: 0.58, y: 0.48, w: 0.28, h: 0.18 },
            },
        ];

        setScanStatus('SCAN: 3 SURFACES DETECTED');
        var applyBtn = document.getElementById('apply-override-btn');
        if (applyBtn) applyBtn.disabled = false;
        drawOverlay();

        reportProgressEvent('reality_scan_completed', {
            mode: 'simulated',
            surface_count: overrideState.surfaces.length,
        });
    }

    function startScan() {
        if (overrideState.scanning) return;

        overrideState.scanning = true;
        overrideState.progress = 0;
        overrideState.surfaces = [];
        drawOverlay();

        setScanStatus('SCAN: ANALYZING SURFACES');

        var rp = document.getElementById('ro-result-panel');
        if (rp) rp.classList.remove('visible');

        var timer = setInterval(function () {
            overrideState.progress += 4;
            updateScanProgress();

            if (overrideState.progress >= 100) {
                clearInterval(timer);
                overrideState.progress = 100;
                updateScanProgress();
                overrideState.scanning = false;
                completeScan();
            }
        }, 80);
    }

    function showOverrideResultPanel() {
        var panel = document.getElementById('ro-result-panel');
        var meta = document.getElementById('ro-result-meta');
        if (panel) panel.classList.add('visible');
        if (meta) {
            meta.textContent =
                'Material: ' +
                overrideState.materialPack +
                ' · Surfaces: ' +
                overrideState.surfaces.length;
        }
    }

    function applyProtocol() {
        var url = urls.save;
        if (!url) {
            showOverrideToast('未配置保存接口', true);
            return;
        }

        var sel = document.getElementById('override-material-select');
        if (sel) overrideState.materialPack = sel.value || overrideState.materialPack;

        var scanPayload = {
            id: 'scan_web_' + Date.now(),
            mode: 'simulated',
            source: 'web_simulator',
            pipeline: 'reality_override_web',
            coordinate_space: 'screen_normalized',
            surfaces: overrideState.surfaces,
            mesh_summary: null,
            raw_ref: null,
        };

        fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCsrfToken(),
            },
            body: JSON.stringify({
                material_pack: overrideState.materialPack,
                scan: scanPayload,
            }),
        })
            .then(function (r) {
                return r.json().then(function (data) {
                    return { httpOk: r.ok, status: r.status, data: data };
                });
            })
            .then(function (res) {
                var data = res.data;
                if (res.status === 403) {
                    showOverrideToast('没有权限或 CSRF 校验失败', true);
                    return;
                }
                if (!data || !data.ok) {
                    showOverrideToast(labelSaveError(data && data.error), true);
                    return;
                }
                showOverrideToast('已保存材质与扫描记录到私人位面配置', false);
                showOverrideResultPanel();
                reportProgressEvent('reality_override_applied', {
                    mode: 'simulated',
                    material_pack: overrideState.materialPack,
                    surface_count: overrideState.surfaces.length,
                });
            })
            .catch(function () {
                showOverrideToast('网络错误，保存失败', true);
            });
    }

    function bind() {
        var cam = document.getElementById('start-camera-btn');
        var scan = document.getElementById('start-scan-btn');
        var apply = document.getElementById('apply-override-btn');
        var sel = document.getElementById('override-material-select');
        if (cam) cam.addEventListener('click', enableCamera);
        if (scan) scan.addEventListener('click', startScan);
        if (apply) apply.addEventListener('click', applyProtocol);
        if (sel) {
            sel.addEventListener('change', function () {
                overrideState.materialPack = sel.value;
            });
            overrideState.materialPack = sel.value;
        }
        window.addEventListener('resize', resizeOverlay);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', bind);
    } else {
        bind();
    }
})();
