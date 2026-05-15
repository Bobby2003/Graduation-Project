/**
 * Depth Scan Import Tester — DEBUG / staff only (page gated server-side).
 */
(function () {
    'use strict';

    var boot = window.DEPTH_SCAN_TESTER_BOOTSTRAP || {};
    var importUrl = boot.importUrl;
    var clearUrl = boot.clearUrl;

    var ERROR_LABELS = {
        INVALID_JSON: 'JSON 格式无效',
        INVALID_PAYLOAD: '请求体无效',
        INVALID_SURFACES: '扫描表面数据无效',
        INVALID_MESH_SUMMARY: '网格摘要无效',
        INVALID_MATERIAL_PACK: '材质协议无效',
        INVALID_SCAN: '扫描对象无效',
        PERMISSION_DENIED: '没有权限执行此操作',
        UNKNOWN: '请求失败',
    };

    var SAMPLE_DEPTH_SCAN = {
        material_pack: 'hacker_matrix',
        surfaces: [
            {
                id: 'floor_plane_001',
                type: 'floor',
                confidence: 0.91,
                center: [0, 0.02, 0.8],
                size: [3.2, 2.8],
                normal: [0, 1, 0],
                material_pack: 'hacker_matrix',
            },
            {
                id: 'wall_plane_001',
                type: 'wall',
                confidence: 0.88,
                center: [0, 1.6, -2.4],
                size: [3.2, 2.2],
                normal: [0, 0, 1],
                material_pack: 'hacker_matrix',
            },
            {
                id: 'desk_zone_001',
                type: 'object_zone',
                confidence: 0.76,
                center: [1.1, 0.6, 0.2],
                size: [1.2, 0.8, 0.7],
                normal: [0, 1, 0],
                material_pack: 'hacker_matrix',
            },
        ],
        mesh_summary: {
            vertices: 12840,
            faces: 24112,
            unit: 'meter',
        },
        pipeline: 'depth_reconstruction_v1',
        coordinate_space: 'metric_room',
        device: {
            type: 'depth_camera',
            name: 'Orbbec / RealSense',
        },
    };

    function getCsrfToken() {
        var inp = document.querySelector('input[name="csrfmiddlewaretoken"]');
        if (inp && inp.value) return inp.value;
        var m = document.cookie.match(/csrftoken=([^;]+)/);
        return m ? decodeURIComponent(m[1]) : '';
    }

    function labelError(code) {
        if (!code) return ERROR_LABELS.UNKNOWN;
        return ERROR_LABELS[code] || code;
    }

    function formatImportResult(res) {
        var data = res.data || {};
        var lines = [];
        lines.push('HTTP ' + res.status + (res.ok ? ' OK' : ''));
        if (res.ok && data.ok) {
            lines.push('导入成功。');
        }
        if (!res.ok && res.status === 403) {
            lines.push('说明: ' + labelError('PERMISSION_DENIED'));
        }
        if (data.error) {
            lines.push('错误码: ' + data.error);
            lines.push('说明: ' + labelError(data.error));
        }
        if (data.ok && data.last_reality_scan) {
            lines.push('--- last_reality_scan ---');
            lines.push(JSON.stringify(data.last_reality_scan, null, 2));
        }
        if (data.progress) {
            lines.push('--- progress ---');
            lines.push(JSON.stringify(data.progress, null, 2));
        }
        if (!data.ok && !data.error && !data.last_reality_scan) {
            lines.push(JSON.stringify(data, null, 2));
        }
        return lines.join('\n');
    }

    function setResult(text) {
        var pre = document.getElementById('depth-import-result');
        if (pre) pre.textContent = text;
    }

    function loadSample() {
        var ta = document.getElementById('depth-scan-json');
        if (ta) ta.value = JSON.stringify(SAMPLE_DEPTH_SCAN, null, 2);
    }

    function importScan() {
        if (!importUrl) {
            setResult('Error: import URL not configured.');
            return;
        }
        var ta = document.getElementById('depth-scan-json');
        var raw = (ta && ta.value) || '';
        var payload;
        try {
            payload = JSON.parse(raw);
        } catch (e) {
            setResult(labelError('INVALID_JSON') + ': ' + e.message);
            return;
        }
        fetch(importUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCsrfToken(),
            },
            body: JSON.stringify(payload),
        })
            .then(function (r) {
                return r.json().then(function (data) {
                    return { ok: r.ok, status: r.status, data: data };
                });
            })
            .then(function (res) {
                setResult(formatImportResult(res));
            })
            .catch(function (err) {
                setResult('网络错误: ' + err);
            });
    }

    function clearScan() {
        if (!clearUrl) {
            setResult('Clear API URL not configured.');
            return;
        }
        if (!confirm('清除 last_reality_scan？（不影响房间模板与灯光）')) return;
        fetch(clearUrl, {
            method: 'POST',
            headers: {
                'X-CSRFToken': getCsrfToken(),
            },
        })
            .then(function (r) {
                return r.json().then(function (data) {
                    return { ok: r.ok, status: r.status, data: data };
                });
            })
            .then(function (res) {
                if (res.data && res.data.ok) {
                    setResult('已清除扫描数据。\n' + JSON.stringify(res.data.realm, null, 2));
                } else {
                    setResult(
                        formatImportResult({
                            ok: res.ok,
                            status: res.status,
                            data: res.data || {},
                        })
                    );
                }
            })
            .catch(function (err) {
                setResult('网络错误: ' + err);
            });
    }

    function bind() {
        var b1 = document.getElementById('load-sample-depth-scan');
        var b2 = document.getElementById('import-depth-scan');
        var b3 = document.getElementById('clear-reality-scan-tester-btn');
        if (b1) b1.addEventListener('click', loadSample);
        if (b2) b2.addEventListener('click', importScan);
        if (b3) b3.addEventListener('click', clearScan);
        loadSample();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', bind);
    } else {
        bind();
    }
})();
