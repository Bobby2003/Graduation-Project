/**
 * 设置页 · 设备区：枚举摄像头并展示信息（无需跳转扫描引擎页）
 */
(function () {
    'use strict';

    var statusEl = document.getElementById('settings-depth-camera-status');
    if (!statusEl) return;

    var centerStatusUrl = statusEl.getAttribute('data-center-status-url') || '';

    function escapeHtml(s) {
        return String(s)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    function setHtml(html) {
        statusEl.innerHTML = html;
    }

    function setText(text) {
        statusEl.textContent = text;
    }

    function formatCameraLines(cameras) {
        if (!cameras.length) return '';
        return cameras
            .map(function (c, i) {
                var label = (c.label && c.label.trim()) || '摄像头 ' + (i + 1);
                var meta = [];
                if (c.deviceId) {
                    meta.push('ID ' + c.deviceId.slice(0, 16) + (c.deviceId.length > 16 ? '…' : ''));
                }
                if (c.groupId) {
                    meta.push('组 ' + c.groupId.slice(0, 8) + '…');
                }
                return (
                    '<div class="settings-camera-line">' +
                    '<span class="settings-camera-name">' +
                    escapeHtml(label) +
                    '</span>' +
                    (meta.length
                        ? '<span class="settings-camera-meta">' + escapeHtml(meta.join(' · ')) + '</span>'
                        : '') +
                    '</div>'
                );
            })
            .join('');
    }

    function renderNoCamera(message) {
        setHtml(
            '<span class="settings-device-muted">' +
                escapeHtml(message || '未检测到视频输入设备') +
                '</span>'
        );
    }

    function renderCameras(cameras, centerNote) {
        var head =
            '<div class="settings-device-ok">已检测到 ' +
            cameras.length +
            ' 个摄像头</div>';
        var body = formatCameraLines(cameras);
        var foot = centerNote
            ? '<div class="settings-device-sub">' + centerNote + '</div>'
            : '';
        setHtml(head + body + foot);
    }

    function fetchCenterNote() {
        if (!centerStatusUrl) return Promise.resolve('');
        return fetch(centerStatusUrl, { credentials: 'same-origin' })
            .then(function (r) {
                return r.ok ? r.json() : null;
            })
            .then(function (data) {
                if (!data || !data.ok) {
                    if (data && data.error) {
                        return (
                            'Center 深度管线：未就绪（' +
                            escapeHtml(String(data.error).slice(0, 80)) +
                            '）'
                        );
                    }
                    return '';
                }
                if (data.running) {
                    return 'Center 深度管线：<span class="settings-device-ok-inline">运行中</span> · 可在绿洲页 START 扫描';
                }
                var pl = data.pipeline;
                if (pl && typeof pl === 'object' && pl.last_error) {
                    return (
                        'Center 深度管线：空闲 · ' +
                        escapeHtml(String(pl.last_error).slice(0, 100))
                    );
                }
                return 'Center 深度管线：空闲 · 奥比中光等设备需在本地启动采集';
            })
            .catch(function () {
                return '';
            });
    }

    async function enumerateVideoInputs() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
            return { error: '当前浏览器不支持 MediaDevices API' };
        }

        var devices = await navigator.mediaDevices.enumerateDevices();
        var cameras = devices.filter(function (d) {
            return d.kind === 'videoinput';
        });

        var needLabels = cameras.length > 0 && cameras.every(function (c) {
            return !c.label;
        });

        if (needLabels) {
            try {
                var stream = await navigator.mediaDevices.getUserMedia({
                    video: true,
                    audio: false,
                });
                stream.getTracks().forEach(function (t) {
                    t.stop();
                });
                devices = await navigator.mediaDevices.enumerateDevices();
                cameras = devices.filter(function (d) {
                    return d.kind === 'videoinput';
                });
            } catch (e) {
                return {
                    cameras: cameras,
                    permission: 'denied',
                    permissionError: e && e.name ? e.name : 'NotAllowedError',
                };
            }
        }

        return { cameras: cameras, permission: 'ok' };
    }

    async function run() {
        setText('正在检测摄像头…');

        try {
            var result = await enumerateVideoInputs();
            if (result.error) {
                renderNoCamera(result.error);
                return;
            }

            var cameras = result.cameras || [];
            var centerNote = await fetchCenterNote();

            if (!cameras.length) {
                var msg = '未检测到摄像头';
                if (result.permission === 'denied') {
                    msg += '（请允许浏览器摄像头权限后刷新）';
                }
                if (centerNote) {
                    setHtml(
                        '<span class="settings-device-muted">' +
                            escapeHtml(msg) +
                            '</span>' +
                            '<div class="settings-device-sub">' +
                            centerNote +
                            '</div>'
                    );
                } else {
                    renderNoCamera(msg);
                }
                return;
            }

            if (result.permission === 'denied') {
                centerNote =
                    (centerNote ? centerNote + '<br>' : '') +
                    '<span class="settings-device-muted">部分设备名称需摄像头权限才能显示</span>';
            }

            renderCameras(cameras, centerNote);
        } catch (e) {
            console.error('[settings] camera probe failed', e);
            renderNoCamera('检测失败，请刷新重试');
        }
    }

    run();
})();
