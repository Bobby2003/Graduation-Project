/**
 * /realm/ — 轮询 GET /api/center/latest-mesh/（Center_pipeline.get_latest_mesh_api_payload）
 * 依赖：window.scene、window.THREE（由 realm-main.js 暴露）
 */
(function () {
    var centerRealtimeMesh = null;
    var centerMeshVersion = null;
    var centerMeshLastVertexCount = null;
    var centerMeshPollingTimer = null;
    var centerMeshPollInFlight = false;
    var centerMeshPollDelayTimer = null;
    var centerStatusPollTimer = null;
    var centerStatusPollInFlight = false;
    var centerAutoResumePending = false;

    /** 前几帧标定显示变换，之后只更新顶点/面，不再整体缩小 */
    var centerMeshTransformLocked = false;
    var centerMeshDisplayFrames = 0;
    var CENTER_MESH_LOCK_AFTER_FRAMES = 3;
    var CENTER_MESH_LOCK_MIN_VERTICES = 24;
    var lockedMeshScale = 1;
    var lockedMeshPosition = null;

    var urls = (window.REALM_BOOTSTRAP && window.REALM_BOOTSTRAP.urls) || {};

    function apiUrl(name, fallback) {
        return urls[name] || fallback;
    }

    function getCenterCsrfToken() {
        var form = document.getElementById('realm-csrf-form');
        if (!form) return '';
        var el = form.querySelector('[name=csrfmiddlewaretoken]');
        return el ? el.value : '';
    }

    function formatCenterFetchError(err) {
        var s = err && err.message != null ? String(err.message) : String(err);
        if (/Failed to fetch|NetworkError|fetch failed|Load failed|ERR_CONNECTION|REFUSED/i.test(s)) {
            return (
                '无法连接 Django（连接被拒绝或网络失败）：请在本页同一站点启动并保持 ' +
                '`python manage.py runserver`（建议加 `--noreload`），确认端口与浏览器地址一致。'
            );
        }
        return s;
    }

    function postCenterApi(url) {
        return fetch(url, {
            method: 'POST',
            headers: { 'X-CSRFToken': getCenterCsrfToken() },
            credentials: 'same-origin',
        }).then(function (res) {
            return res.json();
        });
    }

    function resetCenterMeshTransformLock() {
        centerMeshTransformLocked = false;
        centerMeshDisplayFrames = 0;
        lockedMeshScale = 1;
        lockedMeshPosition = null;
    }

    function startCenterRealtime() {
        resetCenterMeshTransformLock();
        centerMeshVersion = null;
        centerMeshLastVertexCount = null;
        if (window.RealmImu && typeof window.RealmImu.resetAnchor === 'function') {
            window.RealmImu.resetAnchor();
        }
        postCenterApi(apiUrl('centerStart', '/api/center/start/')).then(function (data) {
            updateCenterRealtimeStatus(data);
            if (centerMeshPollDelayTimer) {
                clearTimeout(centerMeshPollDelayTimer);
                centerMeshPollDelayTimer = null;
            }
            centerMeshPollDelayTimer = setTimeout(function () {
                centerMeshPollDelayTimer = null;
                startCenterMeshPolling();
                startCenterStatusPolling();
            }, 500);
        }).catch(function (err) {
            updateCenterRealtimeStatus({ ok: false, error: formatCenterFetchError(err) });
        });
    }

    function stopCenterRealtime() {
        if (centerMeshPollDelayTimer) {
            clearTimeout(centerMeshPollDelayTimer);
            centerMeshPollDelayTimer = null;
        }
        stopCenterMeshPolling();
        stopCenterStatusPolling();
        centerAutoResumePending = false;
        resetCenterMeshTransformLock();
        updateCenterRecoveryBanner({ center_recovery_state: 'normal' });
        postCenterApi(apiUrl('centerStop', '/api/center/stop/'))
            .then(function (data) {
                updateCenterRealtimeStatus(data);
            })
            .catch(function (err) {
                updateCenterRealtimeStatus({ ok: false, error: formatCenterFetchError(err) });
            });
    }

    function startCenterMeshPolling() {
        if (centerMeshPollingTimer) return;
        centerMeshPollingTimer = setInterval(function () {
            refreshCenterLatestMesh();
        }, 800);
    }

    function stopCenterMeshPolling() {
        if (!centerMeshPollingTimer) return;
        clearInterval(centerMeshPollingTimer);
        centerMeshPollingTimer = null;
    }

    function refreshCenterLatestMesh() {
        if (centerMeshPollInFlight) return;
        centerMeshPollInFlight = true;
        var u = apiUrl('centerLatestMesh', '/api/center/latest-mesh/');
        fetch(u, { credentials: 'same-origin' })
            .then(function (res) {
                return res.json();
            })
            .then(function (data) {
                updateCenterRealtimeStatus(data);
                if (!data || data.ok === false) return;
                if (data.too_large) return;
                if (!data.mesh) return;
                if (!window.scene || !window.THREE) return;

                var summaryVerts =
                    data.summary && data.summary.vertices != null ? data.summary.vertices : null;
                if (
                    data.version === centerMeshVersion &&
                    summaryVerts === centerMeshLastVertexCount &&
                    centerMeshTransformLocked
                ) {
                    return;
                }
                centerMeshVersion = data.version;
                centerMeshLastVertexCount = summaryVerts;
                renderCenterMemoryMesh(data.mesh, data);
            })
            .catch(function (err) {
                updateCenterRealtimeStatus({ ok: false, error: formatCenterFetchError(err) });
            })
            .finally(function () {
                centerMeshPollInFlight = false;
            });
    }

    function updateCenterRecoveryBanner(data) {
        var banner = document.getElementById('center-recovery-banner');
        if (!banner) return;
        var state = data && data.center_recovery_state;
        var text = '';
        var visible = false;
        if (state === 'recovering') {
            visible = true;
            text = '跟踪丢失，正在恢复…';
            banner.style.borderColor = 'rgba(0, 242, 255, 0.45)';
            banner.style.background = 'rgba(0, 242, 255, 0.08)';
            banner.style.color = '#7ee8ff';
        } else if (state === 'hard_paused_lost') {
            visible = true;
            text = '跟踪丢失，正在准备恢复…';
            banner.style.borderColor = 'rgba(255, 200, 0, 0.45)';
            banner.style.background = 'rgba(255, 180, 0, 0.08)';
            banner.style.color = '#ffd080';
        } else if (state === 'recovery_failed') {
            visible = true;
            text = (data && (data.hint || data.message)) || '恢复失败，请重新建模';
            banner.style.borderColor = 'rgba(255, 80, 80, 0.5)';
            banner.style.background = 'rgba(255, 40, 40, 0.1)';
            banner.style.color = '#ff9999';
        }
        if (visible) {
            banner.hidden = false;
            banner.style.display = 'block';
            banner.textContent = text;
        } else {
            banner.hidden = true;
            banner.style.display = 'none';
            banner.textContent = '';
        }
    }

    function maybeAutoResumeCenter(data) {
        if (!data || !data.running) return;
        if (data.center_recovery_state !== 'hard_paused_lost') {
            centerAutoResumePending = false;
            return;
        }
        if (centerAutoResumePending) return;
        centerAutoResumePending = true;
        postCenterApi(apiUrl('centerResume', '/api/center/resume/'))
            .then(function (res) {
                if (res && res.center_recovery_state) {
                    updateCenterRecoveryBanner(res);
                }
                updateCenterRealtimeStatus(res);
            })
            .catch(function () {
                centerAutoResumePending = false;
            });
    }

    function refreshCenterPipelineStatus() {
        if (centerStatusPollInFlight) return;
        centerStatusPollInFlight = true;
        fetch(apiUrl('centerStatus', '/api/center/status/'), { credentials: 'same-origin' })
            .then(function (res) {
                return res.json();
            })
            .then(function (data) {
                if (data && data.ok !== false) {
                    updateCenterRecoveryBanner(data);
                    maybeAutoResumeCenter(data);
                }
            })
            .catch(function () {})
            .finally(function () {
                centerStatusPollInFlight = false;
            });
    }

    function startCenterStatusPolling() {
        if (centerStatusPollTimer) return;
        refreshCenterPipelineStatus();
        centerStatusPollTimer = setInterval(refreshCenterPipelineStatus, 1000);
    }

    function stopCenterStatusPolling() {
        if (!centerStatusPollTimer) return;
        clearInterval(centerStatusPollTimer);
        centerStatusPollTimer = null;
    }

    function updateCenterRealtimeStatus(data) {
        if (data && data.center_recovery_state) {
            updateCenterRecoveryBanner(data);
        }
        var el = document.getElementById('center-realtime-status');
        if (!el) return;
        if (!data) {
            el.textContent = 'CENTER: NO DATA';
            return;
        }
        if (data.error && data.ok === false) {
            el.textContent = 'CENTER ERROR: ' + data.error;
            return;
        }
        if (!data.running && (data.pipeline_last_error || data.center_hint)) {
            var parts = [];
            if (data.pipeline_last_error) parts.push(data.pipeline_last_error);
            if (data.center_hint) parts.push(data.center_hint);
            el.textContent = 'CENTER STOPPED: ' + parts.join(' — ');
            return;
        }
        if (data.error && !data.running && !data.mesh) {
            el.textContent = 'CENTER ERROR: ' + data.error;
            return;
        }
        if (data.too_large) {
            var s = data.summary || {};
            el.textContent =
                'CENTER: MESH TOO LARGE ' + (s.vertices || '-') + ' V / ' + (s.faces || '-') + ' F';
            return;
        }
        var summary = data.summary || {};
        var hasMeshMeta = data.version != null || summary.vertices != null || summary.faces != null;
        var running = data.running ? 'RUNNING' : 'IDLE';
        if (!data.mesh && data.message) {
            var line =
                'CENTER: ' + running + ' · ' + data.message + '（点 START 后等几秒；无 mesh 时 TSDF 尚未出表面）';
            if (data.hint) line += ' — ' + data.hint;
            if (data.progress) {
                var p = data.progress;
                line +=
                    ' | push→map ' +
                    (p.pushed_to_mapping != null ? p.pushed_to_mapping : '-') +
                    ' · integ ' +
                    (p.integrated_frames != null ? p.integrated_frames : '-') +
                    ' · ' +
                    (p.tracking_mode != null ? p.tracking_mode : '?') +
                    (p.tracking_success != null ? '/' + p.tracking_success : '') +
                    (p.tracker_backend ? ' · ' + p.tracker_backend : '');
            }
            el.textContent = line;
            if (data.diagnostics && window.console && console.debug) {
                console.debug('[CENTER latest-mesh diagnostics]', data.diagnostics);
            }
            return;
        }
        if (!hasMeshMeta) {
            el.textContent =
                'CENTER: ' + running + ' · 已启动（mesh 统计见轮询；若仍为「无 mesh」说明 TSDF 尚未融合出表面）';
            return;
        }
        var version = data.version != null ? data.version : '-';
        var vCount = summary.vertices != null ? summary.vertices : '-';
        var fCount = summary.faces != null ? summary.faces : '-';
        var lockTag = centerMeshTransformLocked ? ' · LOCK' : '';
        el.textContent =
            'CENTER: ' + running + ' · V' + version + ' · ' + vCount + ' verts · ' + fCount + ' faces' + lockTag;
    }

    function removeCenterRealtimeMesh() {
        if (!centerRealtimeMesh) return;
        if (window.scene) window.scene.remove(centerRealtimeMesh);
        centerRealtimeMesh.traverse(function (obj) {
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) {
                if (Array.isArray(obj.material)) {
                    obj.material.forEach(function (m) {
                        if (m && m.dispose) m.dispose();
                    });
                } else if (obj.material.dispose) {
                    obj.material.dispose();
                }
            }
        });
        centerRealtimeMesh = null;
    }

    var REALM_FLOOR_Y = 0;
    var REALM_MESH_FLOOR_EPS = 0.008;

    function applyLockedMeshTransform(object3d) {
        if (!lockedMeshPosition) return;
        object3d.scale.setScalar(lockedMeshScale);
        object3d.position.copy(lockedMeshPosition);
    }

    function captureLockFromMesh(mesh) {
        lockedMeshScale = mesh.scale.x;
        lockedMeshPosition = mesh.position.clone();
        centerMeshTransformLocked = true;
        if (window.console && console.log) {
            console.log('[CENTER] mesh transform locked', {
                scale: lockedMeshScale,
                position: lockedMeshPosition,
                frames: centerMeshDisplayFrames,
            });
        }
    }

    function normalizeCenterMeshToRealm(object3d) {
        if (centerMeshTransformLocked) {
            applyLockedMeshTransform(object3d);
            return;
        }

        object3d.updateMatrixWorld(true);
        var box = new THREE.Box3().setFromObject(object3d);
        var size = new THREE.Vector3();
        var center = new THREE.Vector3();
        box.getSize(size);
        box.getCenter(center);
        var maxDim = Math.max(size.x, size.y, size.z);
        if (!maxDim || !Number.isFinite(maxDim)) return;

        var targetSize = 4.2;
        var scale = targetSize / maxDim;
        object3d.scale.setScalar(scale);
        object3d.position.set(0, 0, 0);
        object3d.updateMatrixWorld(true);
        box.setFromObject(object3d);
        var minY = box.min.y;
        if (!Number.isFinite(minY)) return;
        object3d.position.set(
            -center.x * scale,
            REALM_FLOOR_Y + REALM_MESH_FLOOR_EPS - minY,
            -center.z * scale
        );
    }

    function updateCenterMeshGeometry(mesh, meshPayload) {
        var vertices = meshPayload.vertices || [];
        var indices = meshPayload.indices || [];
        var normals = meshPayload.normals || null;
        var colors = meshPayload.colors || null;
        if (!vertices.length) return false;

        var geo = mesh.geometry;
        var vertCount = vertices.length / 3;

        geo.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));

        if (indices && indices.length) {
            geo.setIndex(indices);
        } else {
            geo.setIndex(null);
        }

        if (normals && normals.length === vertCount * 3) {
            geo.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
        } else {
            geo.deleteAttribute('normal');
            geo.computeVertexNormals();
        }

        if (colors && colors.length === vertCount * 3) {
            geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
            if (mesh.material && !mesh.material.vertexColors) {
                mesh.material.vertexColors = true;
                mesh.material.needsUpdate = true;
            }
        }

        var posAttr = geo.getAttribute('position');
        if (posAttr) posAttr.needsUpdate = true;
        if (geo.index) geo.index.needsUpdate = true;
        geo.computeBoundingSphere();
        return true;
    }

    function buildMeshFromPayload(meshPayload, meta) {
        var vertices = meshPayload.vertices || [];
        var indices = meshPayload.indices || [];
        var normals = meshPayload.normals || null;
        var colors = meshPayload.colors || null;
        var vertCount = vertices.length / 3;

        var geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
        if (indices && indices.length) {
            geometry.setIndex(indices);
        }

        if (normals && normals.length === vertCount * 3) {
            geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
        } else {
            geometry.computeVertexNormals();
        }

        var material;
        if (colors && colors.length === vertCount * 3) {
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
            material = new THREE.MeshStandardMaterial({
                vertexColors: true,
                roughness: 0.75,
                metalness: 0.05,
                side: THREE.DoubleSide,
            });
        } else {
            material = new THREE.MeshStandardMaterial({
                color: 0xb8d7ff,
                roughness: 0.72,
                metalness: 0.08,
                side: THREE.DoubleSide,
                transparent: true,
                opacity: 0.92,
            });
        }

        var mesh = new THREE.Mesh(geometry, material);
        mesh.name = 'CENTER_MEMORY_REALTIME_MESH';
        mesh.userData.isCenterRealtimeMesh = true;
        mesh.userData.skipMaterialProtocol = true;
        mesh.userData.coordinateSpace = (meta && meta.coordinate_space) || 'reconstruction_world';
        mesh.renderOrder = 10;
        return mesh;
    }

    function renderCenterMemoryMesh(meshPayload, meta) {
        if (!window.THREE) {
            console.error('[CENTER] THREE is not available.');
            return;
        }
        if (!window.scene) {
            console.error('[CENTER] window.scene is not available.');
            return;
        }

        var vertices = meshPayload.vertices || [];
        if (!vertices.length) {
            console.warn('[CENTER] mesh has no vertices.');
            return;
        }

        var vertCount = vertices.length / 3;

        if (centerRealtimeMesh && centerMeshTransformLocked && lockedMeshPosition) {
            try {
                if (updateCenterMeshGeometry(centerRealtimeMesh, meshPayload)) {
                    applyLockedMeshTransform(centerRealtimeMesh);
                    return;
                }
            } catch (err) {
                console.warn('[CENTER] locked geometry update failed, rebuilding mesh', err);
            }
        }

        removeCenterRealtimeMesh();

        var mesh = buildMeshFromPayload(meshPayload, meta);
        normalizeCenterMeshToRealm(mesh);

        if (!centerMeshTransformLocked && vertCount >= CENTER_MESH_LOCK_MIN_VERTICES) {
            centerMeshDisplayFrames += 1;
            if (centerMeshDisplayFrames >= CENTER_MESH_LOCK_AFTER_FRAMES) {
                captureLockFromMesh(mesh);
                applyLockedMeshTransform(mesh);
            }
        }

        centerRealtimeMesh = mesh;
        window.scene.add(mesh);
    }

    window.resetCenterMeshTransformLock = resetCenterMeshTransformLock;

    function onReady() {
        var startBtn = document.getElementById('center-start-btn');
        var stopBtn = document.getElementById('center-stop-btn');
        if (startBtn) {
            startBtn.addEventListener('click', function () {
                startCenterRealtime();
            });
        }
        if (stopBtn) {
            stopBtn.addEventListener('click', function () {
                stopCenterRealtime();
            });
        }
        startCenterMeshPolling();
        startCenterStatusPolling();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', onReady);
    } else {
        onReady();
    }
})();
