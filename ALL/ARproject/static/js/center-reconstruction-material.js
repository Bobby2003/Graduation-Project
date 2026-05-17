/**
 * /realm/ — Center 重建网格（语义分区）换材质：独立图层与路由，不涉及 material_pack / material-protocol。
 * 轮询 scene-latest（独立于 center-realtime-mesh 的 latest-mesh）。
 */
(function () {
    var urls = (window.REALM_BOOTSTRAP && window.REALM_BOOTSTRAP.urls) || {};
    var hintEl = null;
    var targetSelectEl = null;
    var librarySelectEl = null;
    var tilingEl = null;
    var applyBtn = null;
    var detailsEl = null;

    var materialRoot = null;
    var trackedSceneVersion = null;
    var scenePollTimer = null;
    var placementSyncTimer = null;

    var fullLibraryMaterials = [];

    function apiUrl(name, fallback) {
        return urls[name] || fallback;
    }

    function getCenterCsrfToken() {
        var form = document.getElementById('realm-csrf-form');
        if (!form) return '';
        var el = form.querySelector('[name=csrfmiddlewaretoken]');
        return el ? el.value : '';
    }

    function setHint(text, tone) {
        if (!hintEl) return;
        hintEl.textContent = text || '';
        hintEl.style.color =
            tone === 'bad'
                ? '#ff9999'
                : tone === 'ok'
                  ? '#7ee8ff'
                  : 'rgba(255,255,255,0.7)';
    }

    function disposeObject3D(root) {
        if (!root) return;
        root.traverse(function (obj) {
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) {
                var mats = Array.isArray(obj.material) ? obj.material : [obj.material];
                mats.forEach(function (m) {
                    if (m && m.map && m.map.dispose) m.map.dispose();
                    if (m && m.dispose) m.dispose();
                });
            }
        });
    }

    function ensureMaterialSceneRoot() {
        if (!window.THREE || !window.scene) return null;
        if (materialRoot) return materialRoot;
        materialRoot = new THREE.Group();
        materialRoot.name = 'CENTER_MATERIAL_GLTF_ROOT';
        materialRoot.userData.isCenterReconstructionMaterialLayer = true;
        window.scene.add(materialRoot);
        return materialRoot;
    }

    function clearMaterialGltfChildren() {
        if (!materialRoot) return;
        while (materialRoot.children.length) {
            var ch = materialRoot.children[0];
            materialRoot.remove(ch);
            disposeObject3D(ch);
        }
    }

    function syncMaterialLayerToMeshAnchor() {
        if (!materialRoot || !materialRoot.visible) return;
        var getter = window.CenterRealtimeMesh && window.CenterRealtimeMesh.getAnchor;
        var anchor = typeof getter === 'function' ? getter() : null;
        if (!anchor) return;
        materialRoot.scale.copy(anchor.scale);
        materialRoot.position.copy(anchor.position);
        materialRoot.quaternion.copy(anchor.quaternion);
    }

    function startPlacementSync() {
        if (placementSyncTimer) return;
        placementSyncTimer = setInterval(syncMaterialLayerToMeshAnchor, 600);
    }

    function stopPlacementSync() {
        if (!placementSyncTimer) return;
        clearInterval(placementSyncTimer);
        placementSyncTimer = null;
    }

    function getGltfLoaderConstructor() {
        return window.__RealmGLTFLoader || (window.THREE && window.THREE.GLTFLoader);
    }

    /** head 里 type=module 的脚本为 deferred，可能比同步脚本晚挂到 window。 */
    function waitForGltfLoaderConstructor() {
        return new Promise(function (resolve, reject) {
            var tries = 0;
            var maxTries = 300;
            function tick() {
                var Ctor = getGltfLoaderConstructor();
                if (Ctor) {
                    resolve(Ctor);
                    return;
                }
                if (++tries > maxTries) {
                    reject(new Error('THREE.GLTF_LOADER_MISSING'));
                    return;
                }
                setTimeout(tick, 20);
            }
            tick();
        });
    }

    function loadCenterMaterialGlb(glbUrl) {
        ensureMaterialSceneRoot();
        if (!materialRoot || !window.THREE) return Promise.reject(new Error('NO_SCENE'));

        clearMaterialGltfChildren();

        return waitForGltfLoaderConstructor().then(function (LoaderCtor) {
            return new Promise(function (resolve, reject) {
                var loader = new LoaderCtor();
                loader.load(
                    glbUrl,
                    function (gltf) {
                        var rootScene = gltf.scene;
                        rootScene.traverse(function (o) {
                            if (o.isMesh) {
                                o.renderOrder = 11;
                                o.userData.isCenterReconstructionMaterialMesh = true;
                                o.castShadow = true;
                                o.receiveShadow = true;
                            }
                        });
                        materialRoot.add(rootScene);
                        syncMaterialLayerToMeshAnchor();
                        startPlacementSync();
                        resolve(rootScene);
                    },
                    undefined,
                    function (err) {
                        reject(err);
                    }
                );
            });
        });
    }

    function rebuildLibraryOptionsForSegment() {
        if (!librarySelectEl) return;
        while (librarySelectEl.firstChild) {
            librarySelectEl.removeChild(librarySelectEl.firstChild);
        }
        var tgt = selectedTargetSummary();
        var allowed =
            tgt && Array.isArray(tgt.available_materials) ? new Set(tgt.available_materials) : null;

        fullLibraryMaterials.forEach(function (m) {
            if (!m || !m.material_id) return;
            if (allowed && !allowed.has(m.material_id)) return;
            var opt = document.createElement('option');
            opt.value = m.material_id;
            opt.textContent = m.name ? m.material_id + ' — ' + m.name : String(m.material_id);
            librarySelectEl.appendChild(opt);
        });

        if (librarySelectEl.options.length === 0) {
            var empty = document.createElement('option');
            empty.value = '';
            empty.textContent = '(无可用材质)';
            librarySelectEl.appendChild(empty);
        }

        if (tgt && tgt.current_material) {
            var pick = tgt.current_material;
            for (var i = 0; i < librarySelectEl.options.length; i++) {
                if (librarySelectEl.options[i].value === pick) {
                    librarySelectEl.selectedIndex = i;
                    break;
                }
            }
        }
    }

    function selectedTargetSummary() {
        if (!targetSelectEl || !targetSelectEl.value) return null;
        var raw = targetSelectEl.options[targetSelectEl.selectedIndex];
        if (!raw) return null;
        try {
            return JSON.parse(raw.getAttribute('data-target-json') || 'null');
        } catch (e) {
            return null;
        }
    }

    function fetchJson(url, opts) {
        return fetch(url, opts || { credentials: 'same-origin' }).then(function (res) {
            return res.json().then(function (body) {
                return { okHttp: res.ok, status: res.status, body: body };
            });
        });
    }

    function rebuildTargetOptions(payload) {
        if (!targetSelectEl) return;
        while (targetSelectEl.firstChild) {
            targetSelectEl.removeChild(targetSelectEl.firstChild);
        }
        var targets = (payload && payload.targets) || [];
        if (!targets.length) {
            var o = document.createElement('option');
            o.value = '';
            o.textContent = '(无分区)';
            targetSelectEl.appendChild(o);
            return;
        }
        targets.forEach(function (t) {
            var opt = document.createElement('option');
            opt.value = t.segment_key;
            opt.textContent =
                (t.display_name ? t.display_name + ' · ' : '') + String(t.segment_key);
            opt.setAttribute('data-target-json', JSON.stringify(t));
            targetSelectEl.appendChild(opt);
        });
    }

    function refreshPanelsFromServerSilent() {
        var uT = apiUrl('centerMaterialTargets', '/api/center/material/targets/');
        var uL = apiUrl('centerMaterialLibrary', '/api/center/material/library/');
        return fetchJson(uL)
            .catch(function () {
                return { body: { ok: false, materials: [] } };
            })
            .then(function (libWrap) {
                var lb = libWrap.body || {};
                fullLibraryMaterials = Array.isArray(lb.materials) ? lb.materials : [];
                return fetchJson(uT).then(function (tw) {
                    return { targetsWrap: tw, libWrap: libWrap };
                });
            })
            .then(function (bund) {
                var tw = bund.targetsWrap || {};
                var body = tw.body || {};
                if (body.ok && body.targets) {
                    rebuildTargetOptions(body);
                    rebuildLibraryOptionsForSegment();
                    if (applyBtn) applyBtn.disabled = false;
                    return { ready: true, version: body.version };
                }
                if (applyBtn) applyBtn.disabled = true;
                return { ready: false, version: body && body.version };
            });
    }

    function reloadMaterialSceneGlb(reason) {
        var uLatest = apiUrl('centerMaterialSceneLatest', '/api/center/material/scene-latest/');
        return fetchJson(uLatest).then(function (wrap) {
            var b = wrap.body || {};
            if (!b.ok || b.version == null) {
                return null;
            }
            trackedSceneVersion = b.version;
            var glbUrl =
                apiUrl('centerMaterialSceneGlb', '/api/center/material/scene.glb') +
                '?version=' +
                encodeURIComponent(String(b.version)) +
                '&_=' +
                Date.now();

            return loadCenterMaterialGlb(glbUrl).then(function () {
                setHint(
                    '已加载重建材质 scene v' +
                        trackedSceneVersion +
                        (reason ? '（' + reason + '）' : ''),
                    'ok'
                );
            });
        });
    }

    function onScenePollTick() {
        var uLatest = apiUrl('centerMaterialSceneLatest', '/api/center/material/scene-latest/');
        fetchJson(uLatest)
            .then(function (wrap) {
                var b = wrap.body || {};
                if (!b.ok || b.version == null) return null;
                if (trackedSceneVersion !== null && b.version !== trackedSceneVersion) {
                    return reloadMaterialSceneGlb('version');
                }
                return null;
            })
            .catch(function () {});
    }

    function startScenePolling() {
        if (scenePollTimer) return;
        scenePollTimer = setInterval(onScenePollTick, 2400);
    }

    function stopScenePolling() {
        if (!scenePollTimer) return;
        clearInterval(scenePollTimer);
        scenePollTimer = null;
    }

    function onDetailsToggle() {
        if (!detailsEl) return;
        var open = detailsEl.open;
        if (open) {
            bootstrapPanel().catch(function () {});
            startScenePolling();
        } else {
            stopScenePolling();
            stopPlacementSync();
        }
    }

    function bootstrapPanel() {
        setHint('正在查询分区与材质库…');
        if (applyBtn) applyBtn.disabled = true;
        return refreshPanelsFromServerSilent().then(function (meta) {
            ensureMaterialSceneRoot();
            if (!materialRoot) return;
            if (!meta.ready) {
                materialRoot.visible = false;
                setHint('请先启动扫描并等待分区完成（Center Material 就绪后此处可选分区）', '');
                trackedSceneVersion = null;
                return;
            }
            materialRoot.visible = true;
            setHint(
                trackedSceneVersion
                    ? 'Material 场景 v' + trackedSceneVersion
                    : 'Material 就绪，可选分区并切换材质。',
                ''
            );
            return reloadMaterialSceneGlb('open').catch(function (err) {
                console.warn('[CRM]', err);
                setHint(
                    err && err.message === 'THREE.GLTF_LOADER_MISSING'
                        ? '缺少 GLTF 加载器：请确认浏览器支持 importmap，并已强刷页面'
                        : '加载 scene.glb 失败：' + (err && err.message ? err.message : err),
                    'bad'
                );
            });
        });
    }

    function setupDom() {
        detailsEl = document.getElementById('center-reconstruction-material-details');
        hintEl = document.getElementById('crm-recon-hint');
        targetSelectEl = document.getElementById('crm-recon-target');
        librarySelectEl = document.getElementById('crm-recon-library');
        tilingEl = document.getElementById('crm-recon-tiling');
        applyBtn = document.getElementById('crm-recon-apply');

        if (!detailsEl || !applyBtn || !targetSelectEl || !librarySelectEl) return;

        detailsEl.addEventListener('toggle', onDetailsToggle);

        targetSelectEl.addEventListener('change', function () {
            rebuildLibraryOptionsForSegment();
        });

        applyBtn.addEventListener('click', function () {
            var sk = targetSelectEl.value;
            var mid = librarySelectEl.value;
            var tiling = tilingEl ? parseFloat(String(tilingEl.value)) : 1.0;
            if (!sk || !mid) return;
            applyBtn.disabled = true;
            setHint('应用中…');

            fetch(apiUrl('centerMaterialChange', '/api/center/material/change/'), {
                method: 'POST',
                credentials: 'same-origin',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCenterCsrfToken(),
                },
                body: JSON.stringify({
                    segment_key: sk,
                    material_id: mid,
                    tiling: Number.isFinite(tiling) ? tiling : 1.0,
                }),
            })
                .then(function (res) {
                    return res.json().then(function (body) {
                        return { okHttp: res.ok, status: res.status, body: body };
                    });
                })
                .then(function (wrap) {
                    var b = wrap.body || {};
                    if (!b.ok) {
                        applyBtn.disabled = false;
                        if (b.error === 'MATERIAL_NOT_READY') {
                            setHint('请先启动扫描并等待分区完成', '');
                            return;
                        }
                        setHint('应用失败：' + (b.error || wrap.status), 'bad');
                        return;
                    }
                    trackedSceneVersion = b.version != null ? b.version : trackedSceneVersion;
                    return reloadMaterialSceneGlb('change').finally(function () {
                        applyBtn.disabled = false;
                        setHint(
                            '已更新 segment：' +
                                sk +
                                (b.version != null ? ' · v' + b.version : ''),
                            'ok'
                        );
                        return refreshPanelsFromServerSilent().catch(function () {});
                    });
                })
                .catch(function (err) {
                    applyBtn.disabled = false;
                    setHint('网络错误：' + (err && err.message ? err.message : err), 'bad');
                });
        });

        ensureMaterialSceneRoot();
        if (materialRoot) {
            materialRoot.visible = false;
        }

        refreshPanelsFromServerSilent()
            .then(function (meta) {
                if (!meta || !meta.ready) {
                    if (applyBtn) applyBtn.disabled = true;
                    setHint('请先启动扫描并等待分区完成', '');
                } else {
                    if (!(detailsEl && detailsEl.open)) {
                        setHint('', '');
                        if (applyBtn) applyBtn.disabled = false;
                    }
                }
            })
            .catch(function () {});
    }

    function initDelayed() {
        setupDom();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initDelayed);
    } else {
        initDelayed();
    }
})();
