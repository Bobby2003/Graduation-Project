/**

 * /realm/ — 轮询 GET /api/center/latest-mesh/，用内存 vertices/indices 更新 Three.BufferGeometry。

 * 前几帧：后端用深度图 + mesh 包围盒解算真实尺寸（米）与锚点；前端 scale=1，不再压到固定 4.2m。

 */

(function () {

    var centerRealtimeMesh = null;

    var centerMeshAnchor = null;

    var centerMeshVersion = null;

    var centerMeshPollingTimer = null;

    var centerMeshPollInFlight = false;

    var centerMeshPollDelayTimer = null;

    var centerStatusPollTimer = null;

    var centerStatusPollInFlight = false;

    var centerAutoResumePending = false;



    var PLACEMENT_LOCK_SAMPLES = 3;

    var MIN_CALIB_VERTICES = 120;

    var REALM_MESH_FLOOR_EPS = 0.008;



    function readMeshPlacementConfig() {

        var mp = (window.REALM_BOOTSTRAP && window.REALM_BOOTSTRAP.meshPlacement) || {};

        var floorY = mp.floorY;

        if (floorY == null && mp.baseHeightM != null) floorY = mp.baseHeightM;

        if (typeof floorY !== 'number' || !Number.isFinite(floorY)) {
            if (typeof window.readRealmMeshBaseHeightM === 'function') {
                floorY = window.readRealmMeshBaseHeightM();
            } else if (typeof window.readRealmEyeHeightM === 'function') {
                floorY = window.readRealmEyeHeightM();
            } else {
                floorY = 1.65;
            }
        }

        return {

            floorY: floorY,

            useMetricScale: mp.useMetricScale !== false,

            lockSamples: typeof mp.lockSamples === 'number' ? mp.lockSamples : PLACEMENT_LOCK_SAMPLES,

        };

    }



    var meshPlacementCfg = readMeshPlacementConfig();



    var placementLocked = false;

    var lockedPlacement = null;

    var calibrationSamples = [];

    var lastPlacementMeta = null;

    var lastSemanticHorizontalUp = null;



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

                "无法连接 Django（连接被拒绝或网络失败）：请在本页同一站点启动并保持 " +

                "`python manage.py runserver`（建议加 `--noreload`），确认端口与浏览器地址一致。"

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



    function isImmersiveControlMode() {

        var m = getControlMode();

        return m === 'gesture' || m === 'vr';

    }



    function resetCenterMeshSession() {

        removeCenterRealtimeMesh();

        if (centerMeshAnchor && window.scene) {

            window.scene.remove(centerMeshAnchor);

        }

        centerMeshAnchor = null;

        placementLocked = false;

        lockedPlacement = null;

        calibrationSamples = [];

        lastPlacementMeta = null;

        lastSemanticHorizontalUp = null;

        centerMeshVersion = null;

        window.__centerMeshPlacementLocked = false;

    }



    function percentileSorted(sorted, p) {

        if (!sorted.length) return 0;

        var idx = Math.min(sorted.length - 1, Math.max(0, Math.floor(sorted.length * p)));

        return sorted[idx];

    }



    /**

     * reconstruction_world 顶点（米）：真实包围盒；底面对齐 meshPlacement.floorY（默认 1.7m）。

     */

    function computePlacementFromVertices(vertices) {

        var n = Math.floor(vertices.length / 3);

        if (n < 4) return null;



        var minX = Infinity;

        var maxX = -Infinity;

        var minZ = Infinity;

        var maxZ = -Infinity;

        var ys = [];

        var i;

        for (i = 0; i < n; i++) {

            var x = vertices[i * 3];

            var y = vertices[i * 3 + 1];

            var z = vertices[i * 3 + 2];

            if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;

            if (x < minX) minX = x;

            if (x > maxX) maxX = x;

            if (z < minZ) minZ = z;

            if (z > maxZ) maxZ = z;

            ys.push(y);

        }

        if (!ys.length) return null;

        ys.sort(function (a, b) {

            return a - b;

        });



        var floorY = percentileSorted(ys, 0.08);

        var maxY = ys[ys.length - 1];

        var centerX = (minX + maxX) * 0.5;

        var centerZ = (minZ + maxZ) * 0.5;

        var sizeX = maxX - minX;

        var sizeY = maxY - floorY;

        var sizeZ = maxZ - minZ;

        var maxDim = Math.max(sizeX, sizeY, sizeZ, 1e-6);

        if (!Number.isFinite(maxDim) || maxDim < 1e-4) return null;



        var scale = meshPlacementCfg.useMetricScale ? 1 : 4.2 / maxDim;

        var floorTarget = meshPlacementCfg.floorY + REALM_MESH_FLOOR_EPS;

        var position = new THREE.Vector3(

            -centerX * scale,

            floorTarget - floorY * scale,

            -centerZ * scale

        );



        return {

            scale: scale,

            position: position,

            vertexCount: n,

            floorY: floorY,

            maxDim: maxDim,

            extent_m: { x: sizeX, y: sizeY, z: sizeZ },

            method: 'client_metric_bbox',

        };

    }



    function placementFromApiPayload(apiPlacement) {

        if (!apiPlacement) return null;

        var scale = typeof apiPlacement.scale === 'number' ? apiPlacement.scale : 1;

        var pos = apiPlacement.position;

        if (!pos || pos.length < 3) return null;



        var position = new THREE.Vector3(pos[0], pos[1], pos[2]);

        if (meshPlacementCfg.useMetricScale) {
            scale = 1;
        }

        return {

            scale: scale,

            position: position,

            extent_m: apiPlacement.extent_m || null,

            depth_extent_m: apiPlacement.depth_extent_m || null,

            reference_camera_to_world: apiPlacement.reference_camera_to_world || null,

            method: apiPlacement.method || 'server_metric',

            locked: !!apiPlacement.locked,

        };

    }



    function averagePlacement(samples) {

        if (!samples.length) return null;

        var scale = 0;

        var px = 0;

        var py = 0;

        var pz = 0;

        samples.forEach(function (s) {

            scale += s.scale;

            px += s.position.x;

            py += s.position.y;

            pz += s.position.z;

        });

        var n = samples.length;

        return {

            scale: scale / n,

            position: new THREE.Vector3(px / n, py / n, pz / n),

        };

    }



    function applyPlacementToAnchor(placement) {

        if (!centerMeshAnchor || !placement) return;

        centerMeshAnchor.scale.setScalar(placement.scale);

        centerMeshAnchor.position.copy(placement.position);

    }



    /**

     * 用网格 Y 方向两端带状区域内的三角形法线估计地面外法线（网格局部坐标），兼容地面在 ymin 或 ymax 一侧；

     * 三角形样本不足时用顶点法线回退。

     */

    function estimateFloorUpNormalFromGeometry(geometry) {

        if (!geometry || !geometry.attributes || !geometry.attributes.position) {

            return null;

        }

        var pos = geometry.attributes.position;

        var arr = pos.array;

        var itemSize = pos.itemSize || 3;

        var nv = pos.count;

        if (!nv || arr.length < nv * itemSize) {

            return null;

        }

        var ymin = Infinity;

        var ymax = -Infinity;

        var i;

        var y;

        for (i = 0; i < nv; i++) {

            y = arr[i * itemSize + 1];

            if (y < ymin) ymin = y;

            if (y > ymax) ymax = y;

        }

        var span = ymax - ymin;

        var band = Math.max(span * 0.42, 0.18);

        var yBotMax = ymin + band;

        var yTopMin = ymax - band;

        function triNormal(ix0, ix1, ix2) {

            var o0 = ix0 * itemSize;

            var o1 = ix1 * itemSize;

            var o2 = ix2 * itemSize;

            var ax = arr[o1] - arr[o0];

            var ay = arr[o1 + 1] - arr[o0 + 1];

            var az = arr[o1 + 2] - arr[o0 + 2];

            var bx = arr[o2] - arr[o0];

            var by = arr[o2 + 1] - arr[o0 + 1];

            var bz = arr[o2 + 2] - arr[o0 + 2];

            var cx = ay * bz - az * by;

            var cy = az * bx - ax * bz;

            var cz = ax * by - ay * bx;

            var len = Math.sqrt(cx * cx + cy * cy + cz * cz);

            if (len < 1e-12) {

                return null;

            }

            return [cx / len, cy / len, cz / len];

        }

        function centroidY(ix0, ix1, ix2) {

            return (

                arr[ix0 * itemSize + 1] +

                arr[ix1 * itemSize + 1] +

                arr[ix2 * itemSize + 1]

            ) / 3;

        }

        function accumulateBand(bottomBand) {

            var nx = 0;

            var ny = 0;

            var nz = 0;

            var triCount = 0;

            var indexAttr = geometry.index;

            var ii;

            if (indexAttr && indexAttr.array && indexAttr.array.length >= 3) {

                var ia = indexAttr.array;

                for (ii = 0; ii + 2 < ia.length; ii += 3) {

                    var i0 = ia[ii];

                    var i1 = ia[ii + 1];

                    var i2 = ia[ii + 2];

                    var cy = centroidY(i0, i1, i2);

                    if (bottomBand) {

                        if (cy > yBotMax) continue;

                    } else {

                        if (cy < yTopMin) continue;

                    }

                    var nn = triNormal(i0, i1, i2);

                    if (!nn) continue;

                    nx += nn[0];

                    ny += nn[1];

                    nz += nn[2];

                    triCount++;

                }

            } else {

                for (ii = 0; ii + 2 < nv; ii += 3) {

                    var t0 = ii;

                    var t1 = ii + 1;

                    var t2 = ii + 2;

                    var cy2 = centroidY(t0, t1, t2);

                    if (bottomBand) {

                        if (cy2 > yBotMax) continue;

                    } else {

                        if (cy2 < yTopMin) continue;

                    }

                    var nn2 = triNormal(t0, t1, t2);

                    if (!nn2) continue;

                    nx += nn2[0];

                    ny += nn2[1];

                    nz += nn2[2];

                    triCount++;

                }

            }

            return { nx: nx, ny: ny, nz: nz, triCount: triCount };

        }

        function accToVector(acc) {

            if (!acc || acc.triCount < 1) {

                return null;

            }

            var lenN = Math.sqrt(acc.nx * acc.nx + acc.ny * acc.ny + acc.nz * acc.nz);

            if (lenN < 1e-9) {

                return null;

            }

            var nx = acc.nx / lenN;

            var ny = acc.ny / lenN;

            var nz = acc.nz / lenN;

            if (ny < 0) {

                nx = -nx;

                ny = -ny;

                nz = -nz;

            }

            return new THREE.Vector3(nx, ny, nz);

        }

        var botAcc = accumulateBand(true);

        var topAcc = accumulateBand(false);

        var vBot = accToVector(botAcc);

        var vTop = accToVector(topAcc);

        var chosen = null;

        if (vBot && vTop) {

            chosen = botAcc.triCount >= topAcc.triCount ? vBot : vTop;

        } else {

            chosen = vBot || vTop;

        }

        if (!chosen && geometry.attributes.normal) {

            var nrm = geometry.attributes.normal;

            var nar = nrm.array;

            var sbx = 0;

            var sby = 0;

            var sbz = 0;

            var sbn = 0;

            var stx = 0;

            var sty = 0;

            var stz = 0;

            var stn = 0;

            var j;

            for (j = 0; j < nv; j++) {

                var o = j * itemSize;

                var yv = arr[o + 1];

                var vnx = nar[o];

                var vny = nar[o + 1];

                var vnz = nar[o + 2];

                if (yv <= yBotMax) {

                    sbx += vnx;

                    sby += vny;

                    sbz += vnz;

                    sbn++;

                }

                if (yv >= yTopMin) {

                    stx += vnx;

                    sty += vny;

                    stz += vnz;

                    stn++;

                }

            }

            var fb = sbn >= 3 ? accToVector({ nx: sbx, ny: sby, nz: sbz, triCount: sbn }) : null;

            var ft = stn >= 3 ? accToVector({ nx: stx, ny: sty, nz: stz, triCount: stn }) : null;

            if (fb && ft) {

                chosen = sbn >= stn ? fb : ft;

            } else {

                chosen = fb || ft;

            }

        }

        return chosen;

    }



    function quaternionRotateVectorToVector(a, b) {

        var v1 = a.clone().normalize();

        var v2 = b.clone().normalize();

        var axis = new THREE.Vector3().crossVectors(v1, v2);

        var ax = axis.lengthSq();

        var dot = THREE.MathUtils.clamp(v1.dot(v2), -1, 1);

        if (ax < 1e-10) {

            if (dot > 0.99999) {

                return new THREE.Quaternion();

            }

            var orth = new THREE.Vector3(1, 0, 0);

            if (Math.abs(v1.dot(orth)) > 0.85) {

                orth.set(0, 1, 0);

            }

            axis.crossVectors(v1, orth).normalize();

            return new THREE.Quaternion().setFromAxisAngle(axis, Math.PI);

        }

        axis.normalize();

        return new THREE.Quaternion().setFromAxisAngle(axis, Math.acos(dot));

    }



    function cacheSemanticHorizontalUp(meta) {

        if (

            !meta ||

            !Array.isArray(meta.semantic_horizontal_up) ||

            meta.semantic_horizontal_up.length !== 3

        ) {

            return;

        }

        var a = meta.semantic_horizontal_up;

        var sx = Number(a[0]);

        var sy = Number(a[1]);

        var sz = Number(a[2]);

        if (!Number.isFinite(sx) || !Number.isFinite(sy) || !Number.isFinite(sz)) {

            return;

        }

        lastSemanticHorizontalUp = [sx, sy, sz];

    }



    function resolveCenterRealtimeGeometry() {

        if (

            centerRealtimeMesh &&

            centerRealtimeMesh.geometry &&

            centerRealtimeMesh.geometry.attributes &&

            centerRealtimeMesh.geometry.attributes.position

        ) {

            return centerRealtimeMesh.geometry;

        }

        if (!centerMeshAnchor) {

            return null;

        }

        var found = null;

        centerMeshAnchor.traverse(function (o) {

            if (found) return;

            if (!o.isMesh || !o.geometry || !o.geometry.attributes || !o.geometry.attributes.position) {

                return;

            }

            if (o.userData && o.userData.isCenterRealtimeMesh) {

                found = o.geometry;

            }

        });

        if (found) {

            return found;

        }

        centerMeshAnchor.traverse(function (o) {

            if (found) return;

            if (o.isMesh && o.geometry && o.geometry.attributes && o.geometry.attributes.position) {

                found = o.geometry;

            }

        });

        return found;

    }



    function alignCenterMeshAnchorUpright() {

        if (!centerMeshAnchor || typeof THREE === 'undefined') {

            return false;

        }

        var geom = resolveCenterRealtimeGeometry();

        var nLocal = null;

        if (lastSemanticHorizontalUp && lastSemanticHorizontalUp.length === 3) {

            var sx = Number(lastSemanticHorizontalUp[0]);

            var sy = Number(lastSemanticHorizontalUp[1]);

            var sz = Number(lastSemanticHorizontalUp[2]);

            if (Number.isFinite(sx) && Number.isFinite(sy) && Number.isFinite(sz)) {

                nLocal = new THREE.Vector3(sx, sy, sz);

                if (nLocal.lengthSq() > 1e-14) {

                    nLocal.normalize();

                } else {

                    nLocal = null;

                }

            }

        }

        if (!nLocal && geom) {

            nLocal = estimateFloorUpNormalFromGeometry(geom);

        }

        if (!nLocal) {

            return false;

        }

        var upWorld = new THREE.Vector3(0, 1, 0);

        var nWorld = nLocal.clone().applyQuaternion(centerMeshAnchor.quaternion).normalize();

        var qFix = quaternionRotateVectorToVector(nWorld, upWorld);

        qFix.multiply(centerMeshAnchor.quaternion);

        centerMeshAnchor.quaternion.copy(qFix);

        centerMeshAnchor.quaternion.normalize();

        centerMeshAnchor.updateMatrixWorld(true);

        window.dispatchEvent(new CustomEvent('center-mesh-anchor-changed'));

        return true;

    }



    function lockPlacementFromSamples() {

        var avg = averagePlacement(calibrationSamples);

        if (!avg) return;

        lockedPlacement = avg;

        placementLocked = true;

        window.__centerMeshPlacementLocked = true;

        applyPlacementToAnchor(lockedPlacement);

        if (window.console && console.info) {

            console.info('[CENTER] mesh placement locked (client)', {

                scale: lockedPlacement.scale.toFixed(4),

                position: lockedPlacement.position,

                samples: calibrationSamples.length,

            });

        }

    }



    function applyServerPlacement(apiPlacement) {

        var p = placementFromApiPayload(apiPlacement);

        if (!p) return false;

        lastPlacementMeta = apiPlacement;

        if (apiPlacement.locked) {

            lockedPlacement = p;

            placementLocked = true;

            window.__centerMeshPlacementLocked = true;

            applyPlacementToAnchor(p);

            if (window.console && console.info) {

                console.info('[CENTER] mesh placement locked (server)', {

                    scale: p.scale,

                    position: p.position,

                    extent_m: p.extent_m,

                    depth_extent_m: p.depth_extent_m,

                });

            }

            return true;

        }

        applyPlacementToAnchor(p);

        return false;

    }



    function tryCalibratePlacement(vertices, vertexCount, apiPlacement) {

        if (apiPlacement && applyServerPlacement(apiPlacement)) {

            return;

        }

        if (apiPlacement && apiPlacement.calibrating) {

            lastPlacementMeta = apiPlacement;

        }



        if (placementLocked) return;

        if (vertexCount < MIN_CALIB_VERTICES) return;



        var sample = computePlacementFromVertices(vertices);

        if (!sample) return;



        calibrationSamples.push(sample);

        var lockN = meshPlacementCfg.lockSamples;

        if (calibrationSamples.length > lockN) {

            calibrationSamples.shift();

        }



        if (!centerMeshAnchor) {

            ensureMeshAnchor();

        }



        if (calibrationSamples.length >= lockN) {

            lockPlacementFromSamples();

        } else {

            applyPlacementToAnchor(sample);

        }

    }



    function ensureMeshAnchor() {

        if (centerMeshAnchor || !window.scene || !window.THREE) return;

        centerMeshAnchor = new THREE.Group();

        centerMeshAnchor.name = 'CENTER_MESH_ANCHOR';

        centerMeshAnchor.userData.isCenterMeshAnchor = true;

        window.scene.add(centerMeshAnchor);

    }



    var imuPipelineEnsurePromise = null;



    /**

     * AR/VR 进入时仅拉起 IMU 位姿（POST start-imu），不启深度相机、不轮询 mesh。

     * 显式「开始扫描」走 startCenterRealtime() → POST start（scanner）。

     */

    function ensureCenterPipelineForImu() {

        if (!isImmersiveControlMode()) return Promise.resolve(null);

        if (imuPipelineEnsurePromise) return imuPipelineEnsurePromise;



        imuPipelineEnsurePromise = fetch(apiUrl('centerStatus', '/api/center/status/'), {

            credentials: 'same-origin',

        })

            .then(function (res) {

                return res.json();

            })

            .then(function (data) {

                if (data && data.depth_capture_enabled) {

                    startCenterStatusPolling();

                    return data;

                }

                if (data && data.imu_only && data.running) {

                    startCenterStatusPolling();

                    return data;

                }

                return postCenterApi(apiUrl('centerStartImu', '/api/center/start-imu/')).then(

                    function (startData) {

                        updateCenterRealtimeStatus(startData);

                        startCenterStatusPolling();

                        return startData;

                    }

                );

            })

            .catch(function (err) {

                updateCenterRealtimeStatus({ ok: false, error: formatCenterFetchError(err) });

                throw err;

            })

            .finally(function () {

                imuPipelineEnsurePromise = null;

            });



        return imuPipelineEnsurePromise;

    }



    function stopImuPipelineIfActive() {

        return fetch(apiUrl('centerStatus', '/api/center/status/'), { credentials: 'same-origin' })

            .then(function (res) {

                return res.json();

            })

            .then(function (data) {

                if (data && data.imu_only && data.running && !data.depth_capture_enabled) {

                    return postCenterApi(apiUrl('centerStopImu', '/api/center/stop-imu/'));

                }

                return data;

            })

            .catch(function () {

                return null;

            });

    }



    function syncCenterServicesForControlMode() {

        if (isImmersiveControlMode()) {

            ensureCenterPipelineForImu().catch(function () {});

            return;

        }

        stopCenterStatusPolling();

        stopImuPipelineIfActive().catch(function () {});

    }



    function clearCenterSceneMeshes() {

        resetCenterMeshSession();

        if (
            window.CenterReconstructionMaterial &&
            typeof window.CenterReconstructionMaterial.clearScene === 'function'
        ) {
            window.CenterReconstructionMaterial.clearScene();
        }

    }



    function startCenterRealtime() {

        clearCenterSceneMeshes();

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



    function stopCenterRealtimeFixAndMaterial() {

        if (centerMeshPollDelayTimer) {

            clearTimeout(centerMeshPollDelayTimer);

            centerMeshPollDelayTimer = null;

        }

        stopCenterMeshPolling();

        stopCenterStatusPolling();

        centerAutoResumePending = false;

        updateCenterRecoveryBanner({ center_recovery_state: 'normal' });

        postCenterApi(apiUrl('centerStop', '/api/center/stop/')).then(function (data) {

            updateCenterRealtimeStatus(data);

            resetCenterMeshSession();

        }).catch(function (err) {

            updateCenterRealtimeStatus({ ok: false, error: formatCenterFetchError(err) });

        });

    }



    function stopCenterRealtimeAndClear() {

        if (centerMeshPollDelayTimer) {

            clearTimeout(centerMeshPollDelayTimer);

            centerMeshPollDelayTimer = null;

        }

        stopCenterMeshPolling();

        stopCenterStatusPolling();

        centerAutoResumePending = false;

        updateCenterRecoveryBanner({ center_recovery_state: 'normal' });

        var stopUrl = apiUrl('centerStop', '/api/center/stop/');

        var clearUrl = apiUrl('centerClearMaterial', '/api/center/clear-material/');

        postCenterApi(stopUrl)

            .then(function (data) {

                return postCenterApi(clearUrl).then(function (clearData) {

                    return { stop: data, clear: clearData };

                });

            })

            .then(function (bundle) {

                updateCenterRealtimeStatus(bundle.stop);

                clearCenterSceneMeshes();

            })

            .catch(function (err) {

                updateCenterRealtimeStatus({ ok: false, error: formatCenterFetchError(err) });

                clearCenterSceneMeshes();

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

                cacheSemanticHorizontalUp(data);

                if (!data || data.ok === false) return;

                if (data.too_large) return;

                if (!data.mesh) return;

                if (!window.scene || !window.THREE) return;

                if (data.version === centerMeshVersion) return;

                centerMeshVersion = data.version;

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



    function formatExtentTag(ext) {

        if (!ext) return '';

        var x = ext.x != null ? Number(ext.x).toFixed(2) : '?';

        var y = ext.y != null ? Number(ext.y).toFixed(2) : '?';

        var z = ext.z != null ? Number(ext.z).toFixed(2) : '?';

        return x + '×' + y + '×' + z + 'm';

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

            el.textContent = 'CENTER: MESH TOO LARGE ' + (s.vertices || '-') + ' V / ' + (s.faces || '-') + ' F';

            return;

        }

        var summary = data.summary || {};

        var hasMeshMeta = data.version != null || summary.vertices != null || summary.faces != null;

        var running = data.running ? 'RUNNING' : 'IDLE';

        var lockTag = placementLocked ? ' · PLACE LOCKED' : calibrationSamples.length ? ' · CALIBRATING' : '';

        var place = data.placement || lastPlacementMeta;

        if (place) {

            if (place.locked && place.extent_m) {

                lockTag += ' · ' + formatExtentTag(place.extent_m);

            } else if (place.calibrating) {

                lockTag +=

                    ' · depth帧' +

                    (place.depth_samples != null ? place.depth_samples : '?') +

                    '/mesh' +

                    (place.mesh_samples != null ? place.mesh_samples : '?');

            }

        }

        if (!data.mesh && data.message) {

            var line =

                'CENTER: ' + running + ' · ' + data.message + '（点 Start Reconstruct 后等几秒；无 mesh 时 TSDF 尚未出表面）' + lockTag;

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

            return;

        }

        if (!hasMeshMeta) {

            el.textContent = 'CENTER: ' + running + ' · 已启动（mesh 统计见轮询）' + lockTag;

            return;

        }

        var version = data.version != null ? data.version : '-';

        var vCount = summary.vertices != null ? summary.vertices : '-';

        var fCount = summary.faces != null ? summary.faces : '-';

        el.textContent =

            'CENTER: ' + running + ' · V' + version + ' · ' + vCount + ' verts · ' + fCount + ' faces' + lockTag;

    }



    function removeCenterRealtimeMesh() {

        if (!centerRealtimeMesh) return;

        if (centerMeshAnchor) {

            centerMeshAnchor.remove(centerRealtimeMesh);

        } else if (window.scene) {

            window.scene.remove(centerRealtimeMesh);

        }

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



    function buildGeometryFromPayload(meshPayload) {

        var vertices = meshPayload.vertices || [];

        var indices = meshPayload.indices || [];

        var normals = meshPayload.normals || null;

        var colors = meshPayload.colors || null;



        if (!vertices.length) return null;



        var geometry = new THREE.BufferGeometry();

        geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));



        if (indices && indices.length) {

            geometry.setIndex(indices);

        }



        if (normals && normals.length === vertices.length) {

            geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));

        } else {

            geometry.computeVertexNormals();

        }



        if (colors && colors.length === vertices.length) {

            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

        }



        return { geometry: geometry, hasVertexColors: !!(colors && colors.length === vertices.length) };

    }



    function createMeshMaterial(hasVertexColors) {

        if (hasVertexColors) {

            return new THREE.MeshStandardMaterial({

                vertexColors: true,

                roughness: 0.75,

                metalness: 0.05,

                side: THREE.DoubleSide,

            });

        }

        return new THREE.MeshStandardMaterial({

            color: 0xb8d7ff,

            roughness: 0.72,

            metalness: 0.08,

            side: THREE.DoubleSide,

            transparent: true,

            opacity: 0.92,

        });

    }



    function updateExistingMeshGeometry(meshPayload) {

        var built = buildGeometryFromPayload(meshPayload);

        if (!built || !centerRealtimeMesh) return;



        if (centerRealtimeMesh.geometry) {

            centerRealtimeMesh.geometry.dispose();

        }

        centerRealtimeMesh.geometry = built.geometry;



        var wantVc = built.hasVertexColors;

        var mat = centerRealtimeMesh.material;

        if (wantVc && (!mat || !mat.vertexColors)) {

            if (mat && mat.dispose) mat.dispose();

            centerRealtimeMesh.material = createMeshMaterial(true);

        } else if (!wantVc && mat && mat.vertexColors) {

            if (mat.dispose) mat.dispose();

            centerRealtimeMesh.material = createMeshMaterial(false);

        }

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

        var vertexCount = Math.floor(vertices.length / 3);

        if (!vertexCount) {

            console.warn('[CENTER] mesh has no vertices.');

            return;

        }



        tryCalibratePlacement(vertices, vertexCount, meta && meta.placement);

        ensureMeshAnchor();



        if (placementLocked && lockedPlacement) {

            applyPlacementToAnchor(lockedPlacement);

        }



        if (centerRealtimeMesh) {

            updateExistingMeshGeometry(meshPayload);

            return;

        }



        var built = buildGeometryFromPayload(meshPayload);

        if (!built) return;



        var mesh = new THREE.Mesh(built.geometry, createMeshMaterial(built.hasVertexColors));

        mesh.name = 'CENTER_MEMORY_REALTIME_MESH';

        mesh.userData.isCenterRealtimeMesh = true;

        mesh.userData.skipMaterialProtocol = true;

        mesh.userData.coordinateSpace = (meta && meta.coordinate_space) || 'reconstruction_world';

        mesh.renderOrder = 10;

        mesh.scale.set(1, 1, 1);

        mesh.position.set(0, 0, 0);



        centerRealtimeMesh = mesh;

        centerMeshAnchor.add(mesh);

    }



    window.addEventListener('realm-action', function (e) {

        var d = e.detail;

        if (!d) return;

        if (d.action === 'control-mode-changed') {

            syncCenterServicesForControlMode();

            if (placementLocked && lockedPlacement) {

                var p = placementFromApiPayload(lastPlacementMeta) || lockedPlacement;

                if (p) {

                    lockedPlacement = p;

                    applyPlacementToAnchor(p);

                }

            }

        }

        if (d.action === 'scan-start') {

            clearCenterSceneMeshes();

        }

        if (d.action === 'scan-stop') {

            stopCenterRealtimeFixAndMaterial();

        }

        if (d.action === 'scan-stop-clear') {

            stopCenterRealtimeAndClear();

        }

    });



    function onReady() {

        var startBtn = document.getElementById('center-start-btn');

        var fixBtn = document.getElementById('center-fix-material-btn');

        var clearBtn = document.getElementById('center-stop-clear-btn');

        if (startBtn) {

            startBtn.addEventListener('click', function () {

                startCenterRealtime();

            });

        }

        if (fixBtn) {

            fixBtn.addEventListener('click', function () {

                stopCenterRealtimeFixAndMaterial();

            });

        }

        if (clearBtn) {

            clearBtn.addEventListener('click', function () {

                stopCenterRealtimeAndClear();

            });

        }

        syncCenterServicesForControlMode();

    }



    window.CenterRealtimeMesh = {

        resetSession: resetCenterMeshSession,

        clearScene: clearCenterSceneMeshes,

        stopFixAndMaterial: stopCenterRealtimeFixAndMaterial,

        stopAndClear: stopCenterRealtimeAndClear,

        ensurePipelineForImu: ensureCenterPipelineForImu,

        syncForControlMode: syncCenterServicesForControlMode,

        stopImuOnly: stopImuPipelineIfActive,

        startMeshPolling: startCenterMeshPolling,

        isPlacementLocked: function () {

            return placementLocked;

        },

        getAnchor: function () {

            return centerMeshAnchor;

        },

        getPlacementMeta: function () {

            return lastPlacementMeta;

        },

        alignAnchorUprightFromFloor: alignCenterMeshAnchorUpright,



    };



    if (document.readyState === 'loading') {

        document.addEventListener('DOMContentLoaded', onReady);

    } else {

        onReady();

    }

})();


