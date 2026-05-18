/**
 * AR REALM — Desktop Realm MVP（Three.js）
 * 依赖全局 THREE（由 realm.html 先加载 CDN）
 * 路径：static/js/realm/realm-main.js（realm.html 唯一引用）
 */
console.warn('[REALM MAIN] loaded version 2026-05-16-DEBUG');
window.__REALM_MAIN_LOADED_VERSION = '2026-05-16-DEBUG';
console.warn('[REALM MAIN] actual file loaded');
window.__REALM_MAIN_ACTUAL_LOADED = true;

(function () {
    'use strict';

    var bootstrap = window.REALM_BOOTSTRAP || {};
    var realm = bootstrap.realm || {};
    var persistentRoomTemplateForSave = bootstrap.persistentRoomTemplate || null;

    function readEyeHeightM() {
        if (typeof window.readRealmEyeHeightM === 'function') {
            return window.readRealmEyeHeightM();
        }
        var h = bootstrap.eyeHeightM;
        if (typeof h === 'number' && Number.isFinite(h)) return h;
        return 1.7;
    }

    function readRoomHeightM() {
        if (typeof window.readRealmRoomHeightM === 'function') {
            return window.readRealmRoomHeightM();
        }
        var h = bootstrap.roomHeightM;
        if (typeof h === 'number' && Number.isFinite(h)) return h;
        return 6;
    }

    function roomWallCenterY() {
        return readRoomHeightM() * 0.5;
    }

    function applyPlayerEyeHeight(cam) {
        if (!cam) return;
        cam.position.y = readEyeHeightM();
    }

    var PROTOCOL_DEF = {
        cyber_neon: { name: '赛博霓虹', wall: 0x102033, floor: 0x071822, emissive: 0x00f2ff },
        wasteland_rust: { name: '废土铁锈', wall: 0x5a3824, floor: 0x2e2219, emissive: 0xff7a1a },
        starship_alloy: { name: '星舰合金', wall: 0x5a6578, floor: 0x2d3540, emissive: 0x99ccff },
        magic_stone: { name: '魔法石墙', wall: 0x312448, floor: 0x1c1530, emissive: 0xbc00ff },
        forest_temple: { name: '森林神殿', wall: 0x1a3d2a, floor: 0x0f2418, emissive: 0x00ff88 },
        deep_sea: { name: '深海基地', wall: 0x0a2540, floor: 0x051828, emissive: 0x0088cc },
        pixel_retro: { name: '像素复古', wall: 0x3a2d50, floor: 0x201830, emissive: 0xff00aa },
        hacker_matrix: { name: '黑客矩阵', wall: 0x0d280d, floor: 0x061806, emissive: 0x00ff00 },
    };

    var LIGHTING_DEF = {
        cyan_cold: { fog: 0x050608, ambient: 0x404850, point: 0x00f2ff },
        purple_neon: { fog: 0x0a0612, ambient: 0x503850, point: 0xbc00ff },
        warm_amber: { fog: 0x100a06, ambient: 0x605040, point: 0xffaa55 },
        alert_red: { fog: 0x120505, ambient: 0x604040, point: 0xff3333 },
        moonlit: { fog: 0x080a10, ambient: 0x505868, point: 0xccddee },
    };

    var ROOM_NAMES = {
        basic_room: '基础校准房',
        bedroom_small: '旧版小房间',
        living_room: '客厅',
        lab: '实验室',
        underground_bunker: '地下基地',
        starship_cabin: '星舰舱室',
        wasteland_safehouse: '废土安全屋',
        cyber_apartment: '赛博公寓',
        magic_library: '魔法图书馆',
    };

    var scene, camera, renderer;
    var roomGroup;
    var roomObjects = {
        floors: [],
        walls: [],
        ceilings: [],
        trims: [],
        emissives: [],
        glass: [],
        props: [],
        scanGhosts: [],
        ambient: null,
        pointMain: null,
        pointBack: null,
    };
    var cameraMoveBounds = { xMin: -5.35, xMax: 5.35, zMin: -5.35, zMax: 5.35 };
    var activeRoomTemplateKey = 'basic_room';

    function normalizeRoomTemplate(template) {
        var t = template || 'basic_room';
        var aliases = { bedroom_small: 'basic_room' };
        return aliases[t] || t;
    }

    var keys = {};
    var yaw = 0;
    var pitch = 0;

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

    function isDesktopControl() {
        return getControlMode() === 'desktop';
    }

    function isMovementKey(code) {
        return (
            code === 'KeyW' ||
            code === 'KeyA' ||
            code === 'KeyS' ||
            code === 'KeyD' ||
            code === 'ArrowUp' ||
            code === 'ArrowDown' ||
            code === 'ArrowLeft' ||
            code === 'ArrowRight' ||
            code === 'ShiftLeft'
        );
    }

    function shouldIgnoreKeyTarget(target) {
        if (!target || !target.closest) return false;
        return !!target.closest('input, textarea, select, [contenteditable="true"]');
    }

    function isWasdKey(code) {
        return code === 'KeyW' || code === 'KeyA' || code === 'KeyS' || code === 'KeyD';
    }

    function clearMovementKeys() {
        keys.KeyW = false;
        keys.KeyA = false;
        keys.KeyS = false;
        keys.KeyD = false;
        keys.ArrowUp = false;
        keys.ArrowDown = false;
        keys.ArrowLeft = false;
        keys.ArrowRight = false;
        keys.ShiftLeft = false;
    }

    function releaseDesktopPointer() {
        clearMovementKeys();
        if (document.pointerLockElement) {
            document.exitPointerLock();
        }
    }

    var state = {
        material_pack: realm.material_pack || 'cyber_neon',
        lighting: realm.lighting || 'cyan_cold',
        room_template: normalizeRoomTemplate(realm.room_template || 'basic_room'),
        decorations: Array.isArray(realm.decorations) ? realm.decorations : [],
    };

    var reported = {
        moved: false,
        pointerLocked: false,
        lookMoved: false,
    };

    var runtimeMode = bootstrap.mode || 'realm';

    var realmDebugEnabled = false;
    try {
        realmDebugEnabled =
            new URLSearchParams(window.location.search).get('debug') === '1' &&
            !!bootstrap.debugAllowed;
    } catch (eDbg) {}
    var realmDebugLastLog = 0;

    var training = {
        enabled: false,
        currentStep: 0,
        completed: false,
        targetRing: null,
        node: null,
        steps: [
            { key: 'pointer', title: '锁定视角', desc: '点击画布进入指针锁定模式，完成视觉同步。' },
            { key: 'move', title: '移动校准', desc: '使用 WASD 移动到前方绿色光圈。' },
            { key: 'look', title: '视角同步', desc: '移动鼠标，看向训练传送门。' },
            { key: 'interact', title: '节点交互', desc: '靠近紫色训练节点并按 E 激活。' },
            { key: 'material', title: '材质覆写', desc: '按 1–8 或点击 Dock 切换任意材质协议。' },
            { key: 'save', title: '保存位面', desc: '点击保存位面，完成接入训练。' },
        ],
    };

    function makeMat(color, emissive, emissiveIntensity) {
        return new THREE.MeshStandardMaterial({
            color: color,
            roughness: 0.62,
            metalness: 0.38,
            emissive: emissive,
            emissiveIntensity: emissiveIntensity,
        });
    }

    function applyLighting() {
        var L = LIGHTING_DEF[state.lighting] || LIGHTING_DEF.cyan_cold;
        if (roomObjects.ambient) {
            roomObjects.ambient.color.setHex(L.ambient);
        }
        if (roomObjects.pointMain) {
            roomObjects.pointMain.color.setHex(L.point);
        }
        syncSceneFog();
    }

    function syncSceneFog() {
        if (!scene) return;
        var L = LIGHTING_DEF[state.lighting] || LIGHTING_DEF.cyan_cold;
        var near = 12;
        var far = 55;
        if (activeRoomTemplateKey === 'starship_cabin') {
            near = 16;
            far = 90;
        } else if (activeRoomTemplateKey === 'cyber_apartment') {
            near = 14;
            far = 64;
        }
        scene.fog = new THREE.Fog(L.fog, near, far);
        if (scene.fog) scene.fog.color.setHex(L.fog);
    }

    function createLights() {
        var L = LIGHTING_DEF[state.lighting] || LIGHTING_DEF.cyan_cold;
        syncSceneFog();

        var ambient = new THREE.AmbientLight(L.ambient, 0.55);
        scene.add(ambient);
        roomObjects.ambient = ambient;

        var point = new THREE.PointLight(L.point, 1.8, 28);
        point.position.set(0, 3.2, 2);
        scene.add(point);
        roomObjects.pointMain = point;

        var back = new THREE.PointLight(0xbc00ff, 0.9, 22);
        back.position.set(0, 2, -5);
        scene.add(back);
        roomObjects.pointBack = back;
    }

    function resetRoomObjectBuckets() {
        roomObjects.floors = [];
        roomObjects.walls = [];
        roomObjects.ceilings = [];
        roomObjects.trims = [];
        roomObjects.emissives = [];
        roomObjects.glass = [];
        roomObjects.props = [];
        roomObjects.scanGhosts = [];
    }

    function clearRoomGeometry() {
        if (!roomGroup) return;
        var ch = roomGroup.children.slice();
        ch.forEach(function (obj) {
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) {
                if (Array.isArray(obj.material)) {
                    obj.material.forEach(function (m) {
                        if (m && m.dispose) m.dispose();
                    });
                } else if (obj.material.dispose) obj.material.dispose();
            }
            roomGroup.remove(obj);
        });
        resetRoomObjectBuckets();
    }

    function addMeshToRoom(mesh, bucket) {
        roomGroup.add(mesh);
        if (roomObjects[bucket]) roomObjects[bucket].push(mesh);
    }

    function addFloor(width, depth, cx, cz) {
        var geo = new THREE.PlaneGeometry(width, depth);
        var mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: 0x1a1f26 }));
        mesh.rotation.x = -Math.PI / 2;
        mesh.position.set(cx || 0, 0, cz || 0);
        addMeshToRoom(mesh, 'floors');
        return mesh;
    }

    function addWall(w, h, px, py, pz, rotationY) {
        var geo = new THREE.PlaneGeometry(w, h);
        var mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: 0x252b35 }));
        mesh.position.set(px, py, pz);
        mesh.rotation.y = rotationY || 0;
        addMeshToRoom(mesh, 'walls');
        return mesh;
    }

    function addCeiling(width, depth, y, cx, cz) {
        var geo = new THREE.PlaneGeometry(width, depth);
        var mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: 0x1a1f26 }));
        mesh.rotation.x = Math.PI / 2;
        mesh.position.set(cx || 0, y != null ? y : readRoomHeightM(), cz || 0);
        addMeshToRoom(mesh, 'ceilings');
        return mesh;
    }

    function addTrimBox(sx, sy, sz, px, py, pz, colorHex, bucket) {
        bucket = bucket || 'trims';
        var geo = new THREE.BoxGeometry(sx, sy, sz);
        var mat = new THREE.MeshStandardMaterial({
            color: colorHex,
            emissive: colorHex,
            emissiveIntensity: 0.85,
            metalness: 0.35,
            roughness: 0.4,
        });
        var mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(px, py, pz);
        addMeshToRoom(mesh, bucket);
        return mesh;
    }

    function createPortalAt(ringZ) {
        var portalY = Math.min(readEyeHeightM(), readRoomHeightM() * 0.45);
        var ringGeo = new THREE.TorusGeometry(1.1, 0.05, 16, 100);
        var ringMat = new THREE.MeshStandardMaterial({
            color: 0x00f2ff,
            emissive: 0x00f2ff,
            emissiveIntensity: 1.4,
        });
        var ring = new THREE.Mesh(ringGeo, ringMat);
        ring.position.set(0, portalY, ringZ);
        addMeshToRoom(ring, 'trims');

        var coreGeo = new THREE.CircleGeometry(0.95, 64);
        var coreMat = new THREE.MeshBasicMaterial({
            color: 0x00f2ff,
            transparent: true,
            opacity: 0.15,
            side: THREE.DoubleSide,
        });
        var core = new THREE.Mesh(coreGeo, coreMat);
        core.position.set(0, portalY, ringZ + 0.02);
        core.userData.skipMaterialProtocol = true;
        addMeshToRoom(core, 'props');
    }

    function createBasicRoomLayout() {
        var wallH = readRoomHeightM();
        var wallY = roomWallCenterY();
        var pillarH = Math.max(2.5, wallH - 0.4);
        var pillarY = pillarH * 0.5;

        addFloor(12, 12, 0, 0);
        addWall(12, wallH, 0, wallY, -6, 0);
        addWall(12, wallH, -6, wallY, 0, Math.PI / 2);
        addWall(12, wallH, 6, wallY, 0, -Math.PI / 2);
        addCeiling(12, 12, wallH, 0, 0);

        var grid = new THREE.GridHelper(12, 24, 0x00f2ff, 0x153344);
        grid.position.y = 0.01;
        grid.userData.skipMaterialProtocol = true;
        addMeshToRoom(grid, 'props');

        var pillarMat = new THREE.MeshStandardMaterial({
            color: 0x222831,
            metalness: 0.6,
            roughness: 0.35,
            emissive: 0x001020,
            emissiveIntensity: 0.2,
        });
        [[-4, pillarY, -2], [4, pillarY, -2], [-4, pillarY, 2], [4, pillarY, 2]].forEach(function (p) {
            var geo = new THREE.CylinderGeometry(0.12, 0.15, pillarH, 12);
            var m = new THREE.Mesh(geo, pillarMat.clone());
            m.position.set(p[0], p[1], p[2]);
            addMeshToRoom(m, 'walls');
        });

        createPortalAt(-5.85);
        cameraMoveBounds = { xMin: -5.35, xMax: 5.35, zMin: -5.35, zMax: 5.35 };
    }

    var CYBER_WINDOW_LIGHTS = [
        [-2.5, 1.35, 0.12, 0.35, 0x00f2ff],
        [-1.1, 1.9, 0.1, 0.55, 0xbc00ff],
        [0.2, 1.45, 0.14, 0.42, 0x00f2ff],
        [1.4, 2.05, 0.11, 0.38, 0xbc00ff],
        [2.3, 1.55, 0.13, 0.48, 0x00f2ff],
        [-2.1, 2.35, 0.09, 0.28, 0xbc00ff],
        [-0.4, 2.6, 0.12, 0.33, 0x00f2ff],
        [1.0, 2.45, 0.1, 0.4, 0xbc00ff],
        [2.6, 2.2, 0.11, 0.36, 0x00f2ff],
        [-1.8, 1.2, 0.15, 0.5, 0xbc00ff],
        [0.6, 1.25, 0.1, 0.44, 0x00f2ff],
        [2.0, 1.65, 0.12, 0.32, 0xbc00ff],
        [-2.8, 2.75, 0.08, 0.26, 0x00f2ff],
        [1.6, 1.3, 0.13, 0.52, 0xbc00ff],
        [-0.9, 2.85, 0.1, 0.3, 0x00f2ff],
        [2.4, 2.9, 0.09, 0.27, 0xbc00ff],
    ];

    function createCyberApartmentRoom() {
        var wallH = readRoomHeightM();
        var wallY = roomWallCenterY();
        var trimY = wallH * 0.7;

        addFloor(14, 10, 0, 0);
        addCeiling(14, 10, wallH, 0, 0);

        addWall(3.8, wallH, -5.1, wallY, -5, 0);
        addWall(3.8, wallH, 5.1, wallY, -5, 0);
        addWall(10, wallH, -7, wallY, 0, Math.PI / 2);
        addWall(10, wallH, 7, wallY, 0, -Math.PI / 2);

        var winGeo = new THREE.PlaneGeometry(6, wallH * 0.65);
        var winMat = new THREE.MeshBasicMaterial({
            color: 0x00aaff,
            transparent: true,
            opacity: 0.22,
            side: THREE.DoubleSide,
        });
        var win = new THREE.Mesh(winGeo, winMat);
        win.position.set(0, wallH * 0.58, -5.02);
        addMeshToRoom(win, 'glass');

        CYBER_WINDOW_LIGHTS.forEach(function (row) {
            var x = row[0];
            var y = row[1];
            var w = row[2];
            var h = row[3];
            var c = row[4];
            addTrimBox(w, h, 0.02, x, y, -5.08, c, 'emissives');
        });

        var platformGeo = new THREE.BoxGeometry(4, 0.3, 2);
        var platform = new THREE.Mesh(
            platformGeo,
            new THREE.MeshStandardMaterial({ color: 0x2a3040, metalness: 0.45, roughness: 0.42 })
        );
        platform.position.set(-3.5, 0.15, 2.4);
        addMeshToRoom(platform, 'props');

        addTrimBox(0.08, 0.08, 10, -6.92, trimY, 0, 0x00f2ff, 'trims');
        addTrimBox(0.08, 0.08, 10, 6.92, trimY, 0, 0xbc00ff, 'trims');

        var grid = new THREE.GridHelper(14, 28, 0x00f2ff, 0x153344);
        grid.position.y = 0.01;
        grid.userData.skipMaterialProtocol = true;
        addMeshToRoom(grid, 'props');

        createPortalAt(-5.72);
        cameraMoveBounds = { xMin: -6.35, xMax: 6.35, zMin: -4.35, zMax: 4.35 };
    }

    function createStarshipCabinRoom() {
        var wallH = readRoomHeightM();
        var wallY = roomWallCenterY();
        var trimY = wallH * 0.675;
        var doorY = wallH * 0.5;

        addFloor(8, 16, 0, 0);
        addCeiling(8, 16, wallH, 0, 0);

        addWall(16, wallH, -4, wallY, 0, Math.PI / 2);
        addWall(16, wallH, 4, wallY, 0, -Math.PI / 2);
        addWall(8, wallH, 0, wallY, -8, 0);

        var doorRingGeo = new THREE.TorusGeometry(1.05, 0.06, 16, 80);
        var doorRingMat = new THREE.MeshStandardMaterial({
            color: 0x99ccff,
            emissive: 0x99ccff,
            emissiveIntensity: 1.0,
        });
        var doorRing = new THREE.Mesh(doorRingGeo, doorRingMat);
        doorRing.position.set(0, doorY, -7.92);
        addMeshToRoom(doorRing, 'trims');

        var doorCoreGeo = new THREE.CircleGeometry(0.95, 64);
        var doorCoreMat = new THREE.MeshBasicMaterial({
            color: 0x223344,
            transparent: true,
            opacity: 0.55,
            side: THREE.DoubleSide,
        });
        var doorCore = new THREE.Mesh(doorCoreGeo, doorCoreMat);
        doorCore.position.set(0, doorY, -7.9);
        doorCore.userData.skipMaterialProtocol = true;
        addMeshToRoom(doorCore, 'props');

        var z;
        for (z = -6; z <= 6; z += 3) {
            addTrimBox(0.08, 0.08, 1.5, -3.92, trimY, z, 0x99ccff, 'trims');
            addTrimBox(0.08, 0.08, 1.5, 3.92, trimY, z, 0x99ccff, 'trims');
        }

        addTrimBox(0.12, 0.03, 13, 0, 0.035, 0, 0x99ccff, 'trims');

        var grid = new THREE.GridHelper(8, 16, 0x99ccff, 0x243040);
        grid.scale.z = 2;
        grid.position.y = 0.01;
        grid.userData.skipMaterialProtocol = true;
        addMeshToRoom(grid, 'props');

        createPortalAt(-7.86);
        cameraMoveBounds = { xMin: -3.35, xMax: 3.35, zMin: -7.65, zMax: 7.65 };
    }

    function setDefaultCameraForTemplate(key) {
        var eyeY = readEyeHeightM();
        if (key === 'starship_cabin') {
            camera.position.set(0, eyeY, 6.5);
        } else if (key === 'cyber_apartment') {
            camera.position.set(0, eyeY, 3.8);
        } else {
            camera.position.set(0, eyeY, 4.2);
        }
    }

    function adjustLightsForRoom(key) {
        if (!roomObjects.pointBack || !roomObjects.pointMain) return;
        var h = readRoomHeightM();
        var mid = h * 0.5;
        var high = h * 0.8;
        if (key === 'starship_cabin') {
            roomObjects.pointBack.position.set(0, mid, -7);
            roomObjects.pointMain.position.set(0, high, 0);
        } else if (key === 'cyber_apartment') {
            roomObjects.pointBack.position.set(0, mid, -4.5);
            roomObjects.pointMain.position.set(0, high, 1);
        } else {
            roomObjects.pointBack.position.set(0, mid, -5);
            roomObjects.pointMain.position.set(0, high, 2);
        }
    }

    function addScanGhostObject(obj) {
        if (!roomGroup || !obj) return;
        obj.userData.skipMaterialProtocol = true;
        obj.userData.isScanGhost = true;
        roomGroup.add(obj);
        roomObjects.scanGhosts.push(obj);
    }

    function getBackWallZForTemplate() {
        if (activeRoomTemplateKey === 'starship_cabin') return -7.85;
        if (activeRoomTemplateKey === 'cyber_apartment') return -4.95;
        return -5.95;
    }

    function vec3FromArr(arr, def) {
        def = def || [0, 0, 0];
        if (!Array.isArray(arr) || arr.length < 3) {
            return new THREE.Vector3(Number(def[0]) || 0, Number(def[1]) || 0, Number(def[2]) || 0);
        }
        return new THREE.Vector3(
            Number(arr[0]) || 0,
            Number(arr[1]) || 0,
            Number(arr[2]) || 0
        );
    }

    function orientPlaneMeshToNormal(mesh, normal) {
        var n = normal.clone();
        if (n.lengthSq() < 1e-10) {
            n.set(0, 0, 1);
        } else {
            n.normalize();
        }
        var defaultN = new THREE.Vector3(0, 0, 1);
        mesh.quaternion.setFromUnitVectors(defaultN, n);
    }

    function createMetricScanGhost(surface, index) {
        var center = vec3FromArr(surface.center, [0, 0.03, 1.5]);
        var sz = surface.size;
        var w = 1.2;
        var h = 0.9;
        var depth = 0.65;
        if (Array.isArray(sz) && sz.length >= 2) {
            w = Math.min(12, Math.max(0.15, Math.abs(Number(sz[0])) || w));
            h = Math.min(8, Math.max(0.15, Math.abs(Number(sz[1])) || h));
        }
        if (Array.isArray(sz) && sz.length >= 3) {
            depth = Math.min(6, Math.max(0.1, Math.abs(Number(sz[2])) || depth));
        }
        var t = surface.type;

        if (t === 'floor') {
            var geoF = new THREE.PlaneGeometry(w, h);
            var matF = new THREE.MeshBasicMaterial({
                color: 0x00ff88,
                transparent: true,
                opacity: 0.26,
                side: THREE.DoubleSide,
            });
            var meshF = new THREE.Mesh(geoF, matF);
            var nFloor = vec3FromArr(surface.normal, [0, 1, 0]);
            orientPlaneMeshToNormal(meshF, nFloor);
            meshF.position.set(center.x, Math.max(0.02, center.y + 0.02), center.z);
            addScanGhostObject(meshF);
            var egF = new THREE.EdgesGeometry(geoF);
            var lnF = new THREE.LineSegments(egF, new THREE.LineBasicMaterial({ color: 0x00ff88 }));
            lnF.position.copy(meshF.position);
            lnF.quaternion.copy(meshF.quaternion);
            addScanGhostObject(lnF);
        } else if (t === 'wall') {
            var geoW = new THREE.PlaneGeometry(w, h);
            var matW = new THREE.MeshBasicMaterial({
                color: 0x00f2ff,
                transparent: true,
                opacity: 0.24,
                side: THREE.DoubleSide,
            });
            var meshW = new THREE.Mesh(geoW, matW);
            meshW.position.copy(center);
            var nWall = vec3FromArr(surface.normal, [0, 0, 1]);
            orientPlaneMeshToNormal(meshW, nWall);
            addScanGhostObject(meshW);
            var egW = new THREE.EdgesGeometry(geoW);
            var lnW = new THREE.LineSegments(egW, new THREE.LineBasicMaterial({ color: 0x00f2ff }));
            lnW.position.copy(meshW.position);
            lnW.quaternion.copy(meshW.quaternion);
            addScanGhostObject(lnW);
        } else {
            var geoB = new THREE.BoxGeometry(w, h, depth);
            var matB = new THREE.MeshBasicMaterial({
                color: 0xbc00ff,
                wireframe: true,
                transparent: true,
                opacity: 0.75,
            });
            var meshB = new THREE.Mesh(geoB, matB);
            meshB.position.copy(center);
            addScanGhostObject(meshB);
        }
    }

    function createFloorScanGhost(surface, index) {
        var b = surface.bounds || {};
        var bw = typeof b.w === 'number' ? b.w : 0.5;
        var bh = typeof b.h === 'number' ? b.h : 0.26;
        var w = 1.2 + bw * 4;
        var d = 0.7 + bh * 3;
        var geo = new THREE.PlaneGeometry(w, d);
        var mat = new THREE.MeshBasicMaterial({
            color: 0x00ff88,
            transparent: true,
            opacity: 0.22,
            side: THREE.DoubleSide,
        });
        var mesh = new THREE.Mesh(geo, mat);
        mesh.rotation.x = -Math.PI / 2;
        mesh.position.set(-2 + index * 1.4, 0.035, 1.2);
        addScanGhostObject(mesh);

        var edgeGeo = new THREE.EdgesGeometry(geo);
        var line = new THREE.LineSegments(
            edgeGeo,
            new THREE.LineBasicMaterial({ color: 0x00ff88 })
        );
        line.rotation.copy(mesh.rotation);
        line.position.copy(mesh.position);
        addScanGhostObject(line);
    }

    function createWallScanGhost(surface, index) {
        var b = surface.bounds || {};
        var bw = typeof b.w === 'number' ? b.w : 0.5;
        var bh = typeof b.h === 'number' ? b.h : 0.3;
        var w = 1.6 + bw * 3;
        var h = 0.7 + bh * 2;
        var geo = new THREE.PlaneGeometry(w, h);
        var mat = new THREE.MeshBasicMaterial({
            color: 0x00f2ff,
            transparent: true,
            opacity: 0.2,
            side: THREE.DoubleSide,
        });
        var mesh = new THREE.Mesh(geo, mat);
        var zBack = getBackWallZForTemplate();
        mesh.position.set(-2 + index * 1.35, 1.75, zBack + 0.04);
        addScanGhostObject(mesh);

        var edgeGeo = new THREE.EdgesGeometry(geo);
        var line = new THREE.LineSegments(
            edgeGeo,
            new THREE.LineBasicMaterial({ color: 0x00f2ff })
        );
        line.position.copy(mesh.position);
        addScanGhostObject(line);
    }

    function createObjectScanGhost(surface, index) {
        var geo = new THREE.BoxGeometry(1.0, 0.8, 0.6);
        var mat = new THREE.MeshBasicMaterial({
            color: 0xbc00ff,
            wireframe: true,
            transparent: true,
            opacity: 0.75,
        });
        var mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(2 - index * 0.75, 0.75, 1.0);
        addScanGhostObject(mesh);
    }

    function createScreenMappedScanGhost(surface, index) {
        var t = surface.type;
        if (t === 'floor') {
            createFloorScanGhost(surface, index);
        } else if (t === 'wall') {
            createWallScanGhost(surface, index);
        } else {
            createObjectScanGhost(surface, index);
        }
    }

    function createScanGhosts() {
        var scan = realm.last_reality_scan;
        if (!scan || !Array.isArray(scan.surfaces) || !scan.surfaces.length) return;

        var coordinateSpace = scan.coordinate_space || 'screen_normalized';
        scan.surfaces.forEach(function (surface, index) {
            if (coordinateSpace === 'metric_room') {
                createMetricScanGhost(surface, index);
            } else {
                createScreenMappedScanGhost(surface, index);
            }
        });
    }

    function removeScanGhostsOnly() {
        if (!roomGroup) return;
        roomObjects.scanGhosts.forEach(function (obj) {
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) {
                if (Array.isArray(obj.material)) {
                    obj.material.forEach(function (m) {
                        if (m && m.dispose) m.dispose();
                    });
                } else if (obj.material.dispose) obj.material.dispose();
            }
            roomGroup.remove(obj);
        });
        roomObjects.scanGhosts = [];
    }

    function refreshScanGhosts() {
        removeScanGhostsOnly();
        createScanGhosts();
    }

    function createRoom() {
        clearRoomGeometry();
        var templateKey = normalizeRoomTemplate(state.room_template);
        if (templateKey === 'cyber_apartment') {
            createCyberApartmentRoom();
        } else if (templateKey === 'starship_cabin') {
            createStarshipCabinRoom();
        } else {
            createBasicRoomLayout();
        }
        activeRoomTemplateKey =
            templateKey === 'cyber_apartment' || templateKey === 'starship_cabin'
                ? templateKey
                : 'basic_room';
        setDefaultCameraForTemplate(activeRoomTemplateKey);
        adjustLightsForRoom(activeRoomTemplateKey);
        syncSceneFog();
        createScanGhosts();
    }

    function updateMaterialDockActive(code) {
        document.querySelectorAll('[data-material]').forEach(function (btn) {
            btn.classList.toggle('active', btn.getAttribute('data-material') === code);
        });
        var moreRow = document.getElementById('more-protocols-row');
        var moreToggle = document.getElementById('toggle-more-protocols');
        if (!moreRow || !moreToggle) return;
        var activeInMore = moreRow.querySelector('[data-material="' + code + '"]');
        if (activeInMore) {
            moreRow.classList.remove('collapsed');
            moreToggle.textContent = 'LESS PROTOCOLS ▴';
        }
    }

    function applyRoomMaterials(def) {
        var wallMat = makeMat(def.wall, def.emissive, 0.09);
        var floorMat = makeMat(def.floor, def.emissive, 0.14);
        var ceilMat = makeMat(def.wall, def.emissive, 0.05);
        var trimMat = makeMat(def.emissive, def.emissive, 0.85);

        roomObjects.floors.forEach(function (obj) {
            obj.material = floorMat.clone();
        });
        roomObjects.walls.forEach(function (obj) {
            obj.material = wallMat.clone();
        });
        roomObjects.ceilings.forEach(function (obj) {
            obj.material = ceilMat.clone();
        });
        roomObjects.trims.forEach(function (obj) {
            obj.material = trimMat.clone();
        });
        roomObjects.emissives.forEach(function (obj) {
            obj.material = trimMat.clone();
        });
        roomObjects.glass.forEach(function (obj) {
            if (!obj.material) return;
            var m = obj.material;
            var prevOp = typeof m.opacity === 'number' && !isNaN(m.opacity) ? m.opacity : 0.22;
            var wasTransparent = m.transparent === true;
            var tint = new THREE.Color(def.emissive);
            var base = m.color ? m.color.clone() : new THREE.Color(0x4488cc);
            base.lerp(tint, 0.4);
            if (m.color) m.color.copy(base);
            m.opacity = prevOp;
            m.transparent = wasTransparent || prevOp < 0.999;
        });
        roomObjects.props.forEach(function (obj) {
            if (
                obj.isMesh &&
                obj.material &&
                obj.material.isMeshStandardMaterial &&
                !obj.userData.skipMaterialProtocol
            ) {
                obj.material = wallMat.clone();
            }
        });
    }

    function applyMaterialPack(code, options) {
        options = options || {};
        var def = PROTOCOL_DEF[code] || PROTOCOL_DEF.cyber_neon;
        state.material_pack = code in PROTOCOL_DEF ? code : 'cyber_neon';

        applyRoomMaterials(def);

        updateHud();
        window.dispatchEvent(
            new CustomEvent('realm-action', {
                detail: { action: 'material-protocol-changed', material: state.material_pack },
            })
        );
        updateMaterialDockActive(state.material_pack);
        if (!options.silent) {
            reportProgressEvent('material_changed', { material_pack: state.material_pack });
            completeTrainingStep('material');
        }
    }

    function updateHud() {
        var p = PROTOCOL_DEF[state.material_pack] || PROTOCOL_DEF.cyber_neon;
        var elM = document.getElementById('realm-hud-material');
        if (elM) elM.textContent = 'MATERIAL: ' + p.name + ' (' + state.material_pack + ')';

        var elL = document.getElementById('realm-hud-lighting');
        if (elL) elL.textContent = 'LIGHTING: ' + state.lighting;

        var elR = document.getElementById('realm-hud-room');
        if (elR) {
            elR.textContent =
                'ROOM: ' + (ROOM_NAMES[state.room_template] || state.room_template);
        }
        updateScanHud();
    }

    function updateScanHud() {
        var el = document.getElementById('realm-hud-scan');
        if (!el) return;
        var scan = realm.last_reality_scan;
        if (!scan || !Array.isArray(scan.surfaces)) {
            el.textContent = 'SCAN: NONE';
            updateGhostLegend();
            return;
        }
        var n = scan.surfaces.length;
        var source = String(scan.source || scan.mode || 'unknown')
            .toUpperCase()
            .replace(/_/g, ' ');
        var coord = String(scan.coordinate_space || 'screen_normalized').toUpperCase();
        el.textContent = 'SCAN: ' + n + ' / ' + source + ' / ' + coord;
        updateGhostLegend();
    }

    function updateGhostLegend() {
        var el = document.getElementById('realm-hud-ghost-legend');
        if (!el) return;
        var scan = realm.last_reality_scan;
        if (!scan || !Array.isArray(scan.surfaces) || !scan.surfaces.length) {
            el.textContent = '';
            el.style.display = 'none';
            return;
        }
        el.style.display = 'block';
        el.textContent =
            'SCAN GHOSTS · GREEN=FLOOR · CYAN=WALL · PURPLE=OBJECT';
    }

    function onResize() {
        if (!camera || !renderer) return;
        var w = window.innerWidth;
        var h = window.innerHeight;
        if (h < 1) return;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    }

    function getCsrfToken() {
        var inp = document.querySelector('input[name="csrfmiddlewaretoken"]');
        if (inp && inp.value) return inp.value;
        var m = document.cookie.match(/csrftoken=([^;]+)/);
        return m ? decodeURIComponent(m[1]) : '';
    }

    function showGameToast(type, title, desc) {
        var wrap = document.getElementById('game-toast-stack');
        if (!wrap) {
            wrap = document.createElement('div');
            wrap.id = 'game-toast-stack';
            wrap.style.cssText =
                'position:fixed;right:30px;top:90px;z-index:99999;display:flex;flex-direction:column;gap:12px;pointer-events:none;';
            document.body.appendChild(wrap);
        }
        var toast = document.createElement('div');
        toast.style.cssText =
            'width:320px;padding:16px 18px;border:1px solid #ffd700;background:rgba(10,12,18,0.94);box-shadow:0 0 25px rgba(255,215,0,0.25);color:#fff;font-family:Rajdhani,sans-serif;transform:translateX(30px);opacity:0;transition:0.35s ease;';

        var typeEl = document.createElement('div');
        typeEl.style.cssText =
            'font-family:Syncopate,sans-serif;color:#ffd700;font-size:10px;letter-spacing:2px;margin-bottom:8px;';
        typeEl.textContent = type;

        var titleEl = document.createElement('div');
        titleEl.style.cssText = 'font-size:18px;font-weight:700;margin-bottom:6px;';
        titleEl.textContent = title;

        var descEl = document.createElement('div');
        descEl.style.cssText = 'font-size:13px;color:#aaa;line-height:1.5;';
        descEl.textContent = desc || '';

        toast.appendChild(typeEl);
        toast.appendChild(titleEl);
        toast.appendChild(descEl);
        wrap.appendChild(toast);

        requestAnimationFrame(function () {
            toast.style.opacity = '1';
            toast.style.transform = 'translateX(0)';
        });
        setTimeout(function () {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(30px)';
            setTimeout(function () {
                if (toast.parentNode) toast.parentNode.removeChild(toast);
            }, 400);
        }, 3600);
    }

    function handleProgressResponse(data) {
        if (!data) return;
        if (data.profile && typeof data.profile.exp === 'number') {
            window.__REALM_LAST_PROFILE = data.profile;
        }
        if (Array.isArray(data.newly_completed_missions)) {
            data.newly_completed_missions.forEach(function (mission) {
                showGameToast(
                    'MISSION COMPLETE',
                    mission.title,
                    '+' + mission.reward_exp + ' EXP · ₮ ' + mission.reward_credits
                );
            });
        }
        if (Array.isArray(data.newly_unlocked_achievements)) {
            data.newly_unlocked_achievements.forEach(function (ach) {
                var tag = ach.icon_key ? '[' + ach.icon_key + '] ' : '';
                showGameToast(
                    'ACHIEVEMENT UNLOCKED',
                    tag + ach.name,
                    ach.description || ''
                );
            });
        }
    }

    function reportProgressEvent(event, payload) {
        var url = bootstrap.urls && bootstrap.urls.progressEvent;
        if (!url) return;
        payload = payload || {};
        fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCsrfToken(),
            },
            body: JSON.stringify({ event: event, payload: payload }),
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

    function renderTrainingPanel() {
        if (!training.enabled) return;
        var step =
            training.currentStep < training.steps.length ? training.steps[training.currentStep] : null;
        var title = document.getElementById('training-step-title');
        var desc = document.getElementById('training-step-desc');
        var fill = document.getElementById('training-progress-fill');
        var list = document.getElementById('training-step-list');
        if (title) {
            title.textContent = training.completed || !step ? '训练完成' : step.title;
        }
        if (desc) {
            desc.textContent =
                training.completed || !step
                    ? '你的 Desktop Realm 控制协议已完成校准。可通过终端查看任务与成就。'
                    : step.desc;
        }
        if (fill) {
            var pct = training.steps.length
                ? Math.min(100, (100 * training.currentStep) / training.steps.length)
                : 0;
            fill.style.width = pct + '%';
        }
        if (list) {
            list.innerHTML = training.steps
                .map(function (s, index) {
                    var cls = 'training-step-item';
                    if (index < training.currentStep) cls += ' done';
                    if (index === training.currentStep && !training.completed) cls += ' active';
                    var prefix =
                        index < training.currentStep ? '✓' : index === training.currentStep && !training.completed ? '▸' : '·';
                    return '<div class="' + cls + '">' + prefix + ' ' + s.title + '</div>';
                })
                .join('');
        }
    }

    function finishTraining() {
        if (training.completed) return;
        training.completed = true;
        renderTrainingPanel();
        showGameToast(
            'TRAINING COMPLETE',
            '接入训练完成',
            '你已经掌握 Desktop Realm 的基础控制协议。'
        );
        reportProgressEvent('training_completed', { mode: 'desktop' });
    }

    function completeTrainingStep(key) {
        if (!training.enabled || training.completed) return;
        if (training.currentStep >= training.steps.length) return;
        var step = training.steps[training.currentStep];
        if (!step || step.key !== key) return;

        var doneTitle = step.title;
        var doneDesc = step.desc;
        training.currentStep += 1;
        renderTrainingPanel();

        if (training.currentStep >= training.steps.length) {
            finishTraining();
            return;
        }
        showGameToast('TRAINING STEP COMPLETE', doneTitle, doneDesc);
    }

    function createTrainingTarget() {
        var geo = new THREE.TorusGeometry(0.8, 0.035, 12, 80);
        var mat = new THREE.MeshBasicMaterial({
            color: 0x00ff88,
            transparent: true,
            opacity: 0.85,
        });
        var ring = new THREE.Mesh(geo, mat);
        ring.rotation.x = Math.PI / 2;
        ring.position.set(0, 0.05, 2.5);
        scene.add(ring);
        training.targetRing = ring;
    }

    function createTrainingNode() {
        var geo = new THREE.IcosahedronGeometry(0.35, 1);
        var mat = new THREE.MeshStandardMaterial({
            color: 0xbc00ff,
            emissive: 0xbc00ff,
            emissiveIntensity: 1.2,
            roughness: 0.3,
            metalness: 0.4,
        });
        var node = new THREE.Mesh(geo, mat);
        node.position.set(1.8, 1.2, 1.2);
        scene.add(node);
        training.node = node;
    }

    function handleInteract() {
        if (!training.enabled || training.completed) return;
        var currentStep = training.steps[training.currentStep];
        if (!currentStep || currentStep.key !== 'interact') return;
        if (!training.node) return;
        var nx = training.node.position.x;
        var nz = training.node.position.z;
        var dx = camera.position.x - nx;
        var dz = camera.position.z - nz;
        var dist = Math.sqrt(dx * dx + dz * dz);
        if (dist < 2.2) {
            completeTrainingStep('interact');
        } else {
            showGameToast('NODE OUT OF RANGE', '距离过远', '靠近紫色训练节点后按 E。');
        }
    }

    function updateTraining(delta) {
        if (!training.enabled || training.completed) return;
        if (training.targetRing) {
            training.targetRing.rotation.z += delta * 1.5;
        }
        if (training.node) {
            training.node.rotation.y += delta * 1.4;
            training.node.rotation.x += delta * 0.6;
        }
        var currentStep = training.steps[training.currentStep];
        if (currentStep && currentStep.key === 'move' && training.targetRing) {
            var rx = training.targetRing.position.x;
            var rz = training.targetRing.position.z;
            var ddx = camera.position.x - rx;
            var ddz = camera.position.z - rz;
            var dist = Math.sqrt(ddx * ddx + ddz * ddz);
            if (dist < 1.2) {
                completeTrainingStep('move');
            }
        }
    }

    function initTrainingMode() {
        training.enabled = true;
        training.currentStep = 0;
        training.completed = false;
        createTrainingTarget();
        createTrainingNode();
        renderTrainingPanel();
        showGameToast(
            'TRAINING GROUND',
            '接入训练已启动',
            '完成 6 个步骤以校准你的 Desktop Realm 控制协议。'
        );
    }

    function showToast(text) {
        var toast = document.getElementById('realm-toast');
        if (!toast) {
            toast = document.createElement('div');
            toast.id = 'realm-toast';
            toast.style.cssText =
                'position:fixed;left:50%;top:88px;transform:translateX(-50%);padding:12px 26px;border:1px solid #00f2ff;background:rgba(0,20,30,0.92);color:#00f2ff;font-family:Syncopate,sans-serif;font-size:11px;letter-spacing:2px;z-index:10050;pointer-events:none;transition:opacity .35s;';
            document.body.appendChild(toast);
        }
        toast.textContent = text;
        toast.style.opacity = '1';
        clearTimeout(showToast._t);
        showToast._t = setTimeout(function () {
            toast.style.opacity = '0';
        }, 2000);
    }

    function saveRealm() {
        var url = bootstrap.urls && bootstrap.urls.saveRealm;
        if (!url) {
            showToast('SAVE: NO API URL');
            return;
        }
        var roomTpl = state.room_template;
        if (runtimeMode === 'training' && persistentRoomTemplateForSave) {
            roomTpl = persistentRoomTemplateForSave;
        }
        var payload = {
            room_template: roomTpl,
            material_pack: state.material_pack,
            lighting: state.lighting,
            decorations: state.decorations,
        };
        fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCsrfToken(),
            },
            body: JSON.stringify(payload),
        })
            .then(function (r) {
                return r.json();
            })
            .then(function (data) {
                if (data.ok) {
                    showToast('PRIVATE REALM SAVED');
                    if (data.realm) {
                        realm = data.realm;
                        if (runtimeMode === 'training' && persistentRoomTemplateForSave) {
                            state.room_template = 'basic_room';
                        } else {
                            var nextTpl = normalizeRoomTemplate(realm.room_template);
                            if (nextTpl !== normalizeRoomTemplate(state.room_template)) {
                                state.room_template = nextTpl;
                                createRoom();
                            } else {
                                state.room_template = nextTpl;
                            }
                        }
                        state.material_pack = realm.material_pack;
                        state.lighting = realm.lighting;
                        state.decorations = realm.decorations || [];
                        applyLighting();
                        applyMaterialPack(state.material_pack, { silent: true });
                        refreshScanGhosts();
                        updateScanHud();
                    }
                    window.dispatchEvent(
                        new CustomEvent('realm-action', { detail: { action: 'realm-saved' } })
                    );
                    reportProgressEvent('realm_saved', {
                        room_template: roomTpl,
                        material_pack: state.material_pack,
                        lighting: state.lighting,
                    });
                    if (runtimeMode === 'training') {
                        completeTrainingStep('save');
                    }
                } else {
                    showToast('SAVE FAILED: ' + (data.error || 'UNKNOWN'));
                }
            })
            .catch(function () {
                showToast('SAVE FAILED: NETWORK');
            });
    }

    function updateMovement(delta) {
        if (!isDesktopControl()) return;
        if (!renderer) return;

        var speed = keys.ShiftLeft ? 6.5 : 3.2;
        var dir = { x: 0, z: 0 };
        if (keys.KeyW || keys.ArrowUp) dir.z -= 1;
        if (keys.KeyS || keys.ArrowDown) dir.z += 1;
        if (keys.KeyA || keys.ArrowLeft) dir.x -= 1;
        if (keys.KeyD || keys.ArrowRight) dir.x += 1;
        if (dir.x === 0 && dir.z === 0) return;

        if (!reported.moved) {
            reported.moved = true;
            reportProgressEvent('desktop_moved', { mode: 'desktop' });
        }

        var len = Math.sqrt(dir.x * dir.x + dir.z * dir.z);
        dir.x /= len;
        dir.z /= len;

        var forward = new THREE.Vector3();
        camera.getWorldDirection(forward);
        forward.y = 0;
        forward.normalize();

        var right = new THREE.Vector3();
        right.crossVectors(forward, camera.up).normalize();

        camera.position.addScaledVector(forward, -dir.z * speed * delta);
        camera.position.addScaledVector(right, dir.x * speed * delta);

        camera.position.x = Math.max(cameraMoveBounds.xMin, Math.min(cameraMoveBounds.xMax, camera.position.x));
        camera.position.z = Math.max(cameraMoveBounds.zMin, Math.min(cameraMoveBounds.zMax, camera.position.z));
        applyPlayerEyeHeight(camera);
    }

    function updateDebugHud(now) {
        var el = document.getElementById('realm-debug-hud');
        if (el) {
            el.classList.add('visible');
            el.setAttribute('aria-hidden', 'false');
            var b = cameraMoveBounds;
            var n = roomGroup ? roomGroup.children.length : 0;
            el.textContent =
                'X: ' +
                camera.position.x.toFixed(2) +
                ' / Z: ' +
                camera.position.z.toFixed(2) +
                '\nROOM: ' +
                state.room_template +
                '\nACTIVE: ' +
                activeRoomTemplateKey +
                '\nBOUNDS: x[' +
                b.xMin +
                ',' +
                b.xMax +
                '] z[' +
                b.zMin +
                ',' +
                b.zMax +
                ']' +
                '\nMESH: roomGroup.children=' +
                n;
        }
        if (typeof console !== 'undefined' && console.debug && now - realmDebugLastLog > 1500) {
            realmDebugLastLog = now;
            console.debug('[REALM] roomGroup.children:', roomGroup ? roomGroup.children.length : 0);
        }
    }

    var lastT = performance.now();

    function animate() {
        requestAnimationFrame(animate);
        var now = performance.now();
        var delta = Math.min((now - lastT) / 1000, 0.06);
        lastT = now;
        updateMovement(delta);
        updateTraining(delta);
        if (realmDebugEnabled) {
            updateDebugHud(now);
        }
        if (
            window.RealmStereo &&
            typeof window.RealmStereo.cameraPoseIsFinite === 'function' &&
            !window.RealmStereo.cameraPoseIsFinite(camera)
        ) {
            setDefaultCameraForTemplate(activeRoomTemplateKey);
            camera.rotation.set(0, 0, 0, 'YXZ');
            camera.updateMatrixWorld(true);
        }

        var didStereo = false;
        if (window.RealmStereo && typeof window.RealmStereo.renderStereo === 'function') {
            try {
                didStereo = window.RealmStereo.renderStereo(renderer, scene, camera);
            } catch (err) {
                console.error('[REALM] stereo render failed, fallback to mono', err);
                didStereo = false;
            }
        }
        if (!didStereo) {
            renderer.autoClear = true;
            renderer.setScissorTest(false);
            var buf =
                window.RealmStereo && typeof window.RealmStereo.getBufferSize === 'function'
                    ? window.RealmStereo.getBufferSize(renderer)
                    : {
                          w: renderer.domElement ? renderer.domElement.width : 0,
                          h: renderer.domElement ? renderer.domElement.height : 0,
                      };
            if (buf.w < 8 || buf.h < 8) {
                onResize();
                buf =
                    window.RealmStereo && typeof window.RealmStereo.getBufferSize === 'function'
                        ? window.RealmStereo.getBufferSize(renderer)
                        : {
                              w: renderer.domElement ? renderer.domElement.width : 0,
                              h: renderer.domElement ? renderer.domElement.height : 0,
                          };
            }
            if (buf.w > 0 && buf.h > 0) {
                renderer.setViewport(0, 0, buf.w, buf.h);
                renderer.render(scene, camera);
            }
        }
        if (window.RealmStereo && typeof window.RealmStereo.resetRendererFrame === 'function') {
            window.RealmStereo.resetRendererFrame(renderer);
        }
    }

    function bindKeys() {
        console.warn('[REALM MAIN] bindKeys called');
        window.addEventListener('realm-action', function (e) {
            var d = e.detail;
            if (!d || d.action !== 'control-mode-changed') return;
            if (d.mode !== 'desktop') {
                releaseDesktopPointer();
            }
            if (camera) {
                if (d.mode === 'desktop') {
                    setDefaultCameraForTemplate(activeRoomTemplateKey);
                } else {
                    applyPlayerEyeHeight(camera);
                }
            }
            if (window.RealmStereo && typeof window.RealmStereo.applyStereoLayout === 'function') {
                window.RealmStereo.applyStereoLayout(d.mode);
            }
            onResize();
        });

        window.addEventListener(
            'keydown',
            function (e) {
                if (e.repeat) return;
                if (shouldIgnoreKeyTarget(e.target)) return;

                if (isMovementKey(e.code)) {
                    if (isDesktopControl()) {
                        keys[e.code] = true;
                        e.preventDefault();
                    } else {
                        keys[e.code] = false;
                        e.preventDefault();
                        e.stopPropagation();
                    }
                    return;
                }

                if (!isDesktopControl()) return;

                keys[e.code] = true;

                if (e.code === 'Escape' && document.pointerLockElement) {
                    document.exitPointerLock();
                }
                if (e.code === 'KeyE') {
                    handleInteract();
                }
                var num = {
                    Digit1: 'cyber_neon',
                    Digit2: 'wasteland_rust',
                    Digit3: 'starship_alloy',
                    Digit4: 'magic_stone',
                    Digit5: 'forest_temple',
                    Digit6: 'deep_sea',
                    Digit7: 'pixel_retro',
                    Digit8: 'hacker_matrix',
                };
                if (num[e.code]) {
                    applyMaterialPack(num[e.code]);
                }
            },
            true
        );
        window.addEventListener(
            'keyup',
            function (e) {
                if (isWasdKey(e.code)) {
                    keys[e.code] = false;
                    return;
                }
                if (shouldIgnoreKeyTarget(e.target)) return;
                keys[e.code] = false;
            },
            true
        );

        window.addEventListener('blur', clearMovementKeys);

        window.__realmMoveDebug = {
            keys: keys,
            getControlMode: getControlMode,
            isDesktopControl: isDesktopControl,
            clearMovementKeys: clearMovementKeys,
        };
        console.warn('[REALM MAIN] __realmMoveDebug attached', window.__realmMoveDebug);
    }

    function bindMaterialDock() {
        document.querySelectorAll('[data-material]').forEach(function (btn) {
            btn.addEventListener('click', function (e) {
                e.preventDefault();
                e.stopPropagation();
                var code = btn.getAttribute('data-material');
                if (code) applyMaterialPack(code);
            });
        });
        var toggle = document.getElementById('toggle-more-protocols');
        var moreRow = document.getElementById('more-protocols-row');
        if (toggle && moreRow) {
            toggle.addEventListener('click', function (e) {
                e.preventDefault();
                e.stopPropagation();
                var collapsed = moreRow.classList.toggle('collapsed');
                toggle.textContent = collapsed ? 'MORE PROTOCOLS ▾' : 'LESS PROTOCOLS ▴';
            });
        }
        var saveBtn = document.getElementById('save-realm-btn');
        if (saveBtn) {
            saveBtn.addEventListener('click', function (e) {
                e.preventDefault();
                e.stopPropagation();
                saveRealm();
            });
        }
    }

    function bindPointer() {
        var canvas = renderer.domElement;
        canvas.setAttribute('tabindex', '0');
        canvas.style.outline = 'none';
        canvas.addEventListener('click', function () {
            if (!isDesktopControl()) return;
            canvas.focus({ preventScroll: true });
            if (document.pointerLockElement !== canvas) {
                canvas.requestPointerLock();
            }
        });
        document.addEventListener('pointerlockchange', function () {
            if (!isDesktopControl()) return;
            if (document.pointerLockElement === canvas && !reported.pointerLocked) {
                reported.pointerLocked = true;
                reportProgressEvent('pointer_locked', { mode: 'desktop' });
            }
            if (document.pointerLockElement === canvas) {
                completeTrainingStep('pointer');
            }
        });
        document.addEventListener('mousemove', function (e) {
            if (!isDesktopControl()) return;
            if (document.pointerLockElement !== canvas) return;
            if (e.movementX || e.movementY) {
                if (!reported.lookMoved) {
                    reported.lookMoved = true;
                    reportProgressEvent('look_moved', { mode: 'desktop' });
                }
                completeTrainingStep('look');
            }
            yaw -= e.movementX * 0.0022;
            pitch -= e.movementY * 0.0022;
            pitch = Math.max(-1.25, Math.min(1.25, pitch));
            camera.rotation.order = 'YXZ';
            camera.rotation.y = yaw;
            camera.rotation.x = pitch;
        });
    }

    function init() {
        if (typeof THREE === 'undefined') {
            console.error('[REALM] THREE.js not loaded');
            return;
        }
        var root = document.getElementById('realm-canvas-root');
        if (!root) return;

        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x050608);

        camera = new THREE.PerspectiveCamera(72, window.innerWidth / window.innerHeight, 0.08, 120);

        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        root.appendChild(renderer.domElement);
        renderer.domElement.setAttribute('tabindex', '0');

        window.scene = scene;
        window.camera = camera;
        window.renderer = renderer;

        roomGroup = new THREE.Group();
        scene.add(roomGroup);

        createLights();
        createRoom();
        setDefaultCameraForTemplate(activeRoomTemplateKey);
        applyMaterialPack(state.material_pack, { silent: true });
        applyLighting();
        bindKeys();
        bindPointer();
        bindMaterialDock();
        updateHud();

        if (runtimeMode === 'training') {
            initTrainingMode();
        }

        window.addEventListener('resize', onResize);
        onResize();
        animate();
        if (runtimeMode === 'realm') {
            reportProgressEvent('realm_entered', { mode: 'realm' });
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
