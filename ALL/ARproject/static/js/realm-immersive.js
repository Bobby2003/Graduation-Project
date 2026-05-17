/**
 * AR / VR 沉浸模式：隐藏 HUD、手掌菜单、扫描与材质快捷操作
 */
(function () {
    'use strict';

    var menuOpen = false;
    var materialGridVisible = false;

    var MATERIALS = [
        { code: 'cyber_neon', label: '赛博霓虹' },
        { code: 'wasteland_rust', label: '废土铁锈' },
        { code: 'starship_alloy', label: '星舰合金' },
        { code: 'magic_stone', label: '魔法石墙' },
        { code: 'forest_temple', label: '森林神殿' },
        { code: 'deep_sea', label: '深海基地' },
        { code: 'pixel_retro', label: '像素复古' },
        { code: 'hacker_matrix', label: '黑客矩阵' },
    ];

    function getMode() {
        if (window.ARRealmControls && window.ARRealmControls.getMode) {
            return window.ARRealmControls.getMode();
        }
        try {
            return localStorage.getItem('ar_realm_control_mode') || 'desktop';
        } catch (e) {
            return 'desktop';
        }
    }

    function isImmersive() {
        var m = getMode();
        return m === 'gesture' || m === 'vr';
    }

    function el(id) {
        return document.getElementById(id);
    }

    function updateHint() {
        var hint = el('realm-immersive-hint');
        if (!hint) return;
        var m = getMode();
        if (m === 'gesture') {
            hint.textContent =
                '立体分屏 · 手机横屏 · 进入即 IMU 驱动画面 · 「开始扫描」才启深度相机建模';
        } else if (m === 'vr') {
            hint.textContent =
                '立体分屏 · 横屏放入 VR 眼镜 · 进入即 IMU 漫游 · 「开始扫描」启深度相机 · 点「菜单」或切回桌面';
        }
    }

    function setMenuOpen(open) {
        menuOpen = !!open;

        if (window.__arGesture) {
            window.__arGesture._palmMenuOpen = menuOpen;
        }

        var menu = el('realm-ar-menu');
        if (!menu) {
            console.warn('[REALM MENU] #realm-ar-menu not found');
            return;
        }

        menu.classList.toggle('is-open', menuOpen);
        menu.setAttribute('aria-hidden', menuOpen ? 'false' : 'true');

        if (document.body) {
            document.body.classList.toggle('ar-palm-nav-stereo', menuOpen && (getMode() === 'gesture' || getMode() === 'vr'));
        }

        if (menuOpen && window.ARStereoOverlaySync) {
            window.ARStereoOverlaySync.syncById('realm-ar-menu');
        }

        if (!menuOpen) {
            hideMaterialGrid();
        }
    }

    window.setRealmARMenuOpen = setMenuOpen;
    window.toggleRealmARMenu = function () {
        setMenuOpen(!menuOpen);
    };

    function hideMaterialGrid() {
        materialGridVisible = false;
        var grid = el('realm-ar-material-grid');
        if (grid) grid.classList.remove('is-visible');
    }

    function showMaterialGrid() {
        materialGridVisible = true;
        var grid = el('realm-ar-material-grid');
        if (!grid) return;
        if (!grid.childElementCount) {
            MATERIALS.forEach(function (m) {
                var btn = document.createElement('button');
                btn.type = 'button';
                btn.textContent = m.label;
                btn.setAttribute('data-material', m.code);
                btn.addEventListener('click', function (e) {
                    e.preventDefault();
                    e.stopPropagation();
                    applyMaterial(m.code);
                });
                grid.appendChild(btn);
            });
        }
        grid.classList.add('is-visible');
    }

    function applyMaterial(code) {
        document.querySelectorAll('[data-material="' + code + '"]').forEach(function (btn) {
            btn.click();
        });
        window.dispatchEvent(
            new CustomEvent('realm-action', {
                detail: { action: 'material-protocol-changed', material: code, source: 'ar-menu' },
            })
        );
        showToast('已切换材质：' + code);
    }

    function showToast(msg) {
        var t = el('realm-immersive-toast');
        if (!t) {
            t = document.createElement('div');
            t.id = 'realm-immersive-toast';
            t.style.cssText =
                'position:fixed;top:72px;left:50%;transform:translateX(-50%);z-index:220;padding:10px 16px;' +
                'background:rgba(0,40,50,0.9);border:1px solid #00f2ff;color:#bff;font-family:Rajdhani,sans-serif;' +
                'font-size:13px;pointer-events:none;opacity:0;transition:opacity 0.25s;border-radius:6px;';
            document.body.appendChild(t);
        }
        t.textContent = msg;
        t.style.opacity = '1';
        clearTimeout(showToast._timer);
        showToast._timer = setTimeout(function () {
            t.style.opacity = '0';
        }, 2200);
    }

    function clickCenterButton(id) {
        var btn = document.getElementById(id);
        if (btn) btn.click();
    }

    function onMenuAction(action) {
        switch (action) {
            case 'materials':
                if (materialGridVisible) hideMaterialGrid();
                else showMaterialGrid();
                break;
            case 'scan-start':
                clickCenterButton('center-start-btn');
                window.dispatchEvent(
                    new CustomEvent('realm-action', { detail: { action: 'scan-start' } })
                );
                showToast('已开始扫描（Center 管线）');
                setMenuOpen(false);
                break;
            case 'scan-stop':
                clickCenterButton('center-stop-btn');
                window.dispatchEvent(
                    new CustomEvent('realm-action', { detail: { action: 'scan-stop' } })
                );
                showToast('已停止扫描');
                setMenuOpen(false);
                break;
            case 'desktop':
                if (window.ARRealmControls) window.ARRealmControls.setMode('desktop');
                setMenuOpen(false);
                showToast('已切换桌面模式 · WASD 可移动');
                break;
            case 'dashboard': {
                var menuEl = el('realm-ar-menu');
                var href =
                    (menuEl && menuEl.getAttribute('data-dashboard-href')) || '/dashboard/';
                setMenuOpen(false);
                window.location.href = href;
                break;
            }
            case 'close':
                setMenuOpen(false);
                break;
            default:
                break;
        }
    }

    function bindMenu() {
        var menu = el('realm-ar-menu');
        if (!menu) return;

        menu.querySelectorAll('[data-ar-action]').forEach(function (btn) {
            btn.addEventListener('click', function (e) {
                e.preventDefault();
                e.stopPropagation();
                onMenuAction(btn.getAttribute('data-ar-action'));
            });
        });

        menu.querySelectorAll('[data-overlay-backdrop]').forEach(function (backdrop) {
            backdrop.addEventListener('click', function () {
                setMenuOpen(false);
            });
        });

        menu.addEventListener('click', function (e) {
            if (e.target === menu || e.target.classList.contains('ar-stereo-overlay-row')) {
                setMenuOpen(false);
            }
        });
    }

    function bindVrMenuButton() {
        var vrBtn = el('realm-vr-menu-btn');
        if (vrBtn) {
            vrBtn.addEventListener('click', function (e) {
                e.preventDefault();
                setMenuOpen(true);
            });
        }
    }

    function onControlModeChanged() {
        updateHint();
        if (!isImmersive()) {
            setMenuOpen(false);
        }
        if (document.pointerLockElement) {
            document.exitPointerLock();
        }
    }

    function onReady() {
        bindMenu();
        bindVrMenuButton();
        updateHint();

        window.addEventListener('realm-action', function (e) {
            var d = e.detail;
            if (!d) return;
            if (d.action === 'control-mode-changed') onControlModeChanged();
        });

        window.addEventListener('ar-gesture', function (e) {
            var d = e.detail;
            var m = getMode();
            if (!d || (m !== 'gesture' && m !== 'vr')) return;
            if (d.action === 'open-palm-menu') setMenuOpen(true);
            if (d.action === 'close-palm-menu') setMenuOpen(false);
        });

        window.addEventListener(
            'keydown',
            function (e) {
                if (e.code === 'Escape' && menuOpen) {
                    setMenuOpen(false);
                    return;
                }
                if (e.code !== 'KeyM') return;
                var m = getMode();
                console.log('[REALM MENU] KeyM pressed', {
                    mode: m,
                    bodyMode: document.body.dataset.controlMode,
                    bodyClass: document.body.className,
                });
                if (m !== 'gesture' && m !== 'vr') return;
                e.preventDefault();
                e.stopPropagation();
                setMenuOpen(!menuOpen);
                showToast(menuOpen ? '管理菜单已打开 (M)' : '管理菜单已关闭');
            },
            true
        );

        onControlModeChanged();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', onReady);
    } else {
        onReady();
    }
})();
