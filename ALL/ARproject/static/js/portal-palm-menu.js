/**
 * 门户 HUD 页：AR/VR 双屏掌菜导航（无滚动 · 大按钮）
 */
(function () {
    'use strict';

    var menuOpen = false;

    function isImmersive3D() {
        return typeof window.ARRealmIsImmersive3DPage === 'function' && window.ARRealmIsImmersive3DPage();
    }

    function getMode() {
        if (window.ARRealmControls && typeof window.ARRealmControls.getMode === 'function') {
            return window.ARRealmControls.getMode();
        }
        try {
            return localStorage.getItem('ar_realm_control_mode') || 'desktop';
        } catch (e) {
            return 'desktop';
        }
    }

    function isStereoMode() {
        var m = getMode();
        return m === 'gesture' || m === 'vr';
    }

    function isStereoUi() {
        if (window.ARStereoOverlaySync && window.ARStereoOverlaySync.isStereoUiMode) {
            return window.ARStereoOverlaySync.isStereoUiMode() &&
                (window.ARStereoOverlaySync.isPortalStereoActive
                    ? window.ARStereoOverlaySync.isPortalStereoActive()
                    : true);
        }
        return isStereoMode();
    }

    function el(id) {
        return document.getElementById(id);
    }

    function syncMenuMirror() {
        if (window.ARStereoOverlaySync) {
            window.ARStereoOverlaySync.syncById('ar-palm-nav-menu');
        }
    }

    function syncChromeMirror() {
        var chrome = el('ar-portal-stereo-chrome');
        if (!chrome) return;
        var hint = el('ar-portal-gesture-hint');
        var hintMirror = chrome.querySelector('.ar-portal-gesture-hint--mirror');
        var vrBtn = el('ar-vr-nav-btn');
        var vrMirror = chrome.querySelector('.ar-vr-nav-btn--mirror');

        if (hint && hintMirror) {
            hintMirror.textContent = hint.textContent;
            hintMirror.hidden = hint.hidden;
        }
        if (vrBtn && vrMirror) {
            vrMirror.hidden = vrBtn.hidden;
        }
    }

    function setMenuOpen(open) {
        if (isImmersive3D()) return;

        var menu = el('ar-palm-nav-menu');
        if (!menu) return;

        menuOpen = !!open;
        menu.classList.toggle('is-open', menuOpen);
        menu.setAttribute('aria-hidden', menuOpen ? 'false' : 'true');

        if (document.body) {
            document.body.classList.toggle('ar-palm-nav-open', menuOpen);
            document.body.classList.toggle('ar-palm-nav-stereo', menuOpen && isStereoUi());
        }

        if (menuOpen) {
            syncMenuMirror();
        }

        var vrBtn = el('ar-vr-nav-btn');
        if (vrBtn) {
            vrBtn.setAttribute('aria-expanded', menuOpen ? 'true' : 'false');
        }

        if (window.__arGesture) {
            window.__arGesture._palmMenuOpen = menuOpen;
        }
    }

    function updateChrome() {
        var chrome = el('ar-portal-stereo-chrome');
        var hint = el('ar-portal-gesture-hint');
        var vrBtn = el('ar-vr-nav-btn');
        var mode = getMode();
        var stereo = isStereoMode();
        var portalStereo = isStereoUi();

        if (chrome) {
            chrome.hidden = !portalStereo;
        }

        if (hint) {
            if (!isImmersive3D() && mode === 'gesture') {
                hint.hidden = false;
                hint.textContent = '张掌开导航 · 握拳关';
            } else if (!isImmersive3D() && mode === 'vr') {
                hint.hidden = false;
                hint.textContent = '点 NAV 开菜单';
            } else {
                hint.hidden = true;
            }
        }

        if (vrBtn) {
            vrBtn.hidden = isImmersive3D() || mode !== 'vr';
        }

        syncChromeMirror();

        if (!stereo) {
            setMenuOpen(false);
        }
    }

    function bindMenu() {
        var menu = el('ar-palm-nav-menu');
        if (!menu) return;

        menu.querySelectorAll('[data-ar-palm-close]').forEach(function (btn) {
            btn.addEventListener('click', function (e) {
                e.preventDefault();
                setMenuOpen(false);
            });
        });

        menu.querySelectorAll('[data-overlay-backdrop]').forEach(function (backdrop) {
            backdrop.addEventListener('click', function () {
                setMenuOpen(false);
            });
        });

        menu.querySelectorAll('.ar-stereo-overlay-eye--left a[href]').forEach(function (link) {
            link.addEventListener('click', function () {
                setMenuOpen(false);
            });
        });

        var vrBtn = el('ar-vr-nav-btn');
        if (vrBtn) {
            vrBtn.addEventListener('click', function (e) {
                e.preventDefault();
                e.stopPropagation();
                setMenuOpen(!menuOpen);
            });
        }
    }

    function bindKeys() {
        window.addEventListener(
            'keydown',
            function (e) {
                if (isImmersive3D()) return;
                var m = getMode();
                if (m !== 'gesture' && m !== 'vr') return;

                if (e.code === 'Escape' && menuOpen) {
                    e.preventDefault();
                    setMenuOpen(false);
                    return;
                }

                if (e.code === 'KeyM') {
                    e.preventDefault();
                    e.stopPropagation();
                    setMenuOpen(!menuOpen);
                }
            },
            true
        );
    }

    function bindEvents() {
        window.addEventListener('ar-gesture', function (e) {
            if (isImmersive3D()) return;
            var d = e.detail;
            if (!d) return;
            var m = getMode();
            if (m !== 'gesture' && m !== 'vr') return;

            if (d.action === 'open-palm-menu') setMenuOpen(true);
            if (d.action === 'close-palm-menu') setMenuOpen(false);
        });

        window.addEventListener('realm-action', function (e) {
            var d = e.detail;
            if (d && d.action === 'control-mode-changed') {
                window.setTimeout(updateChrome, 0);
            }
        });

        window.addEventListener('resize', function () {
            if (menuOpen) syncMenuMirror();
            syncChromeMirror();
        });
    }

    function onReady() {
        if (isImmersive3D()) return;
        bindMenu();
        bindKeys();
        bindEvents();
        updateChrome();

        document.querySelectorAll('.mode-btn').forEach(function (btn) {
            btn.addEventListener('click', function () {
                window.setTimeout(updateChrome, 50);
            });
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', onReady);
    } else {
        onReady();
    }

    window.ARPortalPalmMenu = {
        setOpen: setMenuOpen,
        isOpen: function () { return menuOpen; },
        updateChrome: updateChrome,
        syncMenuMirror: syncMenuMirror,
    };
})();
