/**
 * AR REALM — 背景音乐（进页自动播放、跨页续播）+ UI 音效
 */
(function (global) {
    'use strict';

    var PREFS_KEY = 'ar_realm_audio_prefs';
    var BGM_TIME_KEY = 'ar_bgm_time';
    var BGM_SAVED_AT_KEY = 'ar_bgm_saved_at';
    var BGM_PLAYING_KEY = 'ar_bgm_playing';
    var BGM_UNLOCKED_KEY = 'ar_bgm_unlocked';

    var defaults = { master: 0.6, bgmEnabled: true, sfxEnabled: true };

    var prefs = loadPrefs();
    var bgmEl = null;
    var hoverLast = 0;
    var bgmBootstrapped = false;

    function urls() {
        return global.AR_AUDIO_URLS || {};
    }

    function loadPrefs() {
        try {
            var raw = localStorage.getItem(PREFS_KEY);
            if (!raw) return copy(defaults);
            var parsed = JSON.parse(raw);
            return {
                master: clamp01(parsed.master != null ? parsed.master : defaults.master),
                bgmEnabled: parsed.bgmEnabled !== false,
                sfxEnabled: parsed.sfxEnabled !== false,
            };
        } catch (e) {
            return copy(defaults);
        }
    }

    function savePrefs() {
        try {
            localStorage.setItem(PREFS_KEY, JSON.stringify(prefs));
        } catch (e) { /* ignore */ }
    }

    function copy(o) {
        return { master: o.master, bgmEnabled: o.bgmEnabled, sfxEnabled: o.sfxEnabled };
    }

    function clamp01(n) {
        n = Number(n);
        if (isNaN(n)) return defaults.master;
        return Math.max(0, Math.min(1, n));
    }

    function sessionGet(key) {
        try {
            return sessionStorage.getItem(key);
        } catch (e) {
            return null;
        }
    }

    function sessionSet(key, val) {
        try {
            sessionStorage.setItem(key, val);
        } catch (e) { /* ignore */ }
    }

    function bgmGainMultiplier() {
        var body = document.body;
        if (!body) return 0.42;
        var attr = body.getAttribute('data-ar-audio-bgm-gain');
        if (attr != null && attr !== '') {
            var g = parseFloat(attr);
            if (!isNaN(g)) return clamp01(g);
        }
        return 0.42;
    }

    function isBgmAllowed() {
        if (document.body && document.body.getAttribute('data-ar-audio') === 'off') return false;
        if (document.body && document.body.getAttribute('data-ar-audio-bgm') === 'off') return false;
        return prefs.bgmEnabled;
    }

    function getBgmElement() {
        if (bgmEl) return bgmEl;
        bgmEl = document.getElementById('ar-realm-bgm');
        if (!bgmEl) {
            var src = urls().bgm;
            if (!src) return null;
            bgmEl = document.createElement('audio');
            bgmEl.id = 'ar-realm-bgm';
            bgmEl.loop = true;
            bgmEl.preload = 'auto';
            bgmEl.autoplay = true;
            bgmEl.muted = true;
            bgmEl.setAttribute('playsinline', '');
            bgmEl.src = src;
            bgmEl.style.cssText = 'position:fixed;width:0;height:0;opacity:0;pointer-events:none';
            (document.body || document.documentElement).appendChild(bgmEl);
        }
        return bgmEl;
    }

    function applyVolumes() {
        if (!bgmEl) return;
        bgmEl.volume = prefs.master * bgmGainMultiplier();
    }

    function computeResumeTime() {
        var t = parseFloat(sessionGet(BGM_TIME_KEY));
        var at = parseFloat(sessionGet(BGM_SAVED_AT_KEY));
        if (isNaN(t) || isNaN(at)) return 0;
        return Math.max(0, t + (Date.now() - at) / 1000);
    }

    function restoreBgmPosition() {
        if (!bgmEl) return;
        var resume = computeResumeTime();
        if (resume > 0) {
            var apply = function () {
                try {
                    if (bgmEl.duration && isFinite(bgmEl.duration) && bgmEl.duration > 0) {
                        bgmEl.currentTime = resume % bgmEl.duration;
                    } else {
                        bgmEl.currentTime = resume;
                    }
                } catch (e) { /* ignore */ }
            };
            if (bgmEl.readyState >= 1) apply();
            else bgmEl.addEventListener('loadedmetadata', apply, { once: true });
        }
    }

    function saveBgmState() {
        if (!bgmEl) return;
        try {
            sessionSet(BGM_TIME_KEY, String(bgmEl.currentTime || 0));
            sessionSet(BGM_SAVED_AT_KEY, String(Date.now()));
            sessionSet(BGM_PLAYING_KEY, bgmEl.paused ? '0' : '1');
        } catch (e) { /* ignore */ }
    }

    function markBgmUnlocked() {
        sessionSet(BGM_UNLOCKED_KEY, '1');
    }

    function tryPlayBgm() {
        if (!isBgmAllowed()) {
            if (bgmEl) bgmEl.pause();
            return Promise.resolve(false);
        }

        var el = getBgmElement();
        if (!el) return Promise.resolve(false);

        restoreBgmPosition();

        function markPlaying() {
            markBgmUnlocked();
            sessionSet(BGM_PLAYING_KEY, '1');
        }

        /* 先保证 muted 下能播（多数浏览器允许），再取消静音 */
        el.muted = true;
        return el.play().then(function () {
            el.muted = false;
            applyVolumes();
            return el.play();
        }).then(function () {
            markPlaying();
            return true;
        }).catch(function () {
            el.muted = true;
            return el.play().then(function () {
                markPlaying();
                window.setTimeout(function () {
                    if (!el || !isBgmAllowed()) return;
                    el.muted = false;
                    applyVolumes();
                    el.play().catch(function () {});
                }, 400);
                return true;
            }).catch(function () {
                return false;
            });
        });
    }

    function pauseBgm() {
        saveBgmState();
        if (bgmEl) {
            bgmEl.pause();
            sessionSet(BGM_PLAYING_KEY, '0');
        }
    }

    function bootstrapBgm() {
        if (bgmBootstrapped) return;
        bgmBootstrapped = true;

        if (!isBgmAllowed()) return;

        getBgmElement();
        applyVolumes();
        restoreBgmPosition();

        tryPlayBgm();

        if (bgmEl) {
            bgmEl.addEventListener('timeupdate', function () {
                if (bgmEl.paused) return;
                if (Math.floor(bgmEl.currentTime) % 4 === 0) saveBgmState();
            });
        }
    }

    function playSfx(kind) {
        if (!prefs.sfxEnabled) return;
        var key = kind === 'hover' ? 'hover' : 'click';
        var src = urls()[key];
        if (!src) return;
        try {
            var a = new Audio(src);
            a.volume = prefs.master * (key === 'hover' ? 0.32 : 0.52);
            var p = a.play();
            if (p && typeof p.catch === 'function') p.catch(function () {});
        } catch (e) { /* ignore */ }
    }

    function isInteractive(el) {
        if (!el || el.closest('[data-ar-audio-silent]')) return false;
        if (el.disabled || el.getAttribute('aria-disabled') === 'true') return false;
        return !!el.closest(
            'a[href], button, [role="button"], input[type="submit"], input[type="button"], ' +
            '.btn-submit, .btn-action, .btn-return, .btn-save, .btn-link, .mode-btn, ' +
            '.realm-card, .loot-item button, .material-btn, .hub-tab, label.toggle-switch'
        );
    }

    function bindUi() {
        document.addEventListener(
            'click',
            function (e) {
                if (!isInteractive(e.target)) return;
                playSfx('click');
                if (bgmEl && bgmEl.paused && isBgmAllowed()) tryPlayBgm();
            },
            true
        );

        document.addEventListener(
            'pointerenter',
            function (e) {
                if (e.pointerType === 'touch') return;
                if (!isInteractive(e.target)) return;
                var now = Date.now();
                if (now - hoverLast < 100) return;
                hoverLast = now;
                playSfx('hover');
            },
            true
        );
    }

    function bindLifecycle() {
        window.addEventListener('pagehide', saveBgmState);
        window.addEventListener('beforeunload', saveBgmState);

        document.addEventListener('visibilitychange', function () {
            if (document.visibilityState === 'hidden') {
                saveBgmState();
            } else if (document.visibilityState === 'visible' && isBgmAllowed()) {
                tryPlayBgm();
            }
        });

        window.addEventListener('pageshow', function () {
            if (isBgmAllowed()) {
                restoreBgmPosition();
                tryPlayBgm();
            }
        });
    }

    function syncSettingsUi() {
        var masterInput = document.querySelector('[data-ar-audio-master]');
        if (masterInput) {
            masterInput.value = Math.round(prefs.master * 100);
            var row = masterInput.closest('.slider-control');
            if (row) {
                var fill = row.querySelector('.slider-fill');
                var label = row.querySelector('.slider-value');
                if (fill) fill.style.width = masterInput.value + '%';
                if (label) label.textContent = masterInput.value + '%';
            }
        }
        var bgmToggle = document.querySelector('[data-ar-audio-bgm]');
        if (bgmToggle) bgmToggle.checked = prefs.bgmEnabled;
        var sfxToggle = document.querySelector('[data-ar-audio-sfx]');
        if (sfxToggle) sfxToggle.checked = prefs.sfxEnabled;
    }

    function bindSettings() {
        var masterInput = document.querySelector('[data-ar-audio-master]');
        if (masterInput) {
            masterInput.addEventListener('input', function () {
                prefs.master = clamp01(Number(this.value) / 100);
                savePrefs();
                applyVolumes();
                var row = this.closest('.slider-control');
                if (row) {
                    var fill = row.querySelector('.slider-fill');
                    var label = row.querySelector('.slider-value');
                    if (fill) fill.style.width = this.value + '%';
                    if (label) label.textContent = this.value + '%';
                }
            });
        }
        var bgmToggle = document.querySelector('[data-ar-audio-bgm]');
        if (bgmToggle) {
            bgmToggle.addEventListener('change', function () {
                prefs.bgmEnabled = this.checked;
                savePrefs();
                if (prefs.bgmEnabled) tryPlayBgm();
                else pauseBgm();
            });
        }
        var sfxToggle = document.querySelector('[data-ar-audio-sfx]');
        if (sfxToggle) {
            sfxToggle.addEventListener('change', function () {
                prefs.sfxEnabled = this.checked;
                savePrefs();
            });
        }
    }

    function init() {
        if (document.body && document.body.getAttribute('data-ar-audio') === 'off') return;
        syncSettingsUi();
        bindUi();
        bindSettings();
        bindLifecycle();
        bootstrapBgm();
    }

    global.ARAudio = {
        getPrefs: function () { return copy(prefs); },
        playSfx: playSfx,
        tryPlayBgm: tryPlayBgm,
        pauseBgm: pauseBgm,
        saveBgmState: saveBgmState,
        setMaster: function (v) {
            prefs.master = clamp01(v);
            savePrefs();
            applyVolumes();
        },
    };

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})(window);
