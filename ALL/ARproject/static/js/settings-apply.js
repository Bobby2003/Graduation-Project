/**
 * 将 ui_settings 应用到全站 HUD（主题 / 透明度 / 扫描线 / 粒子）
 */
(function (global) {
    'use strict';

    var STORAGE_KEY = 'ar_user_settings';
    var DEFAULTS = {
        avatar_emoji: '🎮',
        depth_driver: 'openni2',
        gesture_sensitivity: 70,
        depth_range: '100-3000',
        fps_limit: 30,
        theme: 'cyberpunk',
        hud_opacity: 85,
        scanlines: true,
        particles: false,
        two_factor_enabled: false,
    };

    function copy(obj) {
        var out = {};
        Object.keys(obj).forEach(function (k) {
            out[k] = obj[k];
        });
        return out;
    }

    function load() {
        try {
            var raw = localStorage.getItem(STORAGE_KEY);
            if (!raw) return copy(DEFAULTS);
            var parsed = JSON.parse(raw);
            var out = copy(DEFAULTS);
            Object.keys(DEFAULTS).forEach(function (k) {
                if (parsed[k] !== undefined && parsed[k] !== null) {
                    out[k] = parsed[k];
                }
            });
            return out;
        } catch (e) {
            return copy(DEFAULTS);
        }
    }

    function persist(ui) {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(ui));
        } catch (e) { /* ignore */ }
    }

    function apply(ui) {
        ui = ui || load();
        var root = document.documentElement;
        root.setAttribute('data-ui-theme', ui.theme || 'cyberpunk');
        var op = Number(ui.hud_opacity);
        if (!Number.isFinite(op)) op = DEFAULTS.hud_opacity;
        op = Math.max(30, Math.min(100, op)) / 100;
        root.style.setProperty('--hud-panel-opacity', String(op));

        if (document.body) {
            document.body.classList.toggle('ui-scanlines-on', ui.scanlines !== false);
            document.body.classList.toggle('ui-particles-on', !!ui.particles);
        }
    }

    function syncFromServer(ui) {
        var merged = copy(DEFAULTS);
        if (ui && typeof ui === 'object') {
            Object.keys(DEFAULTS).forEach(function (k) {
                if (ui[k] !== undefined && ui[k] !== null) {
                    merged[k] = ui[k];
                }
            });
        }
        persist(merged);
        apply(merged);
        return merged;
    }

    function init() {
        var bootstrap = global.SETTINGS_BOOTSTRAP;
        if (bootstrap && bootstrap.ui_settings) {
            syncFromServer(bootstrap.ui_settings);
        } else {
            apply(load());
        }
    }

    global.ARUserSettings = {
        get: load,
        apply: apply,
        persist: persist,
        syncFromServer: syncFromServer,
        defaults: copy(DEFAULTS),
    };

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})(window);
