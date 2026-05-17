/**
 * Dashboard 战斗力球：用全局 CSS 变量驱动旋转，避免 AR 立体镜像克隆后动画各自计时不同步。
 */
(function () {
    'use strict';

    var OUTER_PERIOD_S = 20;
    var PULSE_PERIOD_S = 4;
    var startMs = performance.now();
    var rafId = 0;

    function tick(now) {
        var elapsed = (now - startMs) / 1000;
        var root = document.documentElement;
        var outerDeg = ((elapsed % OUTER_PERIOD_S) / OUTER_PERIOD_S) * 360;
        root.style.setProperty('--combat-sphere-outer-deg', outerDeg.toFixed(3) + 'deg');

        var pulseT = (elapsed % PULSE_PERIOD_S) / PULSE_PERIOD_S;
        var glow = 0.2 + 0.3 * (0.5 - 0.5 * Math.cos(pulseT * Math.PI * 2));
        root.style.setProperty('--combat-sphere-glow', String(glow));

        rafId = requestAnimationFrame(tick);
    }

    function start() {
        if (rafId) return;
        startMs = performance.now();
        rafId = requestAnimationFrame(tick);
    }

    function stop() {
        if (rafId) cancelAnimationFrame(rafId);
        rafId = 0;
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', start);
    } else {
        start();
    }

    document.addEventListener('visibilitychange', function () {
        if (document.hidden) stop();
        else start();
    });
})();
