/**
 * 标签页：根据 URL ?tab= 切换面板，更新 tab 高亮
 */
(function () {
    function initHubTabs(root) {
        if (!root) return;
        var tabs = root.querySelectorAll('.hub-tabs a[data-tab]');
        var panels = root.querySelectorAll('.hub-panel[data-panel]');
        if (!tabs.length || !panels.length) return;

        function activate(tabId) {
            tabs.forEach(function (a) {
                a.classList.toggle('is-active', a.getAttribute('data-tab') === tabId);
            });
            panels.forEach(function (p) {
                var on = p.getAttribute('data-panel') === tabId;
                if (on) p.removeAttribute('hidden');
                else p.setAttribute('hidden', 'hidden');
            });
        }

        var params = new URLSearchParams(window.location.search);
        var initial = params.get('tab');
        var first = tabs[0] && tabs[0].getAttribute('data-tab');
        if (!initial || !root.querySelector('.hub-panel[data-panel="' + initial + '"]')) {
            initial = first;
        }
        activate(initial);

        tabs.forEach(function (a) {
            a.addEventListener('click', function (e) {
                var id = a.getAttribute('data-tab');
                if (!id) return;
                e.preventDefault();
                var url = new URL(window.location.href);
                url.searchParams.set('tab', id);
                window.history.replaceState({}, '', url);
                activate(id);
            });
        });
    }

    function boot() {
        document.querySelectorAll('[data-hub-root]').forEach(initHubTabs);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', boot);
    } else {
        boot();
    }
})();
