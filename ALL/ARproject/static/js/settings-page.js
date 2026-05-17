/**
 * 设置页：收集 ui_settings、保存到服务端、账户操作
 */
(function () {
    'use strict';

    var bootstrap = window.SETTINGS_BOOTSTRAP || {};
    var ui = copy(window.ARUserSettings ? window.ARUserSettings.get() : {});

    function copy(o) {
        var c = {};
        Object.keys(o || {}).forEach(function (k) {
            c[k] = o[k];
        });
        return c;
    }

    function csrfToken() {
        var el = document.querySelector('[name=csrfmiddlewaretoken]');
        return el ? el.value : '';
    }

    function postJson(url, body) {
        return fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken(),
            },
            credentials: 'same-origin',
            body: JSON.stringify(body || {}),
        }).then(function (r) {
            return r.json().then(function (data) {
                if (!r.ok) {
                    var err = new Error((data && data.error) || 'request_failed');
                    err.payload = data;
                    throw err;
                }
                return data;
            });
        });
    }

    function toast(msg, ok) {
        var bar = document.getElementById('settings-toast');
        if (!bar) return;
        bar.textContent = msg;
        bar.hidden = false;
        bar.classList.toggle('is-error', !ok);
        bar.classList.toggle('is-ok', ok !== false);
        clearTimeout(toast._t);
        toast._t = setTimeout(function () {
            bar.hidden = true;
        }, 3200);
    }

    function syncSlider(slider) {
        if (!slider) return;
        var container = slider.closest('.slider-control');
        if (!container) return;
        var fill = container.querySelector('.slider-fill');
        var label = container.querySelector('.slider-value');
        var min = Number(slider.min);
        var max = Number(slider.max);
        var val = Number(slider.value);
        var pct = ((val - min) / (max - min)) * 100;
        if (fill) fill.style.width = pct + '%';
        if (label) {
            if (slider.getAttribute('data-slider-unit') === 'fps') {
                label.textContent = val + ' FPS';
            } else {
                label.textContent = val + '%';
            }
        }
    }

    function bindSliders() {
        document.querySelectorAll('[data-setting][type="range"]').forEach(function (slider) {
            syncSlider(slider);
            slider.addEventListener('input', function () {
                syncSlider(slider);
                var key = slider.getAttribute('data-setting');
                if (!key) return;
                var val = Number(slider.value);
                if (key === 'gesture_sensitivity' || key === 'hud_opacity') {
                    ui[key] = val;
                } else if (key === 'fps_limit') {
                    ui[key] = val;
                }
                if (key === 'hud_opacity' && window.ARUserSettings) {
                    var draft = copy(ui);
                    window.ARUserSettings.apply(draft);
                }
            });
        });
    }

    function bindSelects() {
        document.querySelectorAll('select[data-setting]').forEach(function (sel) {
            sel.addEventListener('change', function () {
                var key = sel.getAttribute('data-setting');
                if (key) ui[key] = sel.value;
                if (key === 'theme' && window.ARUserSettings) {
                    window.ARUserSettings.apply(copy(ui));
                }
            });
        });
    }

    function bindDisplayToggles() {
        document.querySelectorAll('[data-setting][type="checkbox"]').forEach(function (cb) {
            cb.addEventListener('change', function () {
                var key = cb.getAttribute('data-setting');
                if (!key) return;
                ui[key] = cb.checked;
                if (window.ARUserSettings) {
                    window.ARUserSettings.apply(copy(ui));
                }
            });
        });
    }

    function populateForm() {
        document.querySelectorAll('[data-setting]').forEach(function (el) {
            var key = el.getAttribute('data-setting');
            if (!key || ui[key] === undefined) return;
            if (el.type === 'checkbox') {
                el.checked = !!ui[key];
            } else if (el.tagName === 'SELECT') {
                el.value = String(ui[key]);
            } else if (el.type === 'range') {
                el.value = String(ui[key]);
                syncSlider(el);
            }
        });

        var avatarEl = document.getElementById('settings-avatar-display');
        if (avatarEl && ui.avatar_emoji) {
            avatarEl.textContent = ui.avatar_emoji;
        }
    }

    function collectUiSettings() {
        var out = copy(ui);
        document.querySelectorAll('[data-setting]').forEach(function (el) {
            var key = el.getAttribute('data-setting');
            if (!key) return;
            if (el.type === 'checkbox') {
                out[key] = el.checked;
            } else if (el.type === 'range') {
                out[key] = Number(el.value);
            } else if (el.tagName === 'SELECT') {
                out[key] = el.value;
            }
        });
        return out;
    }

    function saveSettings() {
        var btn = document.querySelector('.btn-save');
        var payload = collectUiSettings();
        if (btn) {
            btn.disabled = true;
            btn.textContent = 'SAVING...';
        }
        return postJson(bootstrap.urls.save, payload)
            .then(function (data) {
                ui = copy(data.ui_settings || payload);
                if (window.ARUserSettings) {
                    window.ARUserSettings.syncFromServer(ui);
                }
                toast('设置已保存', true);
                if (btn) {
                    btn.textContent = 'SAVED ✓';
                    btn.classList.add('is-saved');
                    setTimeout(function () {
                        btn.textContent = 'SAVE_SETTINGS';
                        btn.classList.remove('is-saved');
                        btn.disabled = false;
                    }, 2000);
                }
            })
            .catch(function (err) {
                toast((err.payload && err.payload.error) || err.message || '保存失败', false);
                if (btn) {
                    btn.textContent = 'SAVE_SETTINGS';
                    btn.disabled = false;
                }
            });
    }

    function openAvatarPicker() {
        var modal = document.getElementById('settings-avatar-modal');
        if (!modal) return;
        var grid = modal.querySelector('.avatar-picker-grid');
        if (grid && !grid.childElementCount) {
            (bootstrap.avatar_presets || []).forEach(function (emoji) {
                var b = document.createElement('button');
                b.type = 'button';
                b.className = 'avatar-pick-btn';
                b.textContent = emoji;
                b.setAttribute('data-emoji', emoji);
                b.addEventListener('click', function () {
                    ui.avatar_emoji = emoji;
                    var display = document.getElementById('settings-avatar-display');
                    if (display) display.textContent = emoji;
                    modal.hidden = true;
                });
                grid.appendChild(b);
            });
        }
        modal.hidden = false;
    }

    function openPasswordModal() {
        var modal = document.getElementById('settings-password-modal');
        if (modal) modal.hidden = false;
    }

    function bindAccountActions() {
        var avatarBtn = document.getElementById('btn-change-avatar');
        if (avatarBtn) {
            avatarBtn.addEventListener('click', openAvatarPicker);
        }

        document.querySelectorAll('[data-dismiss-modal]').forEach(function (btn) {
            btn.addEventListener('click', function () {
                var id = btn.getAttribute('data-dismiss-modal');
                var modal = document.getElementById(id);
                if (modal) modal.hidden = true;
            });
        });

        var pwdBtn = document.getElementById('btn-change-password');
        if (pwdBtn) pwdBtn.addEventListener('click', openPasswordModal);

        var pwdForm = document.getElementById('settings-password-form');
        if (pwdForm) {
            pwdForm.addEventListener('submit', function (e) {
                e.preventDefault();
                var fd = new FormData(pwdForm);
                postJson(bootstrap.urls.changePassword, {
                    old_password: fd.get('old_password'),
                    new_password: fd.get('new_password'),
                    confirm_password: fd.get('confirm_password'),
                })
                    .then(function () {
                        pwdForm.reset();
                        document.getElementById('settings-password-modal').hidden = true;
                        toast('密码已更新', true);
                    })
                    .catch(function (err) {
                        toast((err.payload && err.payload.error) || '修改失败', false);
                    });
            });
        }

        var exportBtn = document.getElementById('btn-export-data');
        if (exportBtn) {
            exportBtn.addEventListener('click', function () {
                window.location.href = bootstrap.urls.export;
            });
        }

        var deleteBtn = document.getElementById('btn-delete-account');
        if (deleteBtn) {
            deleteBtn.addEventListener('click', function () {
                var modal = document.getElementById('settings-delete-modal');
                if (modal) modal.hidden = false;
            });
        }

        var deleteForm = document.getElementById('settings-delete-form');
        if (deleteForm) {
            deleteForm.addEventListener('submit', function (e) {
                e.preventDefault();
                if (!confirm('最后确认：账户将被永久删除，无法恢复。')) return;
                var fd = new FormData(deleteForm);
                postJson(bootstrap.urls.deleteAccount, {
                    password: fd.get('password'),
                    confirm_username: fd.get('confirm_username'),
                })
                    .then(function (data) {
                        window.location.href = data.redirect || '/login/';
                    })
                    .catch(function (err) {
                        toast((err.payload && err.payload.error) || '注销失败', false);
                    });
            });
        }
    }

    function init() {
        if (bootstrap.ui_settings) {
            ui = copy(bootstrap.ui_settings);
        }
        populateForm();
        bindSliders();
        bindSelects();
        bindDisplayToggles();
        bindAccountActions();

        var saveBtn = document.querySelector('.btn-save');
        if (saveBtn) {
            saveBtn.addEventListener('click', function () {
                saveSettings();
            });
        }

        if (window.ARUserSettings) {
            window.ARUserSettings.apply(copy(ui));
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
