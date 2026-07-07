// ==============================================
// AR REALM - scanner.js
// 作者: 系统核心
// 目的: 驱动真实的摄像头流，并提供帧率监控与手势交互
// ==============================================

// 全局状态
let videoStream = null;
let animationFrameId = null;
let frameCount = 0;
let lastFrameTime = 0;
let beFPS = 0; // 由后端更新的帧率

// DOM 元素引用
const scannerStream = document.getElementById('scanner-stream');
const beFPSDisplay = document.getElementById('be-fps');
const feFPSDisplay = document.getElementById('fe-fps');
const latencyDisplay = document.getElementById('latency');
const gestureHint = document.getElementById('gesture-hint');

// 初始化
document.addEventListener('DOMContentLoaded', async () => {
    try {
        // 1. 请求真实的摄像头流（彩色视频）
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                // 如果你后续要用深度流，这里可以改成 { deviceId: { exact: 'depth-device-id' } }
                facingMode: 'environment'
            },
            audio: false
        });

        videoStream = stream;
        scannerStream.srcObject = stream;
        scannerStream.play();

        console.log('[SCAN] ✅ 已成功获取真实摄像头流');

        // 2. 启动帧率监控
        startFPSMonitor();

        // 3. 启动手势监听
        setupGestureListener();

        // 4. 启动“目标识别”模拟（点击屏幕）
        setupTargetDetection();

        // 5. 监听来自后端的帧率更新（预留接口）
        window.addEventListener('message', (event) => {
            if (event.data.type === 'BE_FPS_UPDATE') {
                beFPS = event.data.fps;
                beFPSDisplay.textContent = beFPS.toFixed(1);
            }
        });

    } catch (err) {
        console.error('[SCAN] ❌ 获取摄像头失败:', err);
        beFPSDisplay.textContent = 'ERR';
        feFPSDisplay.textContent = 'ERR';
        gestureHint.textContent = '[权限被拒] 请授予摄像头权限';
    }
});

// ==============================================
// 核心功能模块
// ==============================================

// 1. 帧率监控
function startFPSMonitor() {
    const now = performance.now();
    const elapsed = now - lastFrameTime;

    if (elapsed > 1000) {
        // 一秒过去了，计算 FPS
        const fps = (frameCount * 1000) / elapsed;
        feFPSDisplay.textContent = fps.toFixed(1);
        frameCount = 0;
        lastFrameTime = now;
    }

    frameCount++;
    animationFrameId = requestAnimationFrame(startFPSMonitor);
}

// 2. 手势监听（依赖 gesture.js）
function setupGestureListener() {
    window.addEventListener('ar-gesture', (e) => {
        const { action, message } = e.detail;

        if (action === 'grab') {
            console.log('[GESTURE] 捕获到 "抓取" 动作');
            // 尝试寻找最近的战利品箱
            const loot = document.querySelector('.hologram-chest');
            if (loot) {
                openLootBox(loot);
            }
        } else if (action === 'swipe-right') {
            console.log('[GESTURE] 捕获到 "向右滑动" 动作');
            // 可能用于切换扫描模式
        }
    });
}

// 3. 目标识别模拟（点击屏幕）
function setupTargetDetection() {
    scannerStream.addEventListener('click', (e) => {
        // 计算点击位置的相对坐标
        const rect = scannerStream.getBoundingClientRect();
        const x = ((e.clientX - rect.left) / rect.width) * 100;
        const y = ((e.clientY - rect.top) / rect.height) * 100;

        // 在该位置生成一个战利品箱
        spawnLootBox(x, y);
    });
}

// 4. 生成战利品箱（全息投影）
function spawnLootBox(xPercent, yPercent) {
    const container = document.querySelector('.scanner-viewport');

    const chest = document.createElement('div');
    chest.className = 'hologram-chest';
    chest.style.left = `${xPercent}%`;
    chest.style.top = `${yPercent}%`;
    chest.style.transform = 'translate(-50%, -50%)';
    chest.style.zIndex = '9999';

    chest.innerHTML = `
        <div class="chest-glow" style="width: 120px; height: 120px; border-radius: 50%; background: radial-gradient(circle, rgba(0, 242, 255, 0.3) 0%, transparent 70%);"></div>
        <div class="chest-body" style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; width: 100px;">
            <div style="font-size: 40px; margin-bottom: 8px;">📦</div>
            <div style="color: #00f2ff; font-weight: bold;">稀有战利品</div>
        </div>
    `;

    container.appendChild(chest);

    // 添加点击交互
    chest.addEventListener('click', (e) => {
        e.stopPropagation();
        openLootBox(chest);
    });

    // 添加手势交互
    chest.addEventListener('touchstart', (e) => {
        e.preventDefault();
        openLootBox(chest);
    });

    // 5秒后自动消失（模拟扫描完成）
    setTimeout(() => {
        if (chest.parentNode) {
            chest.classList.add('fade-out');
            setTimeout(() => chest.remove(), 300);
        }
    }, 5000);
}

// 5. 打开战利品箱
function openLootBox(chest) {
    chest.classList.add('opening');

    // 模拟开箱动画
    setTimeout(() => {
        // 生成战利品数据
        const loot = {
            name: '神经反应增强剂 v2.0',
            rarity: 'rare',
            quantity: 1,
            value: 1200
        };

        // 通知全局：玩家获得战利品
        window.dispatchEvent(new CustomEvent('loot-found', { detail: loot }));

        // 通知后端（可选）
        if (window.parent && typeof window.parent.postMessage === 'function') {
            window.parent.postMessage({
                type: 'LOOT_ACQUIRED',
                data: loot
            }, '*');
        }

        // 移除箱子
        chest.remove();
    }, 800);
}
/**
 * ============================================================
 * AR REALM - TACTICAL SCANNER CORE (scanner.js)
 * 状态：[DECOUPLED FROM HARDWARE] - 由后端驱动图像处理
 * ============================================================
 */

// 全局状态监控
frameCount = 0;
let lastFrameTime = performance.now();
let feFPS = 0;

// DOM 元素引用
const elements = {
    stream: document.getElementById('scanner-stream'),
    beFPS: document.getElementById('be-fps'),
    feFPS: document.getElementById('fe-fps'),
    latency: document.getElementById('latency'),
    gestureHint: document.getElementById('gesture-hint'),
    depthVal: document.getElementById('depth-val'),
    posCoord: document.getElementById('pos-coord')
};

/**
 * 核心初始化函数
 */
document.addEventListener('DOMContentLoaded', () => {
    console.log('[SYSTEM] 🛡️ 战术终端已启动。模式：后端流同步 (MJPEG)');

    // 1. 启动帧率监控 (FE FPS)
    startFPSMonitor();

    // 2. 启动坐标与深度模拟逻辑
    updateEnvData();

    // 3. 启动点击目标锁定 (生成战术箱子)
    setupTargetAcquisition();

    // 4. 后端状态同步通知 (模拟)
    elements.beFPS.textContent = "30.0"; // 设定为相机标准帧率
    elements.latency.textContent = "12ms";
    elements.gestureHint.textContent = "[ 系统就绪：点击屏幕锁定深度目标 ]";
});

/**
 * [性能监控] 计算前端显示刷新率 (FE FPS)
 */
function startFPSMonitor() {
    const now = performance.now();
    const elapsed = now - lastFrameTime;

    if (elapsed > 1000) {
        feFPS = (frameCount * 1000) / elapsed;
        if (elements.feFPS) elements.feFPS.textContent = feFPS.toFixed(1);
        frameCount = 0;
        lastFrameTime = now;
    }

    frameCount++;
    requestAnimationFrame(startFPSMonitor);
}

/**
 * [环境数据] 模拟深度数据波动 (实际项目通常由后端 WebSocket 传回)
 */
function updateEnvData() {
    setInterval(() => {
        const fakeDepth = (1.2 + Math.random() * 0.05).toFixed(2);
        if (elements.depthVal) elements.depthVal.textContent = `${fakeDepth} M (STABLE)`;

        // 模拟漂移偏差
        if (elements.posCoord) {
            elements.posCoord.textContent = `N 30.21${Math.floor(Math.random()*10)} / E 114.00${Math.floor(Math.random()*10)}`;
        }
    }, 1500);
}

/**
 * [战术锁定] 点击画面生成全息战利品箱
 */
function setupTargetAcquisition() {
    if (!elements.stream) return;

    elements.stream.addEventListener('click', (e) => {
        // 计算点击在流画面上的相对百分比坐标
        const rect = elements.stream.getBoundingClientRect();
        const x = ((e.clientX - rect.left) / rect.width) * 100;
        const y = ((e.clientY - rect.top) / rect.height) * 100;

        console.log(`[SCAN] 🔒 目标锁定坐标: ${x.toFixed(1)}%, ${y.toFixed(1)}%`);
        spawnLootBox(x, y);
    });
}

/**
 * [全息投影] 在指定坐标生成 UI 元素
 */
function spawnLootBox(x, y) {
    const container = document.querySelector('.scanner-viewport');

    // 创建全息容器
    const chest = document.createElement('div');
    chest.className = 'loot-container';
    chest.style.left = `${x}%`;
    chest.style.top = `${y}%`;
    chest.style.display = 'block';
    chest.style.position = 'absolute';
    chest.style.transform = 'translate(-50%, -50%)';
    chest.style.transition = 'all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275)';
    chest.style.zIndex = '100';

    // 内部全息视觉
    chest.innerHTML = `
        <div class="hologram-effect" style="
            width: 120px; 
            height: 120px; 
            border: 2px solid #00f2ff; 
            background: rgba(0, 242, 255, 0.1);
            border-radius: 10px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            box-shadow: 0 0 20px rgba(0, 242, 255, 0.5);
            backdrop-filter: blur(2px);">
            <div style="font-size: 40px; filter: drop-shadow(0 0 10px #00f2ff);">📦</div>
            <div style="color: #00f2ff; font-weight: bold; font-size: 10px; margin-top: 5px; text-shadow: 0 0 5px #000;">RECOVERY_OBJ</div>
            <div style="font-family: monospace; font-size: 8px; color: rgba(0,242,255,0.7);">ID: ${Math.random().toString(36).substr(2, 6).toUpperCase()}</div>
        </div>
    `;

    container.appendChild(chest);

    // 点击开箱互动
    chest.addEventListener('click', (e) => {
        e.stopPropagation(); // 防止触发底层的识别事件
        openLootBox(chest);
    });

    // 8秒后如果没捡走就自动消失（模拟扫描过期）
    setTimeout(() => {
        if (chest && chest.parentNode) {
            chest.style.opacity = '0';
            chest.style.transform = 'translate(-50%, -100%) scale(0.5)';
            setTimeout(() => chest.remove(), 500);
        }
    }, 8000);
}

/**
 * [交互逻辑] 打开战利品
 */
function openLootBox(chest) {
    // 增加打开动画
    const inner = chest.querySelector('.hologram-effect');
    inner.style.borderColor = '#00ff88';
    inner.style.boxShadow = '0 0 40px rgba(0, 255, 136, 0.8)';
    inner.innerHTML = `
        <div style="font-size: 40px; animation: bounce 0.5s infinite;">💎</div>
        <div style="color: #00ff88; font-weight: bold; font-size: 12px;">ACQUIRED</div>
    `;

    // 延时移除并通知控制台
    setTimeout(() => {
        console.log('[SYSTEM] ✅ 物品已存入玩家神经网络背包');
        chest.remove();

        // 触发全局自定义事件（可以被其他 JS 模块监听）
        window.dispatchEvent(new CustomEvent('loot-found', {
            detail: { name: '深度结晶', value: 100 }
        }));
    }, 1200);
}

// 动画辅助样式
const style = document.createElement('style');
style.textContent = `
    @keyframes bounce { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-10px); } }
    .loot-container:hover { cursor: crosshair; filter: brightness(1.3); }
`;
document.head.appendChild(style);
// 6. 清理函数（页面卸载时调用）
window.addEventListener('beforeunload', () => {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
    }
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
    }
});