/**
 * AR REALM - 神经网关 核心驱动 v3.0 (修复漂移与消失)
 */

// 1. 注入赛博准星 CSS (确定的坐标系)
const cursorStyles = `
    #ar-cursor {
        position: fixed;
        left: 0;
        top: 0;
        width: 18px;
        height: 18px;
        border-radius: 50%;
        background-color: #00f2ff;
        box-shadow: 0 0 15px #00f2ff, 0 0 30px #00f2ff;
        pointer-events: none;
        z-index: 2147483647; /* 置于一切之上 */
        transform: translate(-50%, -50%); /* 核心：让(left,top)落在圆心 */
        transition: width 0.15s cubic-bezier(0.175, 0.885, 0.32, 1.275), 
                    height 0.15s cubic-bezier(0.175, 0.885, 0.32, 1.275), 
                    background-color 0.15s, box-shadow 0.15s, border 0.15s;
        box-sizing: border-box;
        border: 2px solid transparent;
        display: none; /* 初始隐藏 */
    }

    /* 锁定状态：变成大白圈 */
    #ar-cursor.cursor-locked {
        width: 48px !important;
        height: 48px !important;
        background-color: transparent !important;
        border: 3px solid #fff !important;  
        box-shadow: 0 0 20px rgba(255,255,255,0.8), inset 0 0 15px rgba(255,255,255,0.5) !important;
    }
`;

const styleSheet = document.createElement("style");
styleSheet.innerText = cursorStyles;
document.head.appendChild(styleSheet);

class ARGestureController {
    constructor() {
        this.statusText = document.getElementById('gesture-status-text');
        this.ws = null;
        this.isActive = false;
        this.lastX = 0;
        this.lastY = 0;
        this.lastClickTime = 0;
        this.lastHoveredElement = null;

        // 包含榜单、市场、中心页面的所有交互选择器
        this.selectors = 'a, button, input, .realm-card, .market-card, .btn-action, .loot-item, .rank-item, .btn-link';

        this.createCursor();
        this.start();
        console.log("[AR System] 坐标驱动器就绪");
    }

    createCursor() {
        // 如果页面上已经有了，先删掉防止冲突
        const oldCursor = document.getElementById('ar-cursor');
        if(oldCursor) oldCursor.remove();

        this.cursor = document.createElement('div');
        this.cursor.id = 'ar-cursor';
        document.body.appendChild(this.cursor);
    }

    start() {
        if (this.isActive) return;
        const wsProtocol = window.location.protocol === "https:" ? "wss://" : "ws://";
        const wsUrl = wsProtocol + window.location.host + '/ws/gestures/';

        this.ws = new WebSocket(wsUrl);
        this.ws.onopen = () => {
            this.isActive = true;
            if (this.statusText) {
                this.statusText.innerText = "LINK: ONLINE";
                this.statusText.style.color = "#00f2ff";
            }
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleData(data);
        };

        this.ws.onclose = () => {
             this.isActive = false;
             if (this.statusText) {
                 this.statusText.innerText = "LINK: LOST";
                 this.statusText.style.color = "#ff3e3e";
             }
             setTimeout(() => this.start(), 3000);
        };
    }

    handleData(data) {
        // 1. 处理右手光标
        if (data.right === 'point' && data.pointer) {
            // 🌟 核心：计算最纯粹的屏幕坐标
            const x = Math.round(data.pointer.x * window.innerWidth);
            const y = Math.round(data.pointer.y * window.innerHeight);

            this.cursor.style.display = 'block';

            // 🌟 直接修改 left/top，回避 calc()
            this.cursor.style.left = x + 'px';
            this.cursor.style.top = y + 'px';

            this.lastX = x;
            this.lastY = y;

            // 检测悬停
            this.updateHover(x, y);

        } else if (data.right === 'pinch' && this.lastX) {
            // 捏合：通过 CSS scale 快速反馈，不再改 left/top
            this.cursor.style.transform = 'translate(-50%, -50%) scale(0.6)';
            this.cursor.style.backgroundColor = '#ff3e3e';
            this.doClick(this.lastX, this.lastY);
        } else {
            // 如果手势消失，恢复状态
            if(this.cursor.style.display !== 'none'){
                this.cursor.style.display = 'none';
                this.cursor.style.transform = 'translate(-50%, -50%) scale(1)';
                this.clearHover();
            }
        }

        // 2. 左手滚动
        if (data.scroll === 'down') window.scrollBy({ top: 150, behavior: 'smooth' });
        if (data.scroll === 'up') window.scrollBy({ top: -150, behavior: 'smooth' });
    }

    updateHover(x, y) {
        // 获取当前坐标下的元素
        const target = document.elementFromPoint(x, y);
        const interactive = target ? target.closest(this.selectors) : null;

        if (this.lastHoveredElement !== interactive) {
            this.clearHover();
            if (interactive) {
                interactive.classList.add('ar-hover');
                this.cursor.classList.add('cursor-locked');
                this.lastHoveredElement = interactive;
            }
        }
    }

    clearHover() {
        if (this.lastHoveredElement) {
            this.lastHoveredElement.classList.remove('ar-hover');
            this.lastHoveredElement = null;
        }
        this.cursor.classList.remove('cursor-locked');
    }

    doClick(x, y) {
        // 防止连点
        if (Date.now() - this.lastClickTime < 600) return;
        this.lastClickTime = Date.now();

        const target = document.elementFromPoint(x, y);
        if (target) {
            const interact = target.closest(this.selectors);
            (interact || target).click();
            console.log("[AR] 点击了:", (interact || target));
        }
    }
}

// 启动
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => new ARGestureController());
} else {
    new ARGestureController();
}