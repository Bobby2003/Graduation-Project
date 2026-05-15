import json

from channels.generic.websocket import AsyncWebsocketConsumer

# 浏览器端完成识别后，将 JSON 结果广播给其他已连接客户端（本机 UI 已在本地处理）
GESTURE_GROUP = "gesture_relay"


class GestureConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.channel_layer.group_add(GESTURE_GROUP, self.channel_name)
        await self.accept()
        print(f"[WS] ✅ 手势中继链路已接入: {self.channel_name}")

    async def disconnect(self, code):
        await self.channel_layer.group_discard(GESTURE_GROUP, self.channel_name)
        print(f"[WS] ❌ 手势链路断开 (code={code})")

    async def receive(self, text_data=None, bytes_data=None):
        if not text_data:
            return
        try:
            json.loads(text_data)
        except json.JSONDecodeError:
            return

        await self.channel_layer.group_send(
            GESTURE_GROUP,
            {
                "type": "gesture.relay",
                "sender": self.channel_name,
                "payload": text_data,
            },
        )

    async def gesture_relay(self, event):
        if event["sender"] == self.channel_name:
            return
        await self.send(text_data=event["payload"])
