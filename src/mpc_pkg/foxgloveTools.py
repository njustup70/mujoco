import foxglove
from dataclasses import asdict, fields, is_dataclass
from typing import Any, get_args, get_origin


class FoxgloveVisual:
    """Foxglove 可视化发送器（JSON 编码）。

    设计目标：
    1. 外部只需要调用 send(msg)，不需要手动管理 Channel。
    2. 以消息类型名（__qualname__）作为 topic 键，首次发送时自动注册。
    3. 注册后复用同一个 Channel，避免重复注册和重复构建 schema。
    4. 自动把 Python 对象归一化为可 JSON 序列化的数据结构。
    5. 自动推导 schema；当消息顶层是数组时，强制包装成 object 根节点。
    """

    def __init__(self, port):
        self.server = foxglove.start_server(port=port)
        # 缓存结构：{message_name: foxglove.Channel}
        # message_name 默认使用消息类型的 __qualname__。
        # 作用是避免每次 send 都重复创建 Channel。
        self._channels = {}

    def _message_name(self, msg: Any) -> str:
        # 若传入的是类型对象本身（极少见），优先取它的 __qualname__；
        # 常规情况下 msg 是实例对象，取 msg.__class__.__qualname__。
        if hasattr(msg, "__qualname__"):
            return msg.__qualname__
        return msg.__class__.__qualname__

    def _wrap_root_payload(self, payload: Any) -> Any:
        # 业务要求：如果最外层是数组，也要以 object 作为 schema 根节点。
        # 因此把顶层 list 包装成 {"data": [...]}，保证 payload 与 schema 一致。
        # 注意：只包装“顶层”数组，内部字段里的数组保持原样。
        if isinstance(payload, list):
            return {"data": payload}
        return payload

    def _normalize_message(self, msg: Any) -> Any:
        # 将常见 Python 对象归一化为“可直接 JSON 化”的结构。
        # 处理顺序：dataclass -> pydantic(model_dump) -> dict/list/tuple -> 普通对象。
        # 普通对象仅提取公开字段（过滤以下划线开头的内部属性）。
        if hasattr(msg, "tolist") and callable(getattr(msg, "tolist")):
            return msg.tolist()
        if is_dataclass(msg) and not isinstance(msg, type):
            return asdict(msg)
        model_dump = getattr(msg, "model_dump", None)
        if callable(model_dump):
            return model_dump()
        if isinstance(msg, dict):
            return {key: self._normalize_message(value) for key, value in msg.items()}
        if isinstance(msg, (list, tuple)):
            return [self._normalize_message(value) for value in msg]
        if hasattr(msg, "__dict__") and not isinstance(msg, type):
            return {
                key: self._normalize_message(value)
                for key, value in vars(msg).items()
                if not key.startswith("_")
            }
        return msg

    def _schema_from_annotation(self, annotation: Any) -> dict[str, Any] | None:
        # 基于类型注解推导 schema（优先策略，稳定且可控）。
        # 支持：基础类型、list/tuple、dict、dataclass、带 __annotations__ 的类。
        # 无法识别时返回 None，交给值推导逻辑兜底。
        if annotation in (int,):
            return {"type": "integer"}
        if annotation in (float,):
            return {"type": "number"}
        if annotation in (bool,):
            return {"type": "boolean"}
        if annotation in (str,):
            return {"type": "string"}

        origin = get_origin(annotation)
        args = get_args(annotation)

        if origin in (list, tuple):
            item_annotation = args[0] if args else Any
            item_schema = self._schema_from_annotation(item_annotation) or {"type": "object"}
            schema = {"type": "array", "items": item_schema}
            if origin is tuple and len(args) > 1 and args[-1] is not Ellipsis:
                schema["minItems"] = len(args)
                schema["maxItems"] = len(args)
            return schema

        if origin is dict:
            value_annotation = args[1] if len(args) > 1 else Any
            return {
                "type": "object",
                "additionalProperties": self._schema_from_annotation(value_annotation) or {"type": "object"},
            }

        if isinstance(annotation, type):
            if is_dataclass(annotation):
                properties = {}
                required = []
                for field in fields(annotation):
                    properties[field.name] = self._schema_from_annotation(field.type) or {"type": "object"}
                    required.append(field.name)
                return {"type": "object", "properties": properties, "required": required}
            if hasattr(annotation, "__annotations__"):
                properties = {}
                required = []
                for name, field_annotation in annotation.__annotations__.items():
                    if name.startswith("_"):
                        continue
                    properties[name] = self._schema_from_annotation(field_annotation) or {"type": "object"}
                    required.append(name)
                return {"type": "object", "properties": properties, "required": required}

        return None

    def _schema_from_value(self, value: Any) -> dict[str, Any]:
        # 运行时值推导 schema（兜底策略）。
        # 说明：list 的 items 类型取首元素推断；空 list 无法推断具体类型，
        # 因此默认 items 为 object。
        if isinstance(value, bool):
            return {"type": "boolean"}
        if isinstance(value, int) and not isinstance(value, bool):
            return {"type": "integer"}
        if isinstance(value, float):
            return {"type": "number"}
        if isinstance(value, str):
            return {"type": "string"}
        if value is None:
            return {"type": "null"}
        if isinstance(value, dict):
            properties = {}
            for key, item in value.items():
                properties[str(key)] = self._schema_from_value(item)
            return {"type": "object", "properties": properties}
        if isinstance(value, list):
            if not value:
                return {"type": "array", "items": {"type": "object"}}
            return {"type": "array", "items": self._schema_from_value(value[0])}
        return {"type": "object"}

    def _build_schema(self, msg: Any) -> dict[str, Any]:
        # schema 生成优先级：
        # 1) dataclass/类注解推导
        # 2) 运行时值推导
        # 在值推导前先做顶层数组包装，确保根节点始终是 object。
        if is_dataclass(msg) and not isinstance(msg, type):
            schema = self._schema_from_annotation(type(msg))
            if schema is not None:
                return schema

        annotations = getattr(type(msg), "__annotations__", None)
        if annotations:
            schema = self._schema_from_annotation(type(msg))
            if schema is not None:
                return schema

        normalized = self._wrap_root_payload(self._normalize_message(msg))
        return self._schema_from_value(normalized)

    def send(self, msg: Any, topic: str | None = None):
        # 发送主流程：
        # 1) 解析消息名（topic）
        # 2) 归一化 payload，并按规则包装顶层数组
        # 3) 若该 topic 未注册：自动创建 Channel 并附带 schema
        # 4) 已注册则直接复用并发送
        try:
            if topic is None:
                topic = self._message_name(msg)
            
            payload = self._wrap_root_payload(self._normalize_message(msg))
            channel = self._channels.get(topic)
            # test=self._build_schema(msg)
            # print(test)
            if channel is None:
                channel = foxglove.Channel(
                    topic=topic,
                    message_encoding='json',
                    schema=self._build_schema(msg),
                )
                self._channels[topic] = channel

            channel.log(payload)
        except Exception as e:
            print(f"Error sending message to Foxglove: {e}")


'''
my_schema = {
    "type": "object",
    "properties": {
        "v_bus": {"type": "number"},  
        "i_bus": {"type": "number"}, 
        "position": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"}
            }
        }
    }
}
imu_schema = {
    "type": "object",
    "properties": {
        "gyro": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 3,
            "maxItems": 3,
            "description": "角速度 [x, y, z]"
        }
    }
}
self.channel=foxglove.Channel(
    topic="/test",
    message_encoding='json',
    schema=my_schema
)
self.imu_channel=foxglove.Channel(
    topic="/imu",
    message_encoding='json',
    schema=imu_schema
)
foxglove.log('/test',{'b':2.0})
msg_data = {
        "v_bus": 12.5,
        "i_bus": 1.2,
        "position": {"x": 10.0, "y": 20.0}
    }
self.channel.log(msg_data)
# Log IMU data
imu_data = {
    "gyro": [0.1, 0.2, 0.3]
}
self.imu_channel.log(imu_data)
'''
foxgloveViusalInstance=FoxgloveVisual(port=8766)