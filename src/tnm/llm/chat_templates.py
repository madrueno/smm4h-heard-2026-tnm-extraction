"""Chat template overrides for models that need assistant mask support."""


class ChatTemplateOverrides:
    _TEMPLATES = {
        'Qwen/Qwen3.5-27B': """{% for message in messages %}{% if message['role'] == 'system' %}<|im_start|>system
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'user' %}<|im_start|>user
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'assistant' %}<|im_start|>assistant
{% generation %}{{ message['content'] }}{% endgeneration %}<|im_end|>
{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""
    }

    @classmethod
    def get(cls, model_name: str) -> str | None:
        return cls._TEMPLATES.get(model_name)
