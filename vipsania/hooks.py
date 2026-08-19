from typing import TypeVar

import tensorflow as tf

T_Model = TypeVar(
    "T_Model",
    bound=(tf.keras.Model | tf.keras.Layer),
)


def with_hooks(model_cls: type[T_Model]) -> type[T_Model]:

    def hook(self, name: str, x: tf.Tensor) -> tf.Tensor:
        if not hasattr(self, "_hooks_active"):
            self._hooks_active = False
        if not hasattr(self, "hooks"):
            self.hooks = {}
        if self._hooks_active:
            self.hooks.update({name: tf.identity(x)})
        return x

    def attach_hooks(self) -> None:
        self._hooks_active = True

    def release_hooks(self) -> None:
        self._hooks_active = False

    def clear_hooks(self) -> None:
        self.hooks = {}

    setattr(model_cls, "hook", hook)
    setattr(model_cls, "attach_hooks", attach_hooks)
    setattr(model_cls, "release_hooks", release_hooks)
    setattr(model_cls, "clear_hooks", clear_hooks)
    return model_cls
