import tensorflow as tf
import tf2onnx

# EfficientNetLite のバージョンを定義します。
versions = [
    'lite0', 
    'lite1', 
    'lite2', 
    'lite3', 
    'lite4'
]

for version in versions:
    # モデルをロードします。
    model = tf.keras.models.load_model(f"efficientnet_{version}")

    # モデルを ONNX 形式に変換します。
    onnx_model, _ = tf2onnx.convert.from_keras(model)

    # ONNX モデルを保存します。
    with open(f"efficientnet_{version}.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
