import tensorflow as tf
import tensorflow_hub as hub

# EfficientNetLite のバージョンを定義します。
versions = [
    'efficientnet/lite0', 
    'efficientnet/lite1', 
    'efficientnet/lite2', 
    'efficientnet/lite3', 
    'efficientnet/lite4'
]

for version in versions:
    # EfficientNetLite モデルをダウンロードします。
    handle = f"https://tfhub.dev/tensorflow/{version}/feature-vector/2"
    model = tf.keras.Sequential([
        hub.KerasLayer(handle, trainable=False)  # 'trainable=True' を 'trainable=False' に変更します。
    ])

    # 入力形状を定義してモデルをビルドします。
    model.build((None, 224, 224, 3))

    # モデルを保存します。
    model.save(f"efficientnet_{version.split('/')[-1]}")
