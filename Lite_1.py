import onnxruntime as rt
import numpy as np  # この行を追加

# モデルをロードします。
sess = rt.InferenceSession("efficientnet_lite0.onnx")

# モデルを使用して予測を行います。
# ここではダミーデータを使用しますが、実際には入力データを指定します。
dummy_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
input_name = sess.get_inputs()[0].name
result = sess.run(None, {input_name: dummy_input})

print(result)
