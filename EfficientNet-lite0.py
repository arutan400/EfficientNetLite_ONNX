import tensorflow as tf
import tensorflow_hub as hub

#EfficientNet-lite0 モデルのロード
model_url = "https://tfhub.dev/tensorflow/efficientnet/lite0/classification/2"
model = hub.KerasLayer(model_url, input_shape=(224, 224, 3))

# データセットのロード
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# データセットオブジェクトの作成
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

# データの前処理関数
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # 正規化
    image = tf.image.resize(image, (224, 224))  # リサイズ
    return image, label

# データセットの前処理とバッチ処理
batch_size = 32  # バッチサイズの設定
train_dataset = train_dataset.map(preprocess).batch(batch_size)
test_dataset = test_dataset.map(preprocess).batch(batch_size)

# モデルの構築
model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2", input_shape=(224, 224, 3)),
    tf.keras.layers.Dense(10, activation='softmax')  # CIFAR-10のクラス数に合わせて10
])

# モデルのコンパイル
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

# モデルのトレーニング
model.fit(train_dataset, epochs=5, validation_data=test_dataset)

# モデルの評価
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test accuracy: {test_accuracy}")

# モデルをSavedModel形式で保存
tf.saved_model.save(model, saved_model_dir)

!python -m tf2onnx.convert --saved-model saved_model_dir --output model.onnx --opset 11
