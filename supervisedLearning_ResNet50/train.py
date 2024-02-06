
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # すべての GPU のメモリ成長を許可する
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpus, 'GPU')
    except RuntimeError as e:
        print(e)

strategy = tf.distribute.MirroredStrategy()

        
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense
from tensorflow.keras.optimizers.legacy import SGD
import matplotlib.pyplot as plt
from keras.layers import BatchNormalization, Dropout
from keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import time
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

classes = ['female', 'male'] #分類するクラス
nb_classes = len(classes)
 
train_data_dir = './dataset/train'
validation_data_dir = './dataset/val'
test_data_dir = './dataset/test'  # テストデータディレクトリのパス

nb_train_samples = 2868
nb_validation_samples = 628
nb_test_samples = 708  # テストデータのサンプル数

img_width, img_height = 416, 416
batch_size = 32
epochs = 50

# rescale 正規化（各画素値の0~1への正規化）
# zoom_rangeは画像をランダムにズーム, horizontal_flipは画像をランダムに左右反転
train_datagen = ImageDataGenerator(rescale=1.0 / 255, zoom_range=0.2, horizontal_flip=True)

# テストデータに対しては，水増しする必要が無いので正規化のみ
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

# テストデータジェネレータ（正規化のみを適用）
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
  train_data_dir,
  target_size=(img_width, img_height),
  color_mode='rgb',
  classes=classes,
  class_mode='categorical',
  batch_size=batch_size)

validation_generator = validation_datagen.flow_from_directory(
  validation_data_dir,
  target_size=(img_width, img_height),
  color_mode='rgb',
  classes=classes,
  class_mode='categorical',
  batch_size=batch_size)

# テストデータジェネレータの設定
test_generator = test_datagen.flow_from_directory(
  test_data_dir,
  target_size=(img_width, img_height),
  color_mode='rgb',
  classes=classes,
  class_mode='categorical',
  shuffle=False)

with strategy.scope():
    # モデルの構築
    input_tensor = Input(shape=(img_width, img_height, 3))
    ResNet50 = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)

    top_model = Sequential()
    top_model.add(Flatten(input_shape=ResNet50.output_shape[1:]))
    top_model.add(BatchNormalization())
    top_model.add(Dropout(0.5)) 
    top_model.add(Dense(nb_classes, activation='sigmoid'))

    model = Model(inputs=ResNet50.input, outputs=top_model(ResNet50.output))

    # モデルのコンパイル
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(learning_rate=1e-6, momentum=0.9),
                  metrics=['accuracy'])



# EarlyStopping コールバックの設定
# early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# モデルのトレーニング前の時間記録
start_time = time.time()

# モデルのトレーニング
steps_per_epoch = int(len(train_generator.classes) / batch_size) 
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator
    # callbacks=[early_stopping] 
)

# モデルのトレーニング後の時間記録
end_time = time.time()

# 学習時間の計算
training_time = end_time - start_time
print(f"Training time: {training_time} seconds")

# 推論時間の計測開始
start_time = time.time()

# モデルの評価（テストデータで）
test_steps = np.ceil(nb_test_samples / batch_size)
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_steps)

# 推論時間の計測終了
end_time = time.time()

# 推論時間の計算
inference_time = end_time - start_time
print(f"Inference time: {inference_time} seconds")

# テストデータに対する予測
test_generator.reset()
predictions = model.predict(test_generator, steps=test_steps)
predicted_classes = np.argmax(predictions, axis=1)

# 実際のラベル
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# 各クラスごとの正解率
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

# 混同行列（オプション）
cm = confusion_matrix(true_classes, predicted_classes)
print(cm)

def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.ylabel('Predicted label')
    plt.xlabel('True label')
    plt.xticks(np.arange(len(class_names)), class_names)
    plt.yticks(np.arange(len(class_names)), class_names)
    plt.show()

# Plot the confusion matrix
plot_confusion_matrix(cm, class_labels)

# Convert training and inference time from seconds to minutes
training_time_min = training_time / 60
inference_time_min = inference_time / 60

print(f"Training time: {training_time_min:.2f} minutes")
print(f"Inference time: {inference_time_min:.2f} minutes")

# テスト結果の表示
print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)

def plot_graph(train_values, valid_values, train_label, valid_label, save_path):
    epochs = len(train_values)  # Set epochs to the actual number of epochs trained
    plt.figure()  # New figure

    plt.plot(range(1, epochs + 1), train_values, label=train_label)
    plt.plot(range(1, epochs + 1), valid_values, label=valid_label)
    plt.xlabel('Epochs')
    plt.ylabel(train_label)
    plt.legend()
    plt.savefig(save_path)  # Save the graph
    plt.close()  # Close the graph

# history オブジェクトから損失と精度のデータを取得
t_losses = history.history['loss']
v_losses = history.history['val_loss']
t_accus = history.history['accuracy']
v_accus = history.history['val_accuracy']

# Call the function with the corrected epoch length
plot_graph(t_losses, v_losses, 'loss(train)', 'loss(validate)', '/workspaces/2023f_ojus/plot/1/train_validation_loss.png')
plot_graph(t_accus, v_accus, 'accuracy(train)', 'accuracy(validate)', '/workspaces/2023f_ojus/plot/1/train_validation_accuracy.png')

model_save_path = '/workspaces/2023f_ojus/model/model_1.h5'
model.save(model_save_path)
