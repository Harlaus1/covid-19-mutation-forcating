import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
from numpy import savetxt

from zipfile import ZipFile
import os

# uri = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
# zip_path = keras.utils.get_file(origin=uri, fname="jena_climate_2009_2016.csv.zip")
# zip_file = ZipFile(zip_path)
# zip_file.extractall()
csv_path = "EachMutationsRate.csv"

df = pd.read_csv(csv_path)
#print(type(df))
#print(df)
titles = [
    "A-T",
    "A-G",
    "A-C",
    "T-A",
    "T-G",
    "T-C",
    "G-A",
    "G-T",
    "G-C",
    "C-A",
    "C-T",
    "C-G",
]

colors = [
    "blue",
    "orange",
    "green",
    "red",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
    "black",
    "yellow",
]

date_time_key = "Date Time"

#visualize mutation data
def show_raw_visualization(data):
    # time_data = data[date_time_key]
    fig, axes = plt.subplots(
        nrows=6, ncols=2, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k"
    )
    for i in range(len(titles)):
        key = titles[i]
        c = colors[i % (len(colors))]
        t_data = data[key]
        # t_data.index = time_data
        t_data.head()
        ax = t_data.plot(
            ax=axes[i // 2, i % 2],
            color=c,
            title="{} - {}".format(titles[i], key),
            rot=25,
        )
        ax.legend([titles[i]])
    plt.tight_layout()
    plt.show()

# show_raw_visualization(df)

def show_heatmap(data):
    plt.matshow(data.corr())
    plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.show()

# show_heatmap(df)

#分割训练集
split_fraction = 0.715
train_split = int(split_fraction * int(df.shape[0]))
#print(train_split)
#每隔1小时记录一次，原代码为每10分钟一条数据
step = 1
#
# print(train_split)

#past多少行为x，future为y，即用但是past预测多少future
past = 12
future = 1
learning_rate = 0.001
batch_size = 128
epochs = 10


def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std



# features = normalize(df.values, train_split)
# features = pd.DataFrame(features)
# features.head()


# split data
# train_data = features.loc[0: train_split - 1]
# val_data = features.loc[train_split:]

train_data = df.loc[0: train_split - 1]
val_data = df.loc[train_split:]

start = past + future
end = start + train_split


# training data and labels
x_train = train_data.values
# y_train = features.iloc[start:end]
y_train = df.iloc[start:end]
#print(y_train.shape)

sequence_length = int(past / step)
#print(sequence_length)

print(x_train.shape)
print(y_train.shape)
dataset_train = keras.preprocessing.timeseries_dataset_from_array(
    x_train,
    y_train,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)

# validation data
# x_end = len(val_data) - past - future
# label_start = train_split + past + future
x_end = len(val_data)
label_start = train_split


#print(label_start)
#print(x_end)
x_val = val_data.iloc[:x_end].values
# y_val = features.iloc[label_start:]
y_val = df.iloc[label_start:]


dataset_val = keras.preprocessing.timeseries_dataset_from_array(
    x_val,
    y_val,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)

# 将数据变为训练形状
for batch in dataset_train.take(1):
    inputs, targets = batch


print("Input shape:", inputs.numpy().shape)
print("Target shape:", targets.numpy().shape)
# print(inputs)
# print(targets)
print(inputs.shape)

# Training
inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))

lstm_out = keras.layers.LSTM(32)(inputs)
outputs = keras.layers.Dense(12)(lstm_out)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
model.summary()

path_checkpoint = "model_checkpoint.h5"

#当模型不能根据标准继续提升会终止训练
es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

#以一定频率和标准保存模型
modelckpt_callback = keras.callbacks.ModelCheckpoint(
    monitor="val_loss",
    filepath=path_checkpoint,
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
)

history = model.fit(
    dataset_train,
    epochs=epochs,
    validation_data=dataset_val,
    callbacks=[es_callback, modelckpt_callback],
)

print(history.history)

# save training data and training fitting data
fitting = model.predict(dataset_train)
savetxt("fitting.csv", fitting, delimiter = ',')

# save validation_prediction data and original_validation data
print("Evaluate on test data")
results = model.evaluate(dataset_val)
print("test loss, test acc:", results)
results_prediction = model.predict(dataset_val)
print(x_val.shape)
savetxt("results_validation.csv", results_prediction, delimiter=',')

#visualization
def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

visualize_loss(history, "Training and Validation Loss")
print("Generative prediction for a few samples")

# define testing dataset to predict future samples

x_test = x_val[:12]
y_test = y_val[:12]

dataset_test = keras.preprocessing.timeseries_dataset_from_array(
    x_test,
    y_test,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)

predictions = model.predict(dataset_test)
predictions_history = predictions

for i in range(100):
    # print(predictions)
    x_test = np.append(x_test, predictions, axis=0)
    for j in range(predictions.shape[0]):
        x_test = np.delete(x_test, j, 0)

    y_test = np.append(y_test, predictions, axis=0)
    for k in range(predictions.shape[0]):
        y_test = np.delete(y_test, k, 0)

    dataset_test = keras.preprocessing.timeseries_dataset_from_array(
        x_test,
        y_test,
        sequence_length=sequence_length,
        sampling_rate=step,
        batch_size=batch_size,
    )
    predictions = model.predict(dataset_test)
    predictions_history = np.append(predictions_history, predictions, axis=0)
    # print(predictions_history)


print("predictions", predictions_history)
print("prediction shape:", predictions_history.shape)
#预测值保存在predicitons.csv中
savetxt("predicitons.csv", predictions_history, delimiter=',')


# original validation
# visualize the validation
def show_plot(plot_data, delta, title):
    labels = ["History", "True Future", "Model Prediction"]
    marker = [".-", "rx", "go"]
    #?
    time_steps = list(range(-(plot_data[0].shape[0]), 0))
    if delta:
        future = delta
    else:
        future = 0
    plt.title(title)
    for i, val in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    #控制图像宽窄，横坐标长度
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel("Time-Step")
    plt.show()
    # return

# print(x_val)
# print(y_val)
# print(model.predict(dataset_train))

#visualization of validation data
# for x_val, y_val in dataset_val:
#     show_plot(
#         [x_val[0][:, 1].numpy(), y_val[0].numpy(), model.predict(dataset_val)[0]],
#         12,
#         "Single Step Prediction",
#     )