import os
import numpy as np
import tensorflow as tf
from data_processing import DescriptorGraphDataset
from model import create_model
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dataset_dir = "dataset\ADM_dataset"
    exp_name = "Inpaint"

    train_dataset = DescriptorGraphDataset(os.path.join(dataset_dir, "train"), mode="train", batch_size=32)
    valid_dataset = DescriptorGraphDataset(os.path.join(dataset_dir, "valid"), mode="valid", batch_size=32)

    # 從第一個有效batch取得 input_dim
    for (X_batch, A_batch), Y_batch in train_dataset:
        if X_batch.shape[0] > 0:
            input_dim = X_batch.shape[-1]
            break

    model = create_model(input_dim)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"models/{exp_name}_best.ckpt",
            save_weights_only=True,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    ]

    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=10,
        callbacks=callbacks
    )

    # 測試階段
    test_dataset = DescriptorGraphDataset(os.path.join(dataset_dir, "test"), mode="test", batch_size=32)

    # 收集test labels
    y_true = []
    for (X_batch, A_batch), Y_batch in test_dataset:
        y_true.extend(Y_batch)
    y_true = np.array(y_true)

    # 再次創建test_dataset以確保能重新迭代 (Sequence對象可能已用盡)
    test_dataset = DescriptorGraphDataset(os.path.join(dataset_dir, "test"), mode="test", batch_size=32)
    preds = model.predict(test_dataset)

    # 將logits轉為機率分數
    preds_prob = tf.sigmoid(preds).numpy().flatten()
    preds_label = (preds_prob > 0.5).astype(int)

    # ROC Curve & AUC
    fpr, tpr, thresholds = roc_curve(y_true, preds_prob)
    roc_auc = auc(fpr, tpr)
    print(f"Testing AUC: {roc_auc:.4f}")

    # 繪製 ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.4f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - Test Set')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()

    # 輸出 submission
    ids = [str(i).zfill(4) for i in range(1, len(y_true) + 1)]
    import pandas as pd
    df = pd.DataFrame({"Id": ids, "Category": preds_label})
    df.to_csv("submission.csv", index=False)
    print("Prediction results saved to submission.csv")
    print("ROC curve saved as roc_curve.png")

