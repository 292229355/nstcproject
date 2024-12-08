import os
import numpy as np
import tensorflow as tf
from data_processing import DescriptorGraphDataset
from model import create_model
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dataset_dir = os.path.join("dataset", "ADM_dataset")
    exp_name = "Inpaint"

    # Initialize datasets
    train_dataset = DescriptorGraphDataset(os.path.join(dataset_dir, "train"), mode="train", batch_size=32)
    valid_dataset = DescriptorGraphDataset(os.path.join(dataset_dir, "valid"), mode="valid", batch_size=32)

    # Determine input dimension from the first batch
    for (X_batch, A_batch), Y_batch in train_dataset:
        if X_batch.shape[0] > 0:
            input_dim = X_batch.shape[-1]
            break

    # Create model
    model = create_model(input_dim)

    # Define loss function
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # Define optimizer with gradient clipping
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)

    # Compile model with optimizer, loss, and metrics
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    # Define callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join("models", f"{exp_name}_best.ckpt"),
            save_weights_only=True,
            save_best_only=True,
            monitor='val_auc',
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=10,
            restore_best_weights=True,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_auc',
            factor=0.5,
            patience=5,
            verbose=1,
            mode='max',
            min_lr=1e-6
        )
    ]

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=10,  # Increased epochs for better learning
        callbacks=callbacks,
        verbose=1
    )

    # Save the final model
    model.save(os.path.join("models", f"{exp_name}_final_model.h5"))

    # Testing phase
    test_dataset = DescriptorGraphDataset(os.path.join(dataset_dir, "test"), mode="test", batch_size=32)

    # Collect test labels
    y_true = []
    for (X_batch, A_batch), Y_batch in test_dataset:
        y_true.extend(Y_batch)
    y_true = np.array(y_true)

    # Reinitialize test_dataset for prediction
    test_dataset = DescriptorGraphDataset(os.path.join(dataset_dir, "test"), mode="test", batch_size=32)
    preds = model.predict(test_dataset)

    # Convert logits to probabilities
    preds_prob = tf.sigmoid(preds).numpy().flatten()
    preds_label = (preds_prob > 0.5).astype(int)

    # ROC Curve & AUC
    fpr, tpr, thresholds = roc_curve(y_true, preds_prob)
    roc_auc = auc(fpr, tpr)
    print(f"Testing AUC: {roc_auc:.4f}")

    # Plot ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - Test Set')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()

    # Output submission
    ids = [str(i).zfill(4) for i in range(1, len(y_true) + 1)]
    import pandas as pd
    df = pd.DataFrame({"Id": ids, "Category": preds_label})
    df.to_csv("submission.csv", index=False)
    print("Prediction results saved to submission.csv")
    print("ROC curve saved as roc_curve.png")
