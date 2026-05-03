import numpy as np
import matplotlib.pyplot as plt
import os

def evaluate_all(results, y_test, scalers):
    # use any scaler to inverse transform back to real vehicle counts
    # all sites are normalised the same way so any scaler works as a reference
    any_scaler = list(scalers.values())[0]

    # metrics on normalised 0-1 scale
    print('\nNormalised Scale')
    for r in results:
        actual = y_test.flatten()
        preds = r['predictions'].flatten()
        # mask out near-zero values to avoid MAPE blowing up at quiet hours
        mask = actual > 0.05
        mape = np.mean(np.abs((actual[mask] - preds[mask]) / actual[mask])) * 100
        r['mape'] = mape
        print(f"{r['name']}: MAE={r['mae']:.4f}, RMSE={r['rmse']:.4f}, MAPE={mape:.2f}%")

    # inverse transform to real vehicle counts for meaningful comparison
    print('\nReal Vehicle Counts')
    for r in results:
        preds_real = any_scaler.inverse_transform(r['predictions'].reshape(-1, 1)).flatten()
        actual_real = any_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        mae_real = np.mean(np.abs(actual_real - preds_real))
        rmse_real = np.sqrt(np.mean((actual_real - preds_real) ** 2))
        mask = actual_real > 5
        mape_real = np.mean(np.abs((actual_real[mask] - preds_real[mask]) / actual_real[mask])) * 100
        print(f"{r['name']}: MAE={mae_real:.1f} veh/15min, RMSE={rmse_real:.1f} veh/15min, MAPE={mape_real:.2f}%")

    os.makedirs('plots', exist_ok=True)

    # loss curves per model
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    for ax, r in zip(axes, results):
        ax.plot(r['history'].history['loss'], label='train')
        ax.plot(r['history'].history['val_loss'], label='val')
        ax.set_title(f"{r['name']} loss")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE')
        ax.legend()
    plt.tight_layout()
    plt.savefig('plots/loss_curves.png', dpi=150)
    plt.show()

    # bar chart comparing MAE and RMSE across models
    names = [r['name'] for r in results]
    maes = [r['mae'] for r in results]
    rmses = [r['rmse'] for r in results]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.2, maes, 0.4, label='MAE')
    ax.bar(x + 0.2, rmses, 0.4, label='RMSE')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title('Model comparison: MAE and RMSE (normalised scale)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png', dpi=150)
    plt.show()

    # predictions vs actual for first 200 test steps
    fig, axes = plt.subplots(len(results), 1, figsize=(14, 4 * len(results)))
    for ax, r in zip(axes, results):
        ax.plot(y_test.flatten()[:200], label='actual')
        ax.plot(r['predictions'].flatten()[:200], label='predicted', alpha=0.8)
        ax.set_title(f"{r['name']} predictions vs actual (first 200 steps)")
        ax.legend()
    plt.tight_layout()
    plt.savefig('plots/predictions.png', dpi=150)
    plt.show()