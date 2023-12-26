import optuna
from train import main


def objective(trial):
    # Hyperparameters to be tuned
    hidden_dim = [
        trial.suggest_int("hidden_dim_1", 64, 128, step=32),
        trial.suggest_int("hidden_dim_2", 512, 2048, step=256),
        trial.suggest_int("hidden_dim_3", 128, 512, step=64)
    ]
    num_heads = [
        trial.suggest_categorical("num_heads_1", [2, 11]),
        trial.suggest_categorical("num_heads_2", [8, 16, 32, 64]),
        trial.suggest_categorical("num_heads_3", [2, 11])
    ]
    num_layers = [
        trial.suggest_int("num_layers_1", 1, 6),
        trial.suggest_int("num_layers_2", 1, 6)
    ]
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.0)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.0)
    downsample_method = trial.suggest_categorical("downsample_method", ['Linear', 'MaxPool'])
    mode = trial.suggest_categorical("mode", ['concat', 'separate'])
    fusion_layers = trial.suggest_int("fusion_layers", 2, 6)
    n_bottlenecks = trial.suggest_int("n_bottlenecks", 3, 7)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    max_seq_len = trial.suggest_int("max_seq_len", 4, 48, step=4)
    classification_head = trial.suggest_categorical("classification_head", [True])
    head_layer_sizes = [
        trial.suggest_int("Head_layer_1", 64, 512, step=32),
        trial.suggest_int("Head_layer_2", 64, 256, step=16),
        trial.suggest_int("Head_layer_3", 32, 128, step=16)
    ]

    # Calling the main function with the suggested hyperparameters
    _, _, _, _, best_val_acc = main(hidden_dim, num_heads, num_layers, learning_rate,
                                    dropout_rate, weight_decay, downsample_method, mode, fusion_layers,
                                    n_bottlenecks, batch_size, num_epochs=150, verbose=False, fold=1, device='cuda:1',
                                    save_model=False, max_seq_len=max_seq_len, classification_head=classification_head,
                                    plot=False)

    # Optuna aims to maximize the returned value
    return best_val_acc


# Running the optimization
study_name = "tuning_MultiModalityFusion_02"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(direction="maximize", study_name=study_name, storage = storage_name, load_if_exists=True)
study.optimize(objective, n_trials=100, n_jobs=1, show_progress_bar=True)  # Adjust the number of trials as needed

print("Best trial:")
trial = study.best_trial

print(" Value: ", trial.value)
print(" Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
