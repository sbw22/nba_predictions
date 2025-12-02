import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


class EnsembleModels:
    
    def __init__ (self):
        pass

    # ============================================================================
    # ADD THESE ENSEMBLE FUNCTIONS TO YOUR MAIN FILE (after imports)
    # ============================================================================

    def train_ensemble_models(self, X_train, y_train, num_models, model_builder_func, 
                            model_params, epochs=500, validation_split=0.3):
        """
        Train multiple models with different random seeds.
        
        Args:
            X_train: Training data
            y_train: Training labels
            num_models: Number of models to train
            model_builder_func: Your model building function (from ResidualArch)
            model_params: Dict with 'num_players_per_team', 'num_features', 'loss'
            epochs: Training epochs
            validation_split: Validation split ratio
        
        Returns:
            List of trained models
        """
        models = []
        
        for i in range(num_models):
            print(f"\n{'='*60}")
            print(f"Training Ensemble Model {i+1}/{num_models}")
            print(f"{'='*60}\n")
            
            # Set different random seed for each model
            seed = 42 + i
            np.random.seed(seed)
            tf.random.set_seed(seed)
            
            # Build model with all required parameters
            model = model_builder_func(
                model_params['num_players_per_team'], 
                model_params['num_features'],
                model_params['loss']  # Pass the loss function
            )
            
            # Callbacks
            early_stop = EarlyStopping(
                monitor='val_loss', 
                patience=25, 
                restore_best_weights=True
            )
            checkpoint = ModelCheckpoint(
                f'model_and_scalers/ensemble_model_{i}.h5', 
                save_best_only=True
            )
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            )
            
            # Train
            model.fit(
                X_train, y_train, 
                epochs=epochs, 
                validation_split=validation_split, 
                batch_size=64, 
                callbacks=[early_stop, checkpoint, reduce_lr],
                verbose=1
            )
            
            models.append(model)
            print(f"\n✓ Model {i+1} training complete!")
        
        return models


    def predict_ensemble_average(self, models, X, verbose=True):
        """
        Make predictions using ensemble averaging.
        
        Args:
            models: List of trained models
            X: Input data
            verbose: Print progress
        
        Returns:
            Averaged predictions from all models
        """
        predictions = []
        
        for i, model in enumerate(models):
            if verbose:
                print(f"Getting predictions from model {i+1}/{len(models)}...")
            pred = model.predict(X, batch_size=1, verbose=0)
            predictions.append(pred)
        
        # Average all predictions
        ensemble_pred = np.mean(predictions, axis=0)
        
        if verbose:
            print(f"✓ Ensemble predictions complete (averaged {len(models)} models)")
        
        return ensemble_pred


    def save_ensemble(self, models, num_models):
        """Save ensemble models to disk."""
        for i, model in enumerate(models):
            model.save(f'model_and_scalers/ensemble_model_{i}.h5')
        
        print(f"\n✓ Saved {num_models} ensemble models to model_and_scalers/")


    def load_ensemble(self, num_models=5):
        """Load ensemble models from disk."""
        models = []
        
        for i in range(num_models):
            try:
                model = load_model(f'model_and_scalers/ensemble_model_{i}.h5')
                models.append(model)
                print(f"✓ Loaded ensemble model {i+1}/{num_models}")
            except:
                print(f"⚠️  Could not load model {i+1} - file may not exist")
        
        return models


    def train_and_evaluate_ensemble(self):
        # Train ensemble
        TRAIN_NEW_ENSEMBLE = True  # Set False to load saved models
        NUM_ENSEMBLE_MODELS = 5

        if TRAIN_NEW_ENSEMBLE:
            print(f"\nTraining ensemble of {NUM_ENSEMBLE_MODELS} models...")
            models = self.train_ensemble_models(
                X_train_structured, y_train,
                num_models=NUM_ENSEMBLE_MODELS,
                model_builder_func=residual_archs.build_model_with_residuals_new,
                model_params={
                    'num_players_per_team': num_players,
                    'num_features': num_features,
                    'loss': my_loss  # This is the key addition!
                },
                epochs=500,
                validation_split=0.3
            )
            self.save_ensemble(models, NUM_ENSEMBLE_MODELS)
        else:
            models = self.load_ensemble(num_models=NUM_ENSEMBLE_MODELS)

        # Make ensemble predictions
        print("\nMaking ensemble predictions on test data...")
        test_predictions = self.predict_ensemble_average(models, X_test_structured)

        scaled_up_test = score_scaler.inverse_transform(test_predictions)
        rounded_test = np.round(scaled_up_test).tolist()

        scaled_up_actual = score_scaler.inverse_transform(y_test.reshape(-1, 1)).tolist()
        scaled_up_actual = np.round(scaled_up_actual).tolist()
        scaled_up_actual = [[scaled_up_actual[i][0], scaled_up_actual[i+1][0]] 
                            for i in range(0, len(scaled_up_actual), 2)]

        find_trends(rounded_test, scaled_up_actual, team_test_names, year_test_list)

        print("\nMaking ensemble predictions on daily data...")
        daily_predictions = self.predict_ensemble_average(models, X_pred_structured)

        scaled_up_daily = score_scaler.inverse_transform(daily_predictions)
        rounded_daily = np.round(scaled_up_daily).tolist()
        find_betting_lines(rounded_daily, team_pred_names, year_pred_list)