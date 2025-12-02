import tensorflow as tf
from tensorflow import keras

class LossFuncs:

    def __init__ (self):
        pass


    # ============================================================================
    # OPTION 1: Winner Penalty Loss
    # ============================================================================
    def winner_penalty_loss(self, alpha=0.5):
        """
        Custom loss that penalizes getting the winning team wrong.
        
        Args:
            alpha: Weight for winner penalty (default 0.5)
                Higher = care more about picking winner correctly
                Lower = care more about exact score accuracy
        
        Returns:
            Loss function
        """
        def loss(y_true, y_pred):
            # Standard MSE for score accuracy
            mse = tf.reduce_mean(tf.square(y_true - y_pred))
            
            # Calculate score differences (positive = team1 wins)
            true_diff = y_true[:, 0] - y_true[:, 1]
            pred_diff = y_pred[:, 0] - y_pred[:, 1]
            
            # Penalty when signs don't match (predicted wrong winner)
            # If true_diff * pred_diff < 0, they have opposite signs
            wrong_winner = tf.cast(true_diff * pred_diff < 0, tf.float32)
            winner_penalty = tf.reduce_mean(wrong_winner)
            
            return mse + alpha * winner_penalty
        
        return loss


    # ============================================================================
    # OPTION 2: Margin-Aware Loss
    # ============================================================================
    def margin_aware_loss(self, close_game_threshold=5.0, blowout_weight=0.5):
        """
        Weights errors differently for close games vs blowouts.
        Close games get more weight because they're more important to predict accurately.
        
        Args:
            close_game_threshold: Point difference for "close game" (default 5.0)
            blowout_weight: Weight for blowout errors (default 0.5)
                        Lower = care less about exact score in blowouts
        
        Returns:
            Loss function
        """
        def loss(y_true, y_pred):
            # Calculate errors
            errors = tf.square(y_true - y_pred)
            
            # Determine if game is close
            true_margin = tf.abs(y_true[:, 0] - y_true[:, 1])
            is_close = tf.cast(true_margin <= close_game_threshold, tf.float32)
            is_close = tf.expand_dims(is_close, axis=-1)  # Shape for broadcasting
            
            # Apply different weights
            weighted_errors = tf.where(
                is_close > 0.5,
                errors,  # Close games: full weight
                errors * blowout_weight  # Blowouts: reduced weight
            )
            
            return tf.reduce_mean(weighted_errors)
        
        return loss


    # ============================================================================
    # OPTION 3: Comprehensive Betting Loss
    # ============================================================================
    def betting_loss(self, mse_weight=1.0, moneyline_weight=0.3, spread_weight=0.2, total_weight=0.1):
        """
        Loss function optimized for betting predictions.
        Balances score accuracy with betting-relevant metrics.
        
        Args:
            mse_weight: Weight for score accuracy (default 1.0)
            moneyline_weight: Weight for picking winner (default 0.3)
            spread_weight: Weight for predicting point spread (default 0.2)
            total_weight: Weight for predicting total score (default 0.1)
        
        Returns:
            Loss function
        """
        def loss(y_true, y_pred):
            # 1. Score accuracy (MSE)
            mse = tf.reduce_mean(tf.square(y_true - y_pred))
            
            # 2. Moneyline (winner prediction)
            true_diff = y_true[:, 0] - y_true[:, 1]
            pred_diff = y_pred[:, 0] - y_pred[:, 1]
            wrong_winner = tf.cast(true_diff * pred_diff < 0, tf.float32)
            moneyline_error = tf.reduce_mean(wrong_winner)
            
            # 3. Spread accuracy
            spread_error = tf.reduce_mean(tf.square(true_diff - pred_diff))
            
            # 4. Total score accuracy (over/under)
            true_total = y_true[:, 0] + y_true[:, 1]
            pred_total = y_pred[:, 0] + y_pred[:, 1]
            total_error = tf.reduce_mean(tf.square(true_total - pred_total))
            
            # Combine all components
            return (mse_weight * mse + 
                    moneyline_weight * moneyline_error + 
                    spread_weight * spread_error + 
                    total_weight * total_error)
        
        return loss


    # ============================================================================
    # OPTION 4: Huber Loss with Winner Penalty (Robust to Outliers)
    # ============================================================================
    def huber_winner_loss(self, delta=1.0, winner_weight=0.5):
        """
        Huber loss (less sensitive to outliers) + winner penalty.
        Good if you have some games with unusual scores.
        
        Args:
            delta: Huber delta parameter (default 1.0)
            winner_weight: Weight for winner penalty (default 0.5)
        
        Returns:
            Loss function
        """
        def loss(y_true, y_pred):
            # Huber loss for robustness
            huber = tf.keras.losses.Huber(delta=delta)
            huber_loss = huber(y_true, y_pred)
            
            # Winner penalty
            true_diff = y_true[:, 0] - y_true[:, 1]
            pred_diff = y_pred[:, 0] - y_pred[:, 1]
            wrong_winner = tf.cast(true_diff * pred_diff < 0, tf.float32)
            winner_penalty = tf.reduce_mean(wrong_winner)
            
            return huber_loss + winner_weight * winner_penalty
        
        return loss