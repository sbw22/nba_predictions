from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, Add, Input, Lambda
from tensorflow.keras.models import Model
import tensorflow as tf


class ResidualArch:

    def __init__ (self):
        pass

    def residual_block(self, x, units, dropout_rate=0.1, activation='silu'):
        """
        Creates a residual block with skip connection.
        
        Args:
            x: Input tensor
            units: Number of units in the dense layer
            dropout_rate: Dropout rate (default 0.1)
            activation: Activation function (default 'silu')
        
        Returns:
            Output tensor with residual connection applied
        """
        # Store the input for the skip connection
        residual = x
        
        # Main path
        out = Dense(units, activation=activation)(x)
        out = LayerNormalization()(out)
        out = Dropout(dropout_rate)(out)
        
        # If dimensions match, add skip connection directly
        if x.shape[-1] == units:
            out = Add()([out, residual])
        else:
            # If dimensions don't match, project residual to match output dims
            residual_projection = Dense(units, activation=None)(residual)
            out = Add()([out, residual_projection])
        
        return out


    def build_model_with_residuals_old(self, X):
        """
        Old model architecture (flattened input) with residual connections.
        For use when testing_player_stats = True
        """
        input_layer = Input(shape=(X.shape[1],))
        
        # First block - no residual since input might have different dims
        x = Dense(256, activation='silu')(input_layer)
        x = LayerNormalization()(x)
        x = Dropout(0.1)(x)
        
        # Residual blocks
        x = self.residual_block(x, 256, dropout_rate=0.1)
        x = self.residual_block(x, 128, dropout_rate=0.1)
        x = self.residual_block(x, 128, dropout_rate=0.1)  # Same dims = pure residual
        x = self.residual_block(x, 64, dropout_rate=0.1)
        x = self.residual_block(x, 32, dropout_rate=0.1)
        
        # Output layer
        output = Dense(2, activation='linear')(x)
        
        model = Model(inputs=input_layer, outputs=output)
        model.compile(
            optimizer='adamW',
            loss=loss,
            metrics=['mae']
        )
        
        return model


    def build_model_with_residuals_new(self, num_players_per_team, num_features, loss):
        """
        New model architecture (structured input) with residual connections.
        For use when testing_player_stats = False
        """
        # Input shape: (2 teams, num_players, num_features)
        input_layer = Input(shape=(2, num_players_per_team, num_features))
        
        # Per-player MLP with residual connections
        x = Dense(128, activation='silu')(input_layer)
        x = LayerNormalization()(x)
        
        # Residual blocks for player embeddings
        x = self.residual_block(x, 128, dropout_rate=0.1)
        x = self.residual_block(x, 64, dropout_rate=0.1)
        x = self.residual_block(x, 64, dropout_rate=0.1)  # Pure residual
        x = self.residual_block(x, 32, dropout_rate=0.1)
        
        # Aggregate players → team embeddings (mean pooling)
        team_embed = Lambda(lambda t: tf.reduce_mean(t, axis=2))(x)
        # shape is now (batch, 2, 32)
        
        # Flatten both team embeddings into one vector
        game_embed = Lambda(
            lambda t: tf.reshape(t, (-1, 64)),   # 2 teams × 32 features
            output_shape=(64,)
        )(team_embed)
        # shape (batch, 64)
        
        # Final prediction network with residuals
        h = Dense(128, activation='silu')(game_embed)
        h = LayerNormalization()(h)
        h = Dropout(0.1)(h)
        
        h = self.residual_block(h, 128, dropout_rate=0.1)  # Pure residual
        h = self.residual_block(h, 64, dropout_rate=0.1)
        h = self.residual_block(h, 32, dropout_rate=0.1)
        
        # Output layer
        output = Dense(2, activation='linear')(h)
        
        model = Model(inputs=input_layer, outputs=output)
        model.compile(
            optimizer='adamW',
            loss=loss,  # Use custom loss passed as argument, replaces 'mse'
            metrics=['mae']
        )
        
        return model


    # Example: How to update your main() function
    '''def example_usage_in_main(self):
        """
        Replace your model building section with this:
        """
        # ... your existing data preparation code ...
        
        # Build model with residual connections
        if testing_player_stats:
            # Use flattened input model
            model = self.build_model_with_residuals_old(X_train.shape[1])
        else:
            # Use structured input model
            model = self.build_model_with_residuals_new(num_players, num_features)
        
        print(f"Built model with residual connections: {model.summary()}")'''
        
        # Rest of your training code stays the same...


    # Bonus: Even deeper model with bottleneck residual blocks
    def bottleneck_residual_block(x, units, bottleneck_ratio=0.25, dropout_rate=0.1):
        """
        Bottleneck residual block (like ResNet-50).
        Reduces parameters by compressing through a bottleneck.
        
        Flow: x → compress → process → expand → add to x
        """
        residual = x
        bottleneck_units = int(units * bottleneck_ratio)
        
        # Compress
        out = Dense(bottleneck_units, activation='silu')(x)
        out = LayerNormalization()(out)
        
        # Process
        out = Dense(bottleneck_units, activation='silu')(out)
        out = LayerNormalization()(out)
        out = Dropout(dropout_rate)(out)
        
        # Expand back
        out = Dense(units, activation=None)(out)
        out = LayerNormalization()(out)
        
        # Skip connection
        if x.shape[-1] != units:
            residual = Dense(units, activation=None)(residual)
        
        out = Add()([out, residual])
        out = tf.keras.activations.silu(out)  # Activation after addition
        
        return out