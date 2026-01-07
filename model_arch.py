from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, Add, Input, Lambda
from tensorflow.keras.models import Model
import tensorflow as tf
from keras.regularizers import l2
from tensorflow.keras.layers import (
    Flatten, 
    Dropout, 
    BatchNormalization, 
    Concatenate,           
    Input, 
    LayerNormalization, 
    Lambda, 
    MultiHeadAttention     
)

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

    # Apparently the other uncommented version is better, need to check claude (had chat with claude 2:00 AM on 12/31/25) about why this version is better
    '''def build_model_with_residuals_new(self, num_players_per_team, num_features, loss):
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
        
        return model'''
    
    def build_model_with_residuals_new(self, num_players_per_team, num_features, loss):
        """
        Optimized residual architecture with attention and better aggregation
        """
        input_layer = Input(shape=(2, num_players_per_team, num_features))
        
        # Initial projection with stronger normalization
        x = Dense(128, activation='silu', kernel_regularizer=l2(0.001))(input_layer)
        x = LayerNormalization()(x)
        x = Dropout(0.15)(x)  # Slightly higher dropout early
        
        # Deeper player-level feature extraction
        x = self.residual_block(x, 128, dropout_rate=0.1)
        x = self.residual_block(x, 128, dropout_rate=0.1)  # Keep dim for stable learning
        x = self.residual_block(x, 64, dropout_rate=0.1)
        
        # ADD: Multi-head attention to capture player interactions
        # This lets the model learn which players are most relevant
        attn_output = MultiHeadAttention(
            num_heads=4, 
            key_dim=64,
            dropout=0.1
        )(x, x)
        x = LayerNormalization()(x + attn_output)  # Residual connection
        
        x = self.residual_block(x, 64, dropout_rate=0.1)
        x = self.residual_block(x, 32, dropout_rate=0.1)
        
        # IMPROVED: Richer team aggregation (mean + max + min)
        team_mean = Lambda(lambda t: tf.reduce_mean(t, axis=2))(x)
        team_max = Lambda(lambda t: tf.reduce_max(t, axis=2))(x)
        team_std = Lambda(lambda t: tf.math.reduce_std(t, axis=2))(x)  # Capture spread
        
        team_embed = Concatenate(axis=-1)([team_mean, team_max, team_std])
        # shape: (batch, 2, 96) = 2 teams × (32 mean + 32 max + 32 std)
        
        # ADD: Cross-team attention (let teams "see" each other)
        team_embed = LayerNormalization()(team_embed)
        cross_attn = MultiHeadAttention(
            num_heads=2,
            key_dim=48,
            dropout=0.1
        )(team_embed, team_embed)
        team_embed = LayerNormalization()(team_embed + cross_attn)
        
        # Flatten to game-level representation
        game_embed = Flatten()(team_embed)
        # shape: (batch, 192)
        
        # Deeper final prediction network
        h = Dense(256, activation='silu', kernel_regularizer=l2(0.001))(game_embed)
        h = LayerNormalization()(h)
        h = Dropout(0.2)(h)
        
        h = self.residual_block(h, 256, dropout_rate=0.15)
        h = self.residual_block(h, 128, dropout_rate=0.15)
        h = self.residual_block(h, 64, dropout_rate=0.1)
        h = self.residual_block(h, 32, dropout_rate=0.1)
        
        # Output layer
        output = Dense(2, activation='linear')(h)
        
        model = Model(inputs=input_layer, outputs=output)
        
        # IMPROVED: Better optimizer configuration
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=0.001,
            weight_decay=0.01,
            clipnorm=1.0  # Gradient clipping for stability
        )
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['mae', 'mse']  # Track both metrics
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
    


    