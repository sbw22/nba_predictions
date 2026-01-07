from keras.src.layers.core.dense import Dense
from model_arch import ResidualArch
from loss_funcs import LossFuncs
from scale_data import DataScaler
from arrange_data import FormatData
from ensemble_models import EnsembleModels
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential, Model
from keras.layers import Activation
from tensorflow.keras.layers import Flatten, Dropout, BatchNormalization, Concatenate, Input, LayerNormalization, Lambda, MultiHeadAttention
from sklearn.model_selection import GridSearchCV
from keras.regularizers import l2
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
import tensorflow as tf
import keras
from keras.losses import Huber
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
import xgboost as xgb

def import_scaled_game_data():

    with open('pkl_files/scaled_game_pred_list.pkl', 'rb') as file:
        scaled_game_pred_list = pickle.load(file)
    
    with open('pkl_files/score_scaler.pkl', 'rb') as file:
        score_scaler = pickle.load(file)

    with open('pkl_files/stat_scalers_list.pkl', 'rb') as file:
        stat_scalers_list = pickle.load(file)
    
    with open('pkl_files/scaled_train_game_data.pkl', 'rb') as file:
        scaled_train_game_data = pickle.load(file)

    with open('pkl_files/scaled_train_score_data.pkl', 'rb') as file:
        scaled_train_score_data = pickle.load(file)
    
    with open('pkl_files/team_train_names.pkl', 'rb') as file:
        team_train_names = pickle.load(file)
    
    with open('pkl_files/team_test_names.pkl', 'rb') as file:
        team_test_names = pickle.load(file)

    with open('pkl_files/year_train_list.pkl', 'rb') as file:
        year_train_list = pickle.load(file)

    with open('pkl_files/year_test_list.pkl', 'rb') as file:
        year_test_list = pickle.load(file)

    with open('pkl_files/year_pred_list.pkl', 'rb') as file:
        year_pred_list = pickle.load(file)

    with open('pkl_files/team_pred_names.pkl', 'rb') as file:
        team_pred_names = pickle.load(file)
    
    with open('pkl_files/game_id_list.pkl', 'rb') as file:
        game_id_list = pickle.load(file)
 

    return [scaled_game_pred_list, score_scaler, stat_scalers_list, scaled_train_game_data, scaled_train_score_data, team_train_names, team_test_names, team_pred_names, year_train_list, year_test_list, year_pred_list, game_id_list]

def split_data(full_data, X_pred_len, num_of_test_samples):
    '''train = full_data.copy()[:-X_pred_len]
    train = train[:-num_of_test_samples]
    test = train[-num_of_test_samples:] # Use the last x games of the training data as a test set (num_of_test_samples)
    pred = full_data[-X_pred_len:]'''

    # Remove prediction data first
    train_and_test = full_data[:-X_pred_len] if X_pred_len > 0 else full_data.copy()
    
    # Now split train and test from what remains
    train = train_and_test[:-num_of_test_samples]
    test = train_and_test[-num_of_test_samples:]  # ✓ Takes from train_and_test, not train
    pred = full_data[-X_pred_len:] if X_pred_len > 0 else []

    return train, test, pred

def reshape_input(input_data):
    reshaped_data = []
    for data in input_data:
        data = data.reshape(data.shape[0], -1)
        reshaped_data.append(data)
    
    return reshaped_data

def build_old_model(X):

    model = Sequential([
        Input(shape=(X.shape[1]),),
        Dense(256, activation='silu'),
        LayerNormalization(),
        Dropout(0.1),

        Dense(128, activation='silu'),
        LayerNormalization(),
        Dropout(0.1),

        Dense(32, activation='silu'),
        LayerNormalization(),
        Dropout(0.1),

        
        Dense(2, activation='linear')
    ])

    # Compile the model
    model.compile(loss=Huber(delta=1.0), optimizer='AdamW', metrics=['mae'])

    return model

def build_new_model(num_players_per_team, num_features):
    # Input shape: (2 teams, 8 players, 132 features)
    input_layer = Input(shape=(2, num_players_per_team, num_features))

    # Per-player MLP
    x = Dense(128, activation='silu', kernel_regularizer=l2(0.01))(input_layer)
    x = Dense(64, activation='silu', kernel_regularizer=l2(0.01))(x)
    x = Dense(32, activation='silu')(x)

    x = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)

    # Aggregate players → team embeddings (mean pooling)
    team_embed = Lambda(lambda t: tf.reduce_mean(t, axis=2))(x)
    # shape is now (batch, 2, 32)

    # Flatten both team embeddings into one vector
    game_embed = Lambda(
        lambda t: tf.reshape(t, (-1, 64)),   # 2 teams × 32 features
        output_shape=(64,)
    )(team_embed)
    # shape (batch, 64)

    # Final prediction network
    h = Dense(128, activation='silu')(game_embed)
    h = Dense(64, activation='silu')(h)
    h = Dense(32, activation='silu')(h)

    output = Dense(2, activation='linear')(h)

    model = Model(inputs=input_layer, outputs=output)

    model.compile(
        optimizer='AdamW',
        loss='mse',
        metrics=['mse']
    )

    return model


def game_list_to_numpy_array(data):

    numpy_data = np.array([])
    # numpy_data = np.array([])
    counter = 0
    for game in data:
        counter += 1
        if counter % 1000 == 0:
            print(f"Converting game {counter}/{len(data)} to numpy array")
        '''if counter >= 1000:
            break'''
        numpy_game = np.array([])
        for team in game:
            numpy_team = np.array([])
            for player in team:
                numpy_player = np.array(player)
                numpy_team = np.append(numpy_team, numpy_player)
                # numpy_team = np.append(numpy_team, numpy_player)

            numpy_game = np.append(numpy_game, numpy_team)

        numpy_data = np.append(numpy_data, numpy_game)

    return np.array(numpy_data)

def structure_array(X, num_teams, num_players, num_features):
    X_structured = X.reshape(
        X.shape[0],
        num_teams,
        num_players,
        num_features
    )
    
    return X_structured


def flatten_game_list_to_numpy_array(game_data, all_names, len_of_train_and_test, all_years):
    numpy_data = []
    counter = 0
    end_name_list = []
    end_year_list = []

    for game in game_data:
        counter += 1
        if counter % 1000 == 0:
            print(f"Converting game {counter}/{len(game_data)} to numpy array")
        '''if counter >= len_of_train_and_test+1: # Process the exact number of games in the combined train and test set
            break'''

        # Flatten all player stats from both teams into one 1D feature vector
        game_features = []
        for team in game:
            for player in team:
                game_features.extend(player)

        numpy_data.append(game_features)

        # Count number of scores to the same number as amount of games
        try:
            end_name_list.append(all_names[counter - 1])
            end_year_list.append(all_years[counter - 1])
        except IndexError:
            pass


    return np.array(numpy_data), end_name_list, end_year_list


def train_model(X, y, model):

    early_stop = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
    checkpoint = ModelCheckpoint('model_and_scalers/best_model.h5', save_best_only=True)

    reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-6
    )

    # Train the model
    model.fit(X, y, epochs=500, validation_split=0.3, batch_size=64, callbacks=[early_stop, checkpoint, reduce_lr])

    return model


def run_model(model, X, y, score_scaler, team_names, year_list, pred=False):
    # Evaluate the model
    predictions = model.predict(X, batch_size=64, verbose=0)
    # loss, accuracy = model.evaluate(X, y)

    scaled_up_guesses = score_scaler.inverse_transform(predictions)
    rounded_guesses = np.round(scaled_up_guesses).tolist()  # Convert to list for easier manipulation

    # WARNING: DO NOT USE SCALE_UP_ACTUAL_SCORES FOR DAILY PREDICTIONS
    scaled_up_actual_scores = score_scaler.inverse_transform(y.reshape(-1, 1)).tolist()  # Convert to list for easier manipulation
    scaled_up_actual_scores = np.round(scaled_up_actual_scores).tolist()
    scaled_up_actual_scores = [[scaled_up_actual_scores[i][0], scaled_up_actual_scores[i+1][0]] for i in range(0, len(scaled_up_actual_scores), 2)]  # Group actual scores into pairs for each game

    if not pred:
        find_trends(rounded_guesses, scaled_up_actual_scores, team_names, year_list)
    else:
        find_betting_lines(rounded_guesses, team_names, year_list)
    
    return predictions


def find_betting_lines(guesses, team_names, year_list):
    print("\nBetting Lines:")
    for i in range(len(guesses)):
        team1_name, team2_name = team_names[i]
        year = year_list[i]
        team1_score, team2_score = guesses[i]
        team1_over_under = team2_score - team1_score
        team2_over_under = team1_score - team2_score
        total_score = team1_score + team2_score
        moneyline = f"{team1_name}" if team1_score > team2_score else f"{team2_name}"

        print(f"\nGame: {team1_name} vs {team2_name}, Year: {year}")
        print(f"Predicted Score:")
        print(f"{team1_name}: {guesses[i][0]}, {team2_name}: {guesses[i][1]}")
        print(f"Over/Under:")
        print(f"{team1_name} Over/Under: {team1_over_under}")
        print(f"{team2_name} Over/Under: {team2_over_under}")
        print(f"Total Score: {total_score}")
        print(f"Moneyline: {moneyline}")
        print(f"-----------------------------")
        


def avg_err(guesses, actuals):
    total_error = 0
    for i in range(len(guesses)):
        error = abs(guesses[i][0] - actuals[i][0]) + abs(guesses[i][1] - actuals[i][1])
        total_error += error

    avg_error = total_error / (len(guesses) * 2)  # Divide by total number of scores (2 per game)
    print(f"Average error: {avg_error}")


def find_betting_line_errors(guess, actual):
    team1_guess, team2_guess = guess
    team1_actual, team2_actual = actual

    team1_over_under_guess = team2_guess - team1_guess
    team1_over_under_actual = team2_actual - team1_actual
    over_under_error = abs(team1_over_under_guess - team1_over_under_actual)

    total_guess = team1_guess + team2_guess
    total_actual = team1_actual + team2_actual
    total_error = abs(total_guess - total_actual)

    moneyline_guess = 1 if team1_guess > team2_guess else 0
    moneyline_actual = 1 if team1_actual > team2_actual else 0
    moneyline_error = abs(moneyline_guess - moneyline_actual) # 0 if correct, 1 if incorrect

    return over_under_error, total_error, moneyline_error



def find_statistics(guesses, actuals, team_names, year_list):
    print("\nStatistics:")
    for i in range(len(guesses)):
        team1_name, team2_name = team_names[i]
        year = year_list[i]
        print(f"\nGame: {team1_name} vs {team2_name}, Year: {year}")
        print(f"Predicted: {team1_name}: {guesses[i][0]}, {team2_name}: {guesses[i][1]}")
        print(f"Actual: {team1_name}: {actuals[i][0]}, {team2_name}: {actuals[i][1]}")


def find_trends(guesses, actuals, team_names, year_list):
    print("\nTrends:")
    avg_error = avg_err(guesses, actuals)

    total_over_under_error = 0
    total_total_error = 0
    total_moneyline_error = 0

    for i, game in enumerate(guesses):

    
        
        team1_name, team2_name = team_names[i]
        year = year_list[i]
        over_under_error, total_error, moneyline_error = find_betting_line_errors(guesses[i], actuals[i])
        total_over_under_error += over_under_error
        total_total_error += total_error
        total_moneyline_error += moneyline_error

        if i < 10:  # Print first 10 games in detail
            print(f"\nGame: {team1_name} vs {team2_name}, Year: {year}")
            print(f"Predicted: {team1_name}: {guesses[i][0]}, {team2_name}: {guesses[i][1]}")
            print(f"Actual: {team1_name}: {actuals[i][0]}, {team2_name}: {actuals[i][1]}")
            print(f"Over/Under Error: {over_under_error}")
            print(f"Total Error: {total_error}")
            print(f"Moneyline Error: {moneyline_error}")
    
    avg_over_under_error = total_over_under_error / len(guesses)
    avg_total_error = total_total_error / len(guesses)
    avg_moneyline_error = total_moneyline_error / len(guesses)
    print(f"\nAverage Over/Under Error: {avg_over_under_error}")
    print(f"Average Total Error: {avg_total_error}")
    print(f"Average Moneyline Error: {avg_moneyline_error*100}% wrong, {(1-avg_moneyline_error)*100}% correct")
    print(f"Games predicted: {len(guesses)}")


def test_data_structure(X_structured, y, team_names, year_list, score_scaler, X, game_ids=None):
    # team_scores are the same for some reason?

    
    print(f"*******************************")
    '''print(f"y sample: {y[game_num]}")'''
    for game_num in range(3):
        # print(f"X_structured sample: {X_structured[game_num][0]}")
        print(f"Game {game_num + 1}:")
        print(f"team_names sample: {team_names[game_num]}")
        if type(y) == type(np.array([])):
            team_scores = y[game_num]
            print(f"pre-scaled team_scores: {team_scores}")
            for team_score in team_scores:
                print(f"team_score: {team_score}")
                print(f"unscaled team_score: {score_scaler.inverse_transform([[team_score]])}")
            print(f"game_id: {game_ids[game_num]}")
            print(f"scaled X sample: {X[game_num][0]}")
            # team_scores = team_scores[game_num*2:game_num*2+2]  # Get the scores for the current game
            # print(f"team_scores sample: {team_scores[0]}, {team_scores[1]}")
        print(f"year_list sample: {year_list[game_num]}")
    return


def verify_alignment(X_structured, y, team_names, year_list, score_scaler):
    """Check if X games match their y scores"""
    print("\n" + "="*60)
    print("DATA ALIGNMENT VERIFICATION")
    print("="*60)
    
    for i in range(min(5, len(y))):  # Check first 5 games
        print(f"\n--- Game {i+1} ---")
        print(f"Teams: {team_names[i]}")
        print(f"Year: {year_list[i]}")
        
        # Unscale and show scores
        unscaled_scores = score_scaler.inverse_transform(y[i].reshape(1, -1))[0]
        print(f"Scores (scaled): {y[i]}")
        print(f"Scores (actual): {unscaled_scores[0]:.1f} - {unscaled_scores[1]:.1f}")
        
        # Check team stats make sense (look at a few player features)
        team1_first_player = X_structured[i, 0, 0, :5]  # First 5 features of first player (team stats if using team stats instead of player stats)
        team2_first_player = X_structured[i, 1, 0, :5] 
        print(f"Team1 player1 sample features: {team1_first_player}")
        print(f"Team2 player1 sample features: {team2_first_player}")
        
        # Sanity check: if all features are identical, something's wrong
        if np.allclose(team1_first_player, team2_first_player):
            print("⚠️  WARNING: Both teams have identical features!")



def xgboost_model(X_train_structured, y_train, X_test_structured, y_test, X_pred_structured, score_scaler, team_test_names, year_test_list, team_pred_names, year_pred_list):

    # Need to learn how to implement GridSearchCV with XGBoost properly
    param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [300, 500, 700],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 0.9, 1.0]
    }

    grid_search = GridSearchCV(
        xgb.XGBRegressor(random_state=42),
        param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    # ... [all your existing data loading and preprocessing code] ...
    
    # After you have X_train_structured, y_train, etc.
    
    # Flatten the structured data for XGBoost (it expects 2D input)
    X_train_flat = X_train_structured.reshape(X_train_structured.shape[0], -1)
    X_test_flat = X_test_structured.reshape(X_test_structured.shape[0], -1)
    X_pred_flat = X_pred_structured.reshape(X_pred_structured.shape[0], -1)
    
    print(f"Flattened X_train shape: {X_train_flat.shape}")
    
    # Build XGBoost models (one for each team's score)
    print("Training XGBoost models...")
    
    # Model for Team 1 scores
    model_team1 = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective='reg:squarederror',
        random_state=42,
        early_stopping_rounds=25
    )
    
    # Model for Team 2 scores
    model_team2 = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective='reg:squarederror',
        random_state=42,
        early_stopping_rounds=25
    )
    
    # Split into train/validation
    from sklearn.model_selection import train_test_split
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_flat, y_train, test_size=0.3, random_state=42
    )
    
    # Train Team 1 model
    print("Training Team 1 score model...")
    model_team1.fit(
        X_train_split, y_train_split[:, 0],
        eval_set=[(X_val_split, y_val_split[:, 0])],
        verbose=50
    )
    
    # Train Team 2 model
    print("Training Team 2 score model...")
    model_team2.fit(
        X_train_split, y_train_split[:, 1],
        eval_set=[(X_val_split, y_val_split[:, 1])],
        verbose=50
    )
    
    print("Training complete.")
    
    # Make predictions
    print("Running predictions on test data...")
    test_pred_team1 = model_team1.predict(X_test_flat)
    test_pred_team2 = model_team2.predict(X_test_flat)
    test_predictions = np.column_stack([test_pred_team1, test_pred_team2])
    
    # Unscale and evaluate
    scaled_up_test = score_scaler.inverse_transform(test_predictions)
    rounded_test = np.round(scaled_up_test).tolist()
    
    scaled_up_actual = score_scaler.inverse_transform(y_test)
    rounded_actual = np.round(scaled_up_actual).tolist()
    
    find_trends(rounded_test, rounded_actual, team_test_names, year_test_list)
    
    # Daily predictions
    print("Running predictions on daily data...")
    daily_pred_team1 = model_team1.predict(X_pred_flat)
    daily_pred_team2 = model_team2.predict(X_pred_flat)
    daily_predictions = np.column_stack([daily_pred_team1, daily_pred_team2])
    
    scaled_up_daily = score_scaler.inverse_transform(daily_predictions)
    rounded_daily = np.round(scaled_up_daily).tolist()
    
    # find_betting_lines(rounded_daily, team_pred_names, year_pred_list)

    return


'''def enter_unavailable_players():
    # Function to enter players that are not playing
    absent_players = []  # Example: ['Player A', 'Player B']

    return absent_players'''


def main():

    # Would love to add a class or something of lists that contain all players that are out. I can change the list each day before running the predictions.


    data_scaler = DataScaler()
    data_formatter = FormatData()
    players_per_team = data_formatter.players_per_team
    testing_player_stats = data_formatter.testing_player_stats

    num_of_test_samples = 200      #########**************************************** Edit # of samples in test set here ****************************************#########

    # Your known values ------ THESE MIGHT CHANGE IF I CHANGE THE NUMBER OF PLAYERS PER TEAM IN ARRANGE_DATA - Edit: Might not have to worry about this anymore
    num_teams = 2
    num_players = players_per_team


    import_list = import_scaled_game_data()

    score_scaler = import_list[1]
    stat_scalers_list = import_list[2]

    X_train_original = import_list[3]
    X_pred = import_list[0]
    X_pred_len = len(X_pred)
    scaled_train_score_data = import_list[4]

    train_names = import_list[5]
    # test_names = import_list[6]
    pred_names = import_list[7]

    train_years = import_list[8]
    # test_years = import_list[9]
    year_pred = import_list[10]

    game_ids = import_list[11]

    '''print(f"pred_names: {pred_names}")
    return'''

    print(f"Imported scaled game prediction data and scalers.") # ##############################################################################################################


    # This section of code is to make sure test and train data both get padded to the same length, so they have the same dimensions.
    print(f"beginning X_train length: {len(X_train_original)}")
    print(f"beginning X_pred length: {len(X_pred)}")

    X_all = X_train_original + X_pred
    all_names = train_names + pred_names # + test_names
    all_years = train_years + year_pred # + test_years
    # len_of_train_and_test = len(X_all)
    print(f"combined X_train and X_pred length: {len(X_all)}")
    print(f"combined train and test names length: {len(all_names)}")
    print(f"combined year train and test length: {len(all_years)}")


    X_all_flattened, all_names, all_years = flatten_game_list_to_numpy_array(X_all, all_names, len(X_all), all_years)

    # Prepare y data - BEFORE any splitting
    y_all = np.array(scaled_train_score_data).reshape(-1, 2)
    print(f"\ny_all shape: {y_all.shape}")
    print(f"original X_train length: {len(X_train_original)}")

    
    # Verify length matches ORIGINAL X_train (before any splitting)
    if len(y_all) != len(X_train_original):
        print(f"⚠️ WARNING: y_all has {len(y_all)} games but original X_train has {len(X_train_original)} games")
        if len(y_all) > len(X_train_original):
            print(f"   Trimming y_all to match X_train_original")
            y_all = y_all[:len(X_train_original)]
        else:
            print(f"   ERROR: y_all is shorter than X_train_original!")
            return

    # Add dummy y for prediction data
    y_pred_dummy = np.zeros((X_pred_len, 2))
    y_all_extended = np.vstack([y_all, y_pred_dummy])

    print(f"y_all_extended shape after adding dummy pred data: {y_all_extended.shape}")
    print(f"Should match X_all_flattened length: {X_all_flattened.shape[0]}")

    # Verify they match before splitting
    if len(y_all_extended) != len(X_all_flattened):
        print(f"❌ CRITICAL ERROR: y_all_extended ({len(y_all_extended)}) != X_all_flattened ({len(X_all_flattened)})")
    # return
    
    '''y_pred_dummy = np.zeros((len(X_pred), 2))  # Dummy y for prediction data
    y_all_extended = np.vstack((y_all, y_pred_dummy))  # Combine actual scores with dummy scores for prediction data'''


    # NOW split using the same function as X
    y_train, y_test, y_pred = split_data(y_all_extended, X_pred_len, num_of_test_samples)
    X_train, X_test, X_pred = split_data(X_all_flattened, X_pred_len, num_of_test_samples)
    # \/ I don't think these variables relate to X_train and X_pred. team_train_names relates to X_train, but team_test_names relates to the prediction, which I guess would be X_pred now that I think about it. 
    team_train_names, team_test_names, team_pred_names = split_data(all_names, X_pred_len, num_of_test_samples)
    year_train_list, year_test_list, year_pred_list = split_data(all_years, X_pred_len, num_of_test_samples)
    # These lines starting from y_all ensure that y_train and y_test are properly aligned with X_train and X_test, and that y_pred is just a placeholder for the prediction data.



    print(f"\n--- SPLIT VERIFICATION ---")
    print(f"Train: X={len(X_train)}, y={len(y_train)}, names={len(team_train_names)}")
    print(f"Test:  X={len(X_test)}, y={len(y_test)}, names={len(team_test_names)}")
    print(f"Pred:  X={len(X_pred)}, names={len(team_pred_names)}")

    assert len(X_train) == len(y_train) == len(team_train_names), "Train mismatch!"
    assert len(X_test) == len(y_test) == len(team_test_names), "Test mismatch!"
    print("✓ All splits match!")
    


    '''game_ids_train = game_ids.copy()[:-num_of_test_samples]
    game_ids_test = game_ids[-num_of_test_samples:]'''
    print(f"type X_train before reshape: {type(X_train)}")
    print(f"type X_test before reshape: {type(X_test)}")
    print(f"type X_pred before reshape: {type(X_pred)}")
    X_train, X_test, X_pred = reshape_input([X_train, X_test, X_pred])

    '''y_train = y_train.reshape(-1, 2)
    y_test = y_test.reshape(-1, 2)''' # This is already done when we are preparing y_all, might come back to this if we need it

    # Compute number of stats per player
    num_features = X_all_flattened.shape[1] // (num_teams * num_players)
    print("Computed num_features per player =", num_features)

    # Reshape flattened → structured
    X_train_structured = structure_array(X_train, num_teams, num_players, num_features)
    X_test_structured = structure_array(X_test, num_teams, num_players, num_features)
    X_pred_structured = structure_array(X_pred, num_teams, num_players, num_features)
    

    print(f"Reshaped data for model input.")
    # maybe check out if X_train team stats are being structured correctly above in structure_array(). (make sure 60 -> 2 30s correctly)
    print(f"X_train shape: {X_train_structured.shape}")
    # return
    # test_data_structure(X_train_structured, y_train, team_train_names, year_train_list, score_scaler)
    # test_data_structure(X_test_structured, y_test, team_test_names, year_test_list, score_scaler, X_test, game_ids_test,)
    # test_data_structure(X_pred_structured, None, team_pred_names, year_pred_list, score_scaler, X_pred, None)

    '''print(f"y_test[0] scaled = {y_test[0]}")
    print(f"y_test[1] scaled = {y_test[1]}")
    print(f"y_test[2] scaled = {y_test[2]}")

    print(f"y_test[0] unscaled = {score_scaler.inverse_transform(y_test[0].reshape(-1, 1))}")
    print(f"y_test[1] unscaled = {score_scaler.inverse_transform(y_test[1].reshape(-1, 1))}")
    print(f"y_test[2] unscaled = {score_scaler.inverse_transform(y_test[2].reshape(-1, 1))}")'''

    verify_alignment(X_train_structured, y_train, team_train_names, year_train_list, score_scaler)
    verify_alignment(X_test_structured, y_test, team_test_names, year_test_list, score_scaler)


    if False:  # Switch to False to use neural network instead
        xgboost_model(X_train_structured, y_train, X_test_structured, y_test, X_pred_structured, score_scaler, team_test_names, year_test_list, team_pred_names, year_pred_list)

        return
    
    # ------------------------------------------------------ CHANGE MODEL BUILDING HERE ---------------------------------------------------------
    loss_funcs = LossFuncs()
    residual_archs = ResidualArch()
    ensemble_models = EnsembleModels()
    loss = loss_funcs.betting_loss(mse_weight=1.0, moneyline_weight=0.3, spread_weight=0.2, total_weight=0.2)
    # model = build_old_model(X_train)
    if True:  # testing_player_stats:
        # model = build_old_model(X_train)
        # model = build_new_model(num_players, num_features)
        # model = residual_archs.build_model_with_residuals_old(X_train) # changed X_train.shape[1] to X_train, and vice versa in functin
        # model = residual_archs.bottleneck_residual_block(X_train, num_features) # Pretty sure num_features is correct here, idk
        model = residual_archs.build_model_with_residuals_new(num_players, num_features, loss)
        
    else:
        model = residual_archs.build_model_with_residuals_new(num_players, num_features, loss)

    # return




    # """"""""""""""""""""""""""""""""""""""""""""'""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""




    # Train ensemble
    '''TRAIN_NEW_ENSEMBLE = True  # Set False to load saved models
    NUM_ENSEMBLE_MODELS = 5

    if TRAIN_NEW_ENSEMBLE:
        print(f"\nTraining ensemble of {NUM_ENSEMBLE_MODELS} models...")
        models = ensemble_models.train_ensemble_models(
            X_train_structured, y_train,
            num_models=NUM_ENSEMBLE_MODELS,
            model_builder_func=residual_archs.build_model_with_residuals_new,
            model_params={
                'num_players_per_team': num_players,
                'num_features': num_features,
                'loss': loss  # This is the key addition!
            },
            epochs=500,
            validation_split=0.3
        )
        ensemble_models.save_ensemble(models, NUM_ENSEMBLE_MODELS)
    else:
        models = ensemble_models.load_ensemble(num_models=NUM_ENSEMBLE_MODELS)

    # Make ensemble predictions
    print("\nMaking ensemble predictions on test data...")
    test_predictions = ensemble_models.predict_ensemble_average(models, X_test_structured)
    scaled_up_test = score_scaler.inverse_transform(test_predictions)
    rounded_test = np.round(scaled_up_test).tolist()

    scaled_up_actual = score_scaler.inverse_transform(y_test.reshape(-1, 1)).tolist()
    scaled_up_actual = np.round(scaled_up_actual).tolist()
    scaled_up_actual = [[scaled_up_actual[i][0], scaled_up_actual[i+1][0]] 
                        for i in range(0, len(scaled_up_actual), 2)]

    find_trends(rounded_test, scaled_up_actual, team_test_names, year_test_list)

    print("\nMaking ensemble predictions on daily data...")
    daily_predictions = ensemble_models.predict_ensemble_average(models, X_pred_structured)

    scaled_up_daily = score_scaler.inverse_transform(daily_predictions)
    rounded_daily = np.round(scaled_up_daily).tolist()
    find_betting_lines(rounded_daily, team_pred_names, year_pred_list)'''







    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""





    print(f"Built model: {model.summary()}")

    print(f"Training model...")
    # model = train_model(X_train, y_train, model)
    model = train_model(X_train_structured, y_train, model)

    print(f"Training complete.")

    print(f"running predictions on test data...")
    test_predictions = run_model(model, X_test_structured, y_test, score_scaler, team_test_names, year_test_list)
    print(f"Test predictions complete.")
    print(f"running predictions on daily data...")
    daily_predictions = run_model(model, X_pred_structured, y_pred_dummy, score_scaler, team_pred_names, year_pred_list, pred=True)
    print(f"\nDaily predictions complete.")


    # What we know:
    # 1) For testing (at least), the team names and team scores are mixed up. They are 50 spaces apart (I think) in the csv, so I just have to figure out why that is. 
        # Might have figured this out, but need to see why some values are being grabbed at 1123 instad of 1180 (or something like that). 
    # 2) The prediction data is showing that it is grabbing the first games of the season, instead of daily games. 

    # Lucky Number (error) = 5

if __name__ == "__main__":
    main()