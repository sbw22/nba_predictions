from keras.src.layers.core.dense import Dense
from scale_data import DataScaler
from arrange_data import FormatData
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Activation
from tensorflow.keras.layers import Flatten, Dropout, BatchNormalization, Concatenate, Input, LayerNormalization, Lambda
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
import tensorflow as tf
import keras
from keras.losses import Huber
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd

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
 

    return [scaled_game_pred_list, score_scaler, stat_scalers_list, scaled_train_game_data, scaled_train_score_data, team_train_names, team_test_names, team_pred_names, year_train_list, year_test_list, year_pred_list]


def build_old_model(X):

    model = Sequential([
        Input(shape=(X.shape[1],)),
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
    x = Dense(128, activation='silu')(input_layer)
    x = Dense(64, activation='silu')(x)
    x = Dense(32, activation='silu')(x)

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
        optimizer='adam',
        loss='mse',
        metrics=['mae']
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
        if counter >= 1000:
            break
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


def flatten_game_list_to_numpy_array(data, train_and_test_names, len_of_train_and_test, year_train_and_test):
    numpy_data = []
    counter = 0
    end_name_list = []
    end_year_list = []

    for game in data:
        counter += 1
        if counter % 1000 == 0:
            print(f"Converting game {counter}/{len(data)} to numpy array")
        if counter >= len_of_train_and_test+1: # Process the exact number of games in the combined train and test set
            break

        # Flatten all player stats from both teams into one 1D feature vector
        game_features = []
        for team in game:
            for player in team:
                game_features.extend(player)

        numpy_data.append(game_features)

        # Count number of scores to the same number as amount of games
        end_name_list.append(train_and_test_names[counter - 1])
        end_year_list.append(year_train_and_test[counter - 1])

    # Pad/truncate so all games have equal length
    # numpy_data = pad_sequences(numpy_data, dtype='float32', padding='post', truncating='post')
    #  I DONT THINK I WANT TO CONTINUE WITH THIS, AND INSTEAD FIND A WAY TO GET ALL GAMES TO HAVE THE SAME NUMBER OF PLAYERS INSTEAD OF PADDING/TRUNCATING

    return np.array(numpy_data), end_name_list, end_year_list


def train_model(X, y, model):

    early_stop = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
    checkpoint = ModelCheckpoint('model_and_scalers/best_model.h5', save_best_only=True)

    # Train the model
    model.fit(X, y, epochs=200, validation_split=0.2, batch_size=64, callbacks=[early_stop, checkpoint])

    return model


def run_model(model, X, y, score_scaler, team_names, year_list, pred=False):
    # Evaluate the model
    predictions = model.predict(X, batch_size=1, verbose=0)
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
        moneyline = "Team 1" if team1_score > team2_score else "Team 2"

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
    moneyline_error = abs(moneyline_guess - moneyline_actual)

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
        print(f"\nGame: {team1_name} vs {team2_name}, Year: {year}")
        print(f"Predicted: {team1_name}: {guesses[i][0]}, {team2_name}: {guesses[i][1]}")
        print(f"Actual: {team1_name}: {actuals[i][0]}, {team2_name}: {actuals[i][1]}")
        over_under_error, total_error, moneyline_error = find_betting_line_errors(guesses[i], actuals[i])
        total_over_under_error += over_under_error
        total_total_error += total_error
        total_moneyline_error += moneyline_error
        print(f"Over/Under Error: {over_under_error}")
        print(f"Total Error: {total_error}")
        print(f"Moneyline Error: {moneyline_error}")
    
    avg_over_under_error = total_over_under_error / len(guesses)
    avg_total_error = total_total_error / len(guesses)
    avg_moneyline_error = total_moneyline_error / len(guesses)
    print(f"\nAverage Over/Under Error: {avg_over_under_error}")
    print(f"Average Total Error: {avg_total_error}")
    print(f"Average Moneyline Error: {avg_moneyline_error}")
    print(f"Games predicted: {len(guesses)}")


def test_data_structure(X, y, team_names, year_list, score_scaler):
    # team_scores are the same for some reason?

    
    print(f"*******************************")
    '''print(f"X sample: {X[game_num]}")
    print(f"y sample: {y[game_num]}")'''
    for game_num in range(3):
        print(f"Game {game_num + 1}:")
        print(f"team_names sample: {team_names[game_num]}")
        if type(y) == type(np.array([])):
            team_scores = y[game_num]
            print(f"pre-scaled team_scores: {team_scores}")
            for team_score in team_scores:
                print(f"team_score: {team_score}")
                print(f"unscaled team_score: {score_scaler.inverse_transform([[team_score]])}")
            # team_scores = team_scores[game_num*2:game_num*2+2]  # Get the scores for the current game
            # print(f"team_scores sample: {team_scores[0]}, {team_scores[1]}")
        print(f"year_list sample: {year_list[game_num]}")
    return

def main():
    num_of_test_samples = 50      #########**************************************** Edit # of samples in test set here ****************************************#########
    
    data_scaler = DataScaler()
    data_formatter = FormatData()
    players_per_team = data_formatter.players_per_team
    testing_player_stats = data_formatter.testing_player_stats

    import_list = import_scaled_game_data()

    score_scaler = import_list[1]
    stat_scalers_list = import_list[2]

    scaled_game_pred_list = import_list[0]
    scaled_train_game_data = import_list[3]
    scaled_train_score_data = import_list[4]

    team_train_names = import_list[5]
    team_test_names = import_list[6]
    team_pred_names = import_list[7]

    year_train_list = import_list[8]
    year_test_list = import_list[9]
    year_pred_list = import_list[10]


    print(f"Imported scaled game prediction data and scalers.")

    X_pred_len = len(scaled_game_pred_list)

    X_train = scaled_train_game_data  
    X_pred = scaled_game_pred_list

    train_names = team_train_names
    test_names = team_test_names


    year_train = year_train_list
    year_test = year_test_list
    year_pred = year_pred_list

    # This section of code is to make sure test and train data both get padded to the same length, so they have the same dimensions.
    print(f"beginning X_train length: {len(X_train)}")
    print(f"beginning X_pred length: {len(X_pred)}")

    X_train_and_test = X_train + X_pred
    print(f"combined X_train and X_pred length: {len(X_train_and_test)}")
    train_and_test_names = train_names + test_names
    print(f"combined train and test names length: {len(train_and_test_names)}")
    year_train_and_test = year_train + year_test
    print(f"combined year train and test length: {len(year_train_and_test)}")
    len_of_train_and_test = len(X_train_and_test)
    
    '''for i, item in enumerate(X_pred):
        X_train.append(item)
        train_names.append(test_names[i])'''


    
    print(f"appended X_pred to X_train, new length: {len(X_train)}")

    X_train_and_test_flattened, train_and_test_names, year_train_and_test = flatten_game_list_to_numpy_array(X_train_and_test, train_and_test_names, len_of_train_and_test, year_train_and_test)
    
    print(f"new x_train_and_test_flattened type: {type(X_train_and_test_flattened)}")
    print(f"new x_train_and_test_flattened length: {len(X_train_and_test_flattened)}")

    X_train = X_train_and_test_flattened.copy()[:-X_pred_len]
    X_train = X_train[:-num_of_test_samples]
    X_test = X_train[-num_of_test_samples:] # Use the last 100 games of the training data as a test set (num_of_test_samples)
    X_pred = X_train_and_test_flattened[-X_pred_len:]
    
    print(f"X_train shape: {X_train.shape}")
    # I don't think these variables relate to X_train and X_pred. team_train_names relates to X_train, but team_test_names relates to the prediction, which I guess would be X_pred now that I think about it. 
    team_train_names = train_and_test_names.copy()[:-X_pred_len]
    team_train_names = team_train_names[:-num_of_test_samples]
    team_test_names = team_train_names[-num_of_test_samples:]
    team_pred_names = train_and_test_names[-X_pred_len:]

    year_train_list = year_train_and_test.copy()[:-X_pred_len]
    year_train_list = year_train_list[:-num_of_test_samples]
    year_test_list = year_train_and_test[-num_of_test_samples:]
    year_pred_list = year_train_and_test[-X_pred_len:]

    print(f"len of X_train = {len(X_train)}")
    print(f"len of X_test = {len(X_test)}")
    print(f"len of X_pred = {len(X_pred)}")
    # return

    print(f"scaled_train_score_data: {scaled_train_score_data[:5]}, type: {type(scaled_train_score_data)}, length: {len(scaled_train_score_data)}, type of first item: {type(scaled_train_score_data[0])}")
    y_train = np.array(scaled_train_score_data)
    print(f"y_train shape: {y_train.shape}")

    y_test = y_train[-num_of_test_samples:]

    print(f"y_test shape: {y_test.shape}")


    print(f"Converted game data to numpy array.")
    print(f"X shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"X_pred shape: {X_pred.shape}")
    # return


    print(f"Converted score data to numpy array.")
    print(f"y shape: {y_train.shape}")
    print(f"len of team_train_names: {len(team_train_names)}")
    print(f"len of team_test_names: {len(team_test_names)}")
    print(f"len of team_pred_names: {len(team_pred_names)}")
    

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    X_pred = X_pred.reshape(X_pred.shape[0], -1)
    y_train = y_train.reshape(y_train.shape[0], -1)
    print(f"y_train[0]: {y_train[0]}")
    print(f"y_train[1]: {y_train[1]}")
    print(f"y_train[2]: {y_train[2]}")
    print(f"y_train[3]: {y_train[3]}")
    y_test = y_test.reshape(y_test.shape[0], -1)

    # Your known values ------ THESE MIGHT CHANGE IF I CHANGE THE NUMBER OF PLAYERS PER TEAM IN ARRANGE_DATA
    num_teams = 2
    num_players = players_per_team

    # Compute number of stats per player
    num_features = X_train_and_test_flattened.shape[1] // (num_teams * num_players)
    print("Computed num_features per player =", num_features)


    # Reshape flattened → structured
    X_train_structured = structure_array(X_train, num_teams, num_players, num_features)
    X_test_structured = structure_array(X_test, num_teams, num_players, num_features)
    X_pred_structured = structure_array(X_pred, num_teams, num_players, num_features)
    

    print(f"Reshaped data for model input.")
    print(f"X_train shape: {X_train_structured.shape}")

    '''test_data_structure(X_train_structured, y_train, team_train_names, year_train_list, score_scaler)
    test_data_structure(X_test_structured, y_test, team_test_names, year_test_list, score_scaler)
    test_data_structure(X_pred_structured, None, team_pred_names, year_pred_list, score_scaler)'''

    '''print(f"y_test[0] scaled = {y_test[0]}")
    print(f"y_test[1] scaled = {y_test[1]}")
    print(f"y_test[2] scaled = {y_test[2]}")

    print(f"y_test[0] unscaled = {score_scaler.inverse_transform(y_test[0].reshape(-1, 1))}")
    print(f"y_test[1] unscaled = {score_scaler.inverse_transform(y_test[1].reshape(-1, 1))}")
    print(f"y_test[2] unscaled = {score_scaler.inverse_transform(y_test[2].reshape(-1, 1))}")'''
    # return

    # model = build_old_model(X_train)
    if not testing_player_stats:
        model = build_old_model(X_train_structured)
    else:
        model = build_new_model(num_players, num_features)

    print(f"Built model: {model.summary()}")

    print(f"Training model...")
    # model = train_model(X_train, y_train, model)
    model = train_model(X_train_structured, y_train, model)

    print(f"Training complete.")

    print(f"running predictions on test data...")
    test_predictions = run_model(model, X_test_structured, y_test, score_scaler, team_test_names, year_test_list)
    print(f"Test predictions complete.")
    print(f"running predictions on daily data...")
    # daily_predictions = run_model(model, X_pred_structured, y_test, score_scaler, team_pred_names, year_pred_list, pred=True)
    print(f"\nDaily predictions complete.")


    # Next Steps: 
    # Find average error, over/under error, total error, etc. for test set
    # 


if __name__ == "__main__":
    main()