import csv
import ast
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import pickle



class DataScaler:
    def __init__(self):
        pass

    def print_sample_data(self, train_game_data):
        return

    def print_data_shape(self, sample_data):

        numpy_data = np.array(sample_data)

        print(f"numpy_data shape: {numpy_data.shape}")

        return



    def import_game_data(self, file_path):

        total_stats = []

        with open(file_path, 'r') as file:

            reader = csv.reader(file)
            reader.__next__()  # Skip the header row
            for row in reader:
                new_row = [int(row[0]), ast.literal_eval(row[1]), str(row[2]), ast.literal_eval(row[3]), ast.literal_eval(row[4])]
                total_stats.append(new_row)
        
        return total_stats
    
    '''def import_team_data(self, file_path):
        # Might not need this function, because team data is included in game data (I think).

        total_stats = []

        with open(file_path, 'r') as file:

            reader = csv.reader(file)'''

    def import_score_data(self, file_path):
        
        total_stats = []

        df = pd.read_csv(file_path, header=None)  # <-- keep the first CSV row as data

        ext_row = df.iloc[0, :].to_list()

        ext_row = [float(score) for score in ext_row]

        score_train_data = df.iloc[2, :].values.tolist()

        # score_train_list = ast.literal_eval(score_train_data)[:10]  # Limit to the first 1000 games for now
        

        score_train_list = []
        counter = 0

        for game_scores in score_train_data:
            counter += 1
            '''if counter >= 1000: # This is just to limit the number of games for now. MAKE SURE THIS IS THE SAME NUMBER AS IN THE CORRESPONDING TRAIN_MODEL FUNCTIONS
                break'''
            try:
                score_train_list.append(ast.literal_eval(game_scores))
            except Exception as e:
                print(f"Error parsing game scores: {e}")
                break
        
        print(f"score_train_list: {score_train_list[:5]}, type: {type(score_train_list)}, length: {len(score_train_list)}, type of first item: {type(score_train_list[0])}")
        
        # print(f"ext_row: {ext_row}")
        print(f"ext_row: {ext_row[:5]}")

        return [ext_row, score_train_list]

        with open(file_path, 'r') as file:

            reader = csv.reader(file)
            reader.__next__()  # Skip the header row
            
            ext_row = reader[0]
            total_stats.append([int(score) for score in ext_row])
        
            '''for row in reader:
                new_row = [ast.literal_eval(row)]
                total_stats.append(new_row)'''
        
        return total_stats

    def import_stat_data(self, file_path):
        
        total_stats = []

        with open(file_path, 'r') as file:

            reader = csv.reader(file)
            reader.__next__()  # Skip the header row
            for row in reader:
                new_row = [ast.literal_eval(row[0])]
                total_stats.append(new_row)
        
        
        return total_stats



    def get_score_scaler(self, score_data):

        spread_score_data = []

        '''for score_pair in score_data:

            spread_score_data.extend(score_pair)'''

        # Process scores
        score_scaler = MinMaxScaler(feature_range=(0,1))

        numpy_scores = np.array(score_data).reshape(-1, 1)  # Converts all numbers in training set to numpy.

        # print(f"numpy_scores: {numpy_scores.shape}")
        processed_scores = score_scaler.fit_transform(numpy_scores)  # Fit the scaler to the data and transform it.

        return score_scaler


    def get_stat_scaler(self, stat_data):

        stat_scaler_list = [] 

        for stat_list in stat_data:

            stat_scaler = MinMaxScaler(feature_range=(0,1))
            numpy_stat_list = np.array(stat_list).reshape(-1, 1)  # Converts all numbers in training set to numpy.

            processed_stat_list = stat_scaler.fit_transform(numpy_stat_list)  # Fit the scaler to the data and transform it.

            # print(f"Processing stat list with length: {len(stat_list)}")

            stat_scaler_list.append(stat_scaler)

        return stat_scaler_list
    

    def scale_stat_data(self, stat_data, stat_scalers_list):

        scaled_game_stats_list = []
        counter = 0
        for game in stat_data:
            counter += 1
            '''if counter >= 1000:  # Only process the first 1000 games for now
                break'''
            if counter % 1000 == 0:
                print(f"Scaling game {counter}/{len(stat_data)}")
            team_stats = game[4] # Assuming game[4] contains the team stats
            scaled_game_stats = []
            for team in team_stats:
                scaled_team_stats = []
                for player_stats in team:
                    scaled_player_stats = []
                    for i, stat in enumerate(player_stats):
                        scaler = stat_scalers_list[i]
                        stat = scaler.transform([[stat]])   #[0][0]
                        scaled_player_stats.append(stat)
                    
                    scaled_team_stats.append(scaled_player_stats)
                
                scaled_game_stats.append(scaled_team_stats)
            
            scaled_game_stats_list.append(scaled_game_stats)
        
        print(f"Scaled all game stats.")


        return scaled_game_stats_list
    

    def scale_score_data(self, score_data, score_scaler):

        print(f"type score_data: {type(score_data)}")
        # print(f"score_data: {score_data[:5]}")

        scaled_score_data = []

        for game in score_data:
            score1, score2 = game[0], game[1]

            scaled_score1 = score_scaler.transform([[score1]])  # Scale the score using the fitted scaler
            scaled_score2 = score_scaler.transform([[score2]])  # Scale the score using the fitted scaler
            scaled_score_data.append([scaled_score1, scaled_score2])

        return scaled_score_data



def main():

    data_scaler = DataScaler()

    print(f"getting train_game_data...")
    train_game_data = data_scaler.import_game_data("csv_data_files/game_train_data.csv")

    # !!!!!! READ GET_PREDICT_DATA.PY NOTE BEFORE RUNNING THIS !!!!!!
    # Make sure to check the structure of the data and the scalers being used.

    print(f"getting test_game_data...")
    test_game_data = data_scaler.import_game_data("csv_data_files/game_test_data.csv")

    '''print(f"getting training_team_data...")
    train_team_data = data_scaler.import_team_data("csv_data_files/game_train_teams.csv")

    print(f"getting test_team_data...")
    test_team_data = data_scaler.import_team_data("csv_data_files/game_test_teams.csv")'''

    #########################################################################

    print(f"getting score_data...")
    score_import = data_scaler.import_score_data("csv_data_files/game_train_scores.csv")
    ext_score_data = score_import[0]
    score_train_list = score_import[1]

    print(f"out of import_score_data")
    print(f"score_train_list: {score_train_list[:5]}, type: {type(score_train_list)}, length: {len(score_train_list)}, type of first item: {type(score_train_list[0])}")

    print(f"getting full_stat_data...")
    full_stat_data = data_scaler.import_stat_data("csv_data_files/game_train_stats.csv")


    print(f"Imported all data.")


    data_scaler.print_data_shape(full_stat_data)

    score_scaler = data_scaler.get_score_scaler(ext_score_data)

    stat_scalers_list = data_scaler.get_stat_scaler(full_stat_data)

    scaled_train_game_data = data_scaler.scale_stat_data(train_game_data, stat_scalers_list)

    scaled_train_score_data = data_scaler.scale_score_data(score_train_list, score_scaler)

    print(f"scaled_train_score_data: {scaled_train_score_data[:5]}, type: {type(scaled_train_score_data)}, length: {len(scaled_train_score_data)}, type of first item: {type(scaled_train_score_data[0])}")

    print(f"len of scaled_train_score_data: {len(scaled_train_score_data)}")

    print(f"Saving scalers and scaled data...")

    with open('pkl_files/score_scaler.pkl', 'wb') as file:
        pickle.dump(score_scaler, file)
    
    with open('pkl_files/stat_scalers_list.pkl', 'wb') as file:
        pickle.dump(stat_scalers_list, file)

    with open('pkl_files/scaled_train_game_data.pkl', 'wb') as file:
        pickle.dump(scaled_train_game_data, file)

    with open('pkl_files/scaled_train_score_data.pkl', 'wb') as file:
        pickle.dump(scaled_train_score_data, file)

    # Need to fix how I am importing stat data and score data. # might be done with this.







    pass






if __name__ == "__main__":
    main()