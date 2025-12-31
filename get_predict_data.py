from collect_data import CollectGameAndPlayerData
from arrange_data import FormatData
from nba_api.stats.endpoints import scoreboardv2
import pickle
import csv
import ast
import numpy as np
import pandas as pd
import datetime


def import_train_game_data(file_path = "csv_data_files/nba_player_averages_2000-2024.csv"):

    data_formatter = FormatData()
    data_dict_list = data_formatter.get_data()

    train_game_dict_list = data_dict_list[2]
    # !!!!!!! ^^^^^^^^^^^^^
    print(f"train_game_dict_list: {train_game_dict_list[0].keys()}")

    return train_game_dict_list


    total_stats = []

    with open(file_path, 'r') as file:

        reader = csv.reader(file)
        reader.__next__()  # Skip the header row
        for row in reader:
            new_row = [int(row[0]), ast.literal_eval(row[1]), str(row[2]), ast.literal_eval(row[3]), ast.literal_eval(row[4])]
            total_stats.append(new_row)

    return total_stats


def get_scoreboard_data():
    current_date = datetime.datetime.now().date()
    next_day = current_date + datetime.timedelta(days=1)
    current_date = current_date.strftime("%m/%d/%Y")
    next_day = next_day .strftime("%m/%d/%Y")

    # current_date = next_day   #######################################################################################################################
    

    board = scoreboardv2.ScoreboardV2(game_date=current_date)

    print(f"board: {board}")

    df = board.get_data_frames()[0]

    print(f"df sample:")
    print(df.sample())
    for item in df.columns:
        print(f"item: {item}")

    for item in df['GAME_ID']:
        print(f"game_id: {item}")
    
    game_ids = df['GAME_ID'].tolist()
    team1_ids = df['HOME_TEAM_ID'].tolist()
    team2_ids = df['VISITOR_TEAM_ID'].tolist()
    game_year = df['SEASON'].tolist()
    

    scoreboard_list = [game_ids, team1_ids, team2_ids, game_year]
    return scoreboard_list, df



def find_team_names(team1_id, team2_id, train_game_dict_list):

    # print(f"train_game_dict_list: {train_game_dict_list}")
    team1_name = ""
    team2_name = ""


    for game in train_game_dict_list:

        if team1_name != "" and team2_name != "":
            break
        
        if game['TEAM_ID'] == team1_id:
            team1_name = game['TEAM_ABBREVIATION']
        if game['TEAM_ID'] == team2_id:
            team2_name = game['TEAM_ABBREVIATION']        
        

    return team1_name, team2_name




def getting_future_game_data(data_formatter, train_game_dict_list):

    # player_dict_list, game_dict_list, 
    # player_stat_names

    percentage_stat_names = data_formatter.percentage_player_stat_names
    average_stat_names = data_formatter.average_player_stat_names
    ranked_stat_names = data_formatter.ranked_player_stat_names

    player_stat_names = [percentage_stat_names, average_stat_names, ranked_stat_names]
    ext_player_stat_names = percentage_stat_names + average_stat_names + ranked_stat_names

    data_dict_list = data_formatter.get_data()

    train_player_dict_list = data_dict_list[0]
    test_player_dict_list = data_dict_list[1]   # Contains the player stats for players in the current season (2025 right now (maybe), I think you have to change this manually)
    train_game_dict_list = data_dict_list[2]
    test_game_dict_list = data_dict_list[3]
    train_team_dict_list = data_dict_list[4]
    test_team_dict_list = data_dict_list[5]
    boxscore_dict_list = data_dict_list[6]




    print(f"Collecting player data...")

    game_pred_list = []


    scoreboard_data_id_lists, test_players_df = get_scoreboard_data()

    test_players_df.to_csv('csv_data_files/daily_scoreboards.csv', index=False)

    print(f"scoreboard_data_lists: {scoreboard_data_id_lists}")

    # FIGURE OUT

    print(f"len of scoreboard_data_id_lists: {len(scoreboard_data_id_lists)}")
    print(f"len of scoreboard_data_id_lists[0]: {len(scoreboard_data_id_lists[0])}")
    print(f"len of scoreboard_data_id_lists[1]: {len(scoreboard_data_id_lists[1])}")
    print(f"len of scoreboard_data_id_lists[2]: {len(scoreboard_data_id_lists[2])}")
    print(f"len of scoreboard_data_id_lists[3]: {len(scoreboard_data_id_lists[3])}")
    #return

    for i in range(len(scoreboard_data_id_lists[0])):
        print(f"\nGetting data for game {i+1}/{len(scoreboard_data_id_lists[0])}")


        game_id = scoreboard_data_id_lists[0][i]
        team1_id = scoreboard_data_id_lists[1][i]
        team2_id = scoreboard_data_id_lists[2][i]
        game_year = scoreboard_data_id_lists[3][i]
        game_year = f"{game_year}-{int(game_year)+1-2000}"


        team1_name, team2_name = find_team_names(team1_id, team2_id, train_game_dict_list)

        # print(f"game_year_str: {game_year}")

        team_names = [team1_name, team2_name] # Exists in the old location of scores, just something to keep in mind


        # Change number of players per team here !!! ************************************************************************************
        # Must be the same as the same variable in arrange_data.py Edit: don't really have to worry about this now, made the variables belong to the class
        player_stats_per_team = data_formatter.players_per_team
        testing_player_stats = data_formatter.testing_player_stats
        # game_pred_list = data_formatter.game_format(test_player_dict_list, test_game_dict_list, player_stat_names, players_per_team)

        # print(f"game_id: {game_id}, team1_id: {team1_id}, team2_id: {team2_id}, game_year: {game_year}")

        # Change type of stats we are testing on and whether we are testing on player or team stats here !!! **********************************************************************************************************************
        print(f"Getting player stats for game_id: {game_id}")
        team1_player_stats = data_formatter.get_player_stats_for_game(team1_id, game_id, game_year, test_player_dict_list, test_team_dict_list, boxscore_dict_list, player_stat_names, player_stats_per_team, testing_player_stats, True)
        team2_player_stats = data_formatter.get_player_stats_for_game(team2_id, game_id, game_year, test_player_dict_list, test_team_dict_list, boxscore_dict_list, player_stat_names, player_stats_per_team, testing_player_stats, True)

        print(f"after getting player stats for game_id: {game_id}")

        # print(f"team1_player_stats: {team1_player_stats}")

        game_list = [game_id, [team1_id, team2_id], game_year, team_names, [team1_player_stats, team2_player_stats]]

        print(f"team_names: {team_names}")

        game_pred_list.append(game_list)

        # print(f"game_id: {game_id}")
    
    return game_pred_list


def scale_future_game_data(game_pred_list):


    scaled_game_stats_list = []

    
    with open('pkl_files/stat_scalers_list.pkl', 'rb') as file:
        stat_scalers_list = pickle.load(file)

        for game in game_pred_list:
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


    return scaled_game_stats_list


    # Scale the game_pred_list using the loaded scalers

    # (Implementation of scaling logic goes here)

    return scaled_game_pred_list





def main():

    data_formatter = FormatData()

    game_and_player_data_collector_class = CollectGameAndPlayerData()

    players_per_team = data_formatter.players_per_team
    testing_player_stats = data_formatter.testing_player_stats


    train_game_dict_list = import_train_game_data()

    game_pred_list = getting_future_game_data(data_formatter, train_game_dict_list) # True means this is for prediction data, so we want to exclude team names

    team_names = [game_list[3] for game_list in game_pred_list] # Extract team names from game_pred_list

    game_pred_list, alt_score_pred_list, year_pred_list = data_formatter.filter_game_data(game_pred_list, players_per_team, True)
    print(f"alt_score_pred_list: {alt_score_pred_list[:1]}")
    print(f"year_pred_list: {year_pred_list[:1]}")



    # Indv. game data format: [ [ (Team1) [player1_stats], [player2_stats], ... ], [ (Team2) [player1_stats], [player2_stats], ... ] ]
    scaled_game_pred_list = scale_future_game_data(game_pred_list)
    # print(f"scaled_game_pred_list: {scaled_game_pred_list[:1]}")

    data_formatter.export_to_csv(game_pred_list, "csv_data_files/game_pred_data.csv", ['GAME_ID', 'TEAM_IDS', 'GAME_YEAR', 'SCORES', 'TEAM1_PLAYER_STATS', 'TEAM2_PLAYER_STATS'])

    with open('pkl_files/scaled_game_pred_list.pkl', 'wb') as file:
        pickle.dump(scaled_game_pred_list, file)
    
    print(f"len of team_names: {len(team_names)}")
    print(f"team_names: {team_names[:1]}")
    
    with open('pkl_files/team_pred_names.pkl', 'wb') as file:
        pickle.dump(team_names, file)

    with open('pkl_files/year_pred_list.pkl', 'wb') as file:
        pickle.dump(year_pred_list, file)


    # I think this is all I need to do for game_pred_list. Now, all the future games are in the same format as the training data.
    # Next step is to scale the data in the same way as the training data.


    # THE PLAN:
    # Try to get data of future games in the same format as training data. ---- DONE ---- (I think)
    # Scale the data using the same scalers as training data. ---- Done ----  (I think)
        # Import the scalers from scale_data.py ---- Done ----
        # Scale the data. ---- Done ----
    # Feed the data into the model to get predictions. ---- TO DO ----




    # !!!!!! SEE IF THERE ARE ANY BUGS BECAUSE OF GAME_TEST_DATA AND GAME_TRAIN_DATA HAVING EMPTY TEAM 2 PLAYER STAT COLUMNS. 
    # TEAM1_PLAYER_STATS HAS BOTH TEAMS' PLAYER STATS IN IT.

    # To Do: 
    # See why name lists are empty. Might be because of the way I am importing the data/using the wrong data. This could be an issue based in import_train_game_data(),
    # or getting_future_game_data(). Also, why are we using train data again? Shouldn't we be using the test data for this? Maybe we should be using the train data to 
    # get the team names, but the test data to get the player stats? Idk




if __name__ == "__main__":
    main()