import pandas as pd
import sys
import csv
import pickle
from nba_api.stats.endpoints import boxscoretraditionalv2



class FormatData:

    # Change number of players per team here !!! **********************************************************************************************************************
    # Should be 1 for team stats
    players_per_team = 7
    # Change type of stats we are testing on here !!! **********************************************************************************************************************
    testing_player_stats = True

    # Stats of players
    '''
    percentage_player_stat_names = ['W_PCT', 'FG_PCT', 'FG3_PCT', 'FT_PCT']
    average_player_stat_names = ['MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS']
    ranked_player_stat_names = ['W_RANK', 'L_RANK', 'W_PCT_RANK', 'MIN_RANK', 'FGM_RANK', 'FGA_RANK', 'FG_PCT_RANK', 'FG3M_RANK', 'FG3A_RANK', 'FG3_PCT_RANK', 'FTM_RANK', 'FTA_RANK', 'FT_PCT_RANK', 'OREB_RANK', 'DREB_RANK', 'REB_RANK', 'AST_RANK', 'TOV_RANK', 'STL_RANK', 'BLK_RANK', 'BLKA_RANK', 'PF_RANK', 'PFD_RANK', 'PTS_RANK', 'PLUS_MINUS_RANK']
    '''
    # These values are taken from the correlation analysis to trim down the number of stats used for modeling
    trimmed_percentage_player_stat_names = ['W_PCT']
    trimmed_average_player_stat_names = ['BLKA', 'PLUS_MINUS']
    trimmed_ranked_player_stat_names = ['W_RANK', 'L_RANK', 'W_PCT_RANK', 'FG3_PCT_RANK', 'BLK_RANK', 'PF_RANK', 'PLUS_MINUS_RANK']

    percentage_player_stat_names = trimmed_percentage_player_stat_names
    average_player_stat_names = trimmed_average_player_stat_names
    ranked_player_stat_names = trimmed_ranked_player_stat_names


    player_stat_names = [percentage_player_stat_names, average_player_stat_names, ranked_player_stat_names]
    ext_player_stat_names = percentage_player_stat_names + average_player_stat_names + ranked_player_stat_names
    
    # list of stats: 'GP' (might not use this in team_stat_names), 'W_PCT', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'PLUS_MINUS', 
    # 'FGM_RANK', 'FGA_RANK', 'FG_PCT_RANK', 'FG3M_RANK', 'FG3A_RANK', 'FG3_PCT_RANK', 'FTM_RANK', 'FTA_RANK', 'FT_PCT_RANK', 'OREB_RANK', 'DREB_RANK', 'REB_RANK', 'AST_RANK', 'TOV_RANK', 'STL_RANK', 'BLK_RANK', 'BLKA_RANK', 'PLUS_MINUS_RANK'
    # Stats of teams

    percentage_team_stat_names = ['W_PCT', 'FG_PCT', 'FG3_PCT', 'FT_PCT']
    average_team_stat_names = ['OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'PLUS_MINUS']
    ranked_team_stat_names = ['FGM_RANK', 'FGA_RANK', 'FG_PCT_RANK', 'FG3M_RANK', 'FG3A_RANK', 'FG3_PCT_RANK', 'FTM_RANK', 'FTA_RANK', 'FT_PCT_RANK', 'OREB_RANK', 'DREB_RANK', 'REB_RANK', 'AST_RANK', 'TOV_RANK', 'STL_RANK', 'BLK_RANK', 'BLKA_RANK', 'PLUS_MINUS_RANK'] 
    
    team_stat_names = [percentage_team_stat_names, average_team_stat_names, ranked_team_stat_names]
    ext_team_stat_names = percentage_team_stat_names + average_team_stat_names + ranked_team_stat_names
    
    def __init__(self):
        pass

    def import_data(self, file_path):
        return pd.read_csv(file_path)


    def get_data(self):
        train_player_df = self.import_data('csv_data_files/nba_player_averages_2000-2024.csv')
        # print(train_player_df.head())
        test_player_df = self.import_data('csv_data_files/nba_player_averages_2025.csv')
        # print(test_player_df.head())
        train_game_df = self.import_data('csv_data_files/nba_game_logs_2000-2024.csv')
        # print(train_game_df.head())
        test_game_df = self.import_data('csv_data_files/nba_game_logs_2025.csv')
        # print(test_game_df.head())
        test_teams_df = self.import_data('csv_data_files/nba_team_averages_2025.csv')
        # print(test_teams_df.head())
        train_teams_df = self.import_data('csv_data_files/nba_team_averages_2000-2024.csv')
        # print(train_teams_df.head())

        print(f"type of train_player_df: {type(train_player_df)}")

        train_player_dict_list = self.df_to_dict(train_player_df)
        test_player_dict_list = self.df_to_dict(test_player_df)
        train_game_dict_list = self.df_to_dict(train_game_df)
        test_game_dict_list = self.df_to_dict(test_game_df)
        train_teams_dict_list = self.df_to_dict(train_teams_df)
        test_teams_dict_list = self.df_to_dict(test_teams_df)

        return [train_player_dict_list, test_player_dict_list, train_game_dict_list, test_game_dict_list, train_teams_dict_list, test_teams_dict_list]

    def df_to_dict(self, df):
        new_dict = df.to_dict(orient='records')
        '''print(f"sameple of new_dict: {new_dict[:2]}")
        print(f"type of new_dict: {type(new_dict)}")
        value = new_dict[0]['PLAYER_NAME']
        print(f"sample value from new_dict: {value}")'''
        return new_dict
    

    def find_team_names(self, team1_id, team2_id, train_game_dict_list):

        # print(f"train_game_dict_list: {train_game_dict_list}")
        team1_name = ""
        team2_name = ""

        for game in train_game_dict_list:

            if team1_name == "" and team2_name == "":
                break
            
            if game['TEAM_ID'] == team1_id:
                team1_name = game['TEAM_ABBREVIATION']
            if game['TEAM_ID'] == team2_id:
                team2_name = game['TEAM_ABBREVIATION']

        return team1_name, team2_name
    

    def get_team_stats_for_game(self, og_team_id, game_year, team_dict_list):


        team_stats_list = []

        for team in team_dict_list:
            
            # print(f"team: {team}")
            team_id = team['TEAM_ID']
            team_name = team['TEAM_NAME']
            #team_year = team['SEASON']
            # list of stats: 'GP', 'W_PCT', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'PLUS_MINUS', 
            # 'FGM_RANK', 'FGA_RANK', 'FG_PCT_RANK', 'FG3M_RANK', 'FG3A_RANK', 'FG3_PCT_RANK', 'FTM_RANK', 'FTA_RANK', 'FT_PCT_RANK', 'OREB_RANK', 'DREB_RANK', 'REB_RANK', 'AST_RANK', 'TOV_RANK', 'STL_RANK', 'BLK_RANK', 'BLKA_RANK', 'PLUS_MINUS_RANK'
            list_of_pct = [team['W_PCT'], team['FG_PCT'], team['FG3_PCT'], team['FT_PCT']]
            list_of_pergame = [team['OREB'] / team['GP'], team['DREB'] / team['GP'], team['REB'] / team['GP'], team['AST'] / team['GP'], team['TOV'] / team['GP'], team['STL'] / team['GP'], team['BLK'] / team['GP'], team['PLUS_MINUS'] / team['GP']]
            list_of_ranks = [team['FGM_RANK'], team['FGA_RANK'], team['FG_PCT_RANK'], team['FG3M_RANK'], team['FG3A_RANK'], team['FG3_PCT_RANK'], team['FTM_RANK'], team['FTA_RANK'], team['FT_PCT_RANK'], team['OREB_RANK'], team['DREB_RANK'], team['REB_RANK'], team['AST_RANK'], team['TOV_RANK'], team['STL_RANK'], team['BLK_RANK'], team['BLKA_RANK'], team['PLUS_MINUS_RANK']]

            list_of_pct = [float(stat) for stat in list_of_pct]
            list_of_pergame = [float(stat) for stat in list_of_pergame]
            list_of_ranks = [float(stat) for stat in list_of_ranks]

            if og_team_id == team_id: # and team_year == game_year:

                games_played = team['GP']
            
                # List of stats we are predicting on
                '''win_pct = team['W_PCT']
                fg_pct = team['FG_PCT']
                fg3_pct = team['FG3_PCT']
                ft_pct = team['FT_PCT']

                oreb_per_game = team['OREB'] / games_played
                dreb_per_game = team['DREB'] / games_played
                reb_per_game = team['REB'] / games_played
                ast_per_game = team['AST'] / games_played
                tov_per_game = team['TOV'] / games_played
                stl_per_game = team['STL'] / games_played
                blk_per_game = team['BLK'] / games_played
                plus_minus_per_game = team['PLUS_MINUS'] / games_played


                fgm_rank = team['FGM_RANK']
                fga_rank = team['FGA_RANK']
                fg_pct_rank = team['FG_PCT_RANK']
                fg3m_rank = team['FG3M_RANK']
                fg3a_rank = team['FG3A_RANK']
                fg3_pct_rank = team['FG3_PCT_RANK']
                ftm_rank = team['FTM_RANK']
                fta_rank = team['FTA_RANK']
                ft_pct_rank = team['FT_PCT_RANK']
                oreb_rank = team['OREB_RANK']
                dreb_rank = team['DREB_RANK']
                reb_rank = team['REB_RANK']
                ast_rank = team['AST_RANK']
                tov_rank = team['TOV_RANK']
                stl_rank = team['STL_RANK']
                blk_rank = team['BLK_RANK']
                blk_att_rank = team['BLKA_RANK']
                plus_minus_rank = team['PLUS_MINUS_RANK']'''

                team_stats_list = [list_of_pct + list_of_pergame + list_of_ranks]

                break
            
        # print(f"at end of get_team_stats_for_game")

        # team_stats_list = []

        return team_stats_list
    
    def sort_players_by_stat(self, list_of_players, stat_index):
        # Sort the list of players by the specified stat index in descending order
        sorted_players = sorted(list_of_players, key=lambda x: x[stat_index], reverse=True)
        return sorted_players

    def get_player_stats_for_game(self, team_id, game_id, game_year, player_dict_list, team_dict_list, player_stat_names, players_per_team, testing_player_stats, from_predict):

        # should be in the same format as the player stats
        list_of_team_stats = self.get_team_stats_for_game(team_id, game_year, team_dict_list)

        # This line determines whether we are training on player stats or team stats! Comment it out to train on player stats. Edit: don't really have to worry about this now
        # *************************************************************************************************

        if not testing_player_stats:
            return list_of_team_stats

        if not from_predict:
            try:
                df = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id).get_data_frames()[0]
                print(f"successfully fetched box score for game_id={game_id}")
                print("df from boxscoretraditionalv2: ", df.head())
                sys.exit(0)
            except KeyError as e:
                print(f"Error: Could not fetch box score for game_id={game_id}. Using team stats instead.")
                
        
        # sys.exit(0)
        
        # Find players for the given team and game year

        list_of_players = []

        player_counter = 0 # Counter to limit the number of players per team. This can be a parameter to tune the model.
        # max_players_per_team = 8
        max_players_per_team = players_per_team


        for player in player_dict_list:

            player_list = []
            player_team_id = player['TEAM_ID']
            player_season = player['SEASON']

            if (player_team_id != team_id) or (player_season != game_year):
                continue

            # Add relevant player stats to player_list

            games = player['GP']
            if games == 0:
                continue  # avoid division by zero

            if player_counter >= max_players_per_team:
                '''print(f"Reached max players per team: {max_players_per_team}")
                print(f"num of players = {player_counter}")'''
                break

            '''if print_info:  
                print(f"found player")'''
            player_counter += 1 # Only count players that are added to the list


            # percentage_stats = [float(player['W_PCT']), float(player['FG_PCT']), float(player['FG3_PCT']), float(player['FT_PCT'])]
            percentage_stats = [float(player[f"{stat_name}"]) for stat_name in player_stat_names[0]]

            # Ex: Average_stats has 'W_PCT': 0.61, 'MIN': 1411.8183333333334, 'FGM': 144, 'FGA': 324, 'FG_PCT': 0.444, 'FG3M': 0, 'FG3A': 6, 'FG3_PCT': 0.0, 'FTM': 79, 'FTA': 111, 'FT_PCT': 0.712, 'OREB': 107, 'DREB': 206, 'REB': 313, 'AST': 39, 'TOV': 45, 'STL': 30, 'BLK': 8, 'BLKA': 29, 'PF': 119, 'PFD': 1, 'PTS': 367, 'PLUS_MINUS': 104
            # average_stats = [player['MIN'], player['FGM'], player['FGA'], player['FG3M'], player['FG3A'],player['FTM'], player['FTA'], player['OREB'], player['DREB'], player['REB'], player['AST'], player['TOV'], player['STL'], player['BLK'], player['BLKA'], player['PF'], player['PFD'], player['PTS'], player['PLUS_MINUS']]
            average_stats = [player[f"{stat_name}"] for stat_name in player_stat_names[1]]
            average_stats = [float(f"{stat:.2f}")/games for stat in average_stats]

            # Ex: Ranked_stats has 'W_RANK': 38, 'L_RANK': 262, 'W_PCT_RANK': 130, 'MIN_RANK': 188, 'FGM_RANK': 213, 'FGA_RANK': 216, 'FG_PCT_RANK': 171, 'FG3M_RANK': 295, 'FG3A_RANK': 257, 'FG3_PCT_RANK': 295, 'FTM_RANK': 192, 'FTA_RANK': 189, 'FT_PCT_RANK': 259, 'OREB_RANK': 91, 'DREB_RANK': 135, 'REB_RANK': 115, 'AST_RANK': 273, 'TOV_RANK': 265, 'STL_RANK': 232, 'BLK_RANK': 271, 'BLKA_RANK': 263, 'PF_RANK': 220, 'PFD_RANK': 39, 'PTS_RANK': 212, 'PLUS_MINUS_RANK': 95
            # ranked_stats = [float(player['W_RANK']), float(player['L_RANK']), float(player['W_PCT_RANK']), float(player['MIN_RANK']), float(player['FGM_RANK']), float(player['FGA_RANK']), float(player['FG_PCT_RANK']), float(player['FG3M_RANK']), float(player['FG3A_RANK']), float(player['FG3_PCT_RANK']), float(player['FTM_RANK']), float(player['FTA_RANK']), float(player['FT_PCT_RANK']), float(player['OREB_RANK']), float(player['DREB_RANK']), float(player['REB_RANK']), float(player['AST_RANK']), float(player['TOV_RANK']), float(player['STL_RANK']), float(player['BLK_RANK']), float(player['BLKA_RANK']), float(player['PF_RANK']), float(player['PFD_RANK']), float(player['PTS_RANK']), float(player['PLUS_MINUS_RANK'])]
            ranked_stats = [float(player[f"{stat_name}"]) for stat_name in player_stat_names[2]]

            player_list.extend(percentage_stats)
            player_list.extend(average_stats)
            player_list.extend(ranked_stats)

            # print(f"player_list: {player_list}")
            # sys.exit(0)

            list_of_players.append(player_list)

        # print(f"list_of_players: {len(list_of_players)}")

        list_of_players = self.sort_players_by_stat(list_of_players, -1)  # Sort players by last stat (right now), should be plus_minus_rank

        return list_of_players



    def game_format(self, player_dict_list, game_dict_list, team_dict_list, player_stat_names, players_per_team, testing_player_stats, from_predict, get_scores=True):

        # players_per_team preset to 1 to assume we are getting team stats. 


        # print(f"type of train_game_dict: {type(game_dict_list)}")

        
        # This function is where we will create a list of games with player stats and outcomes.

        game_train_list = []

        # Individual game format
        # [game_id, [team1_id, team2_id], game_year, [team1_score, team2_score], [ {player1_team1_stats}, {player2_team1_stats}, ... ], [ {player1_team2_stats}, {player2_team2_stats}, ... ]]

        for game in game_dict_list:

            game_list = []

            matchup = game['MATCHUP']

            game_id = game['GAME_ID']
            team_id = game['TEAM_ID']
            season_id = game['SEASON_ID']

            game_year = game['SEASON']
            team1_score = int(game['PTS']) # Score of the current team1 in the current game
            team2_score = team1_score - int(game['PLUS_MINUS'])  # PLUS_MINUS is from the perspective of the player/team in the log

            team1_name = game['TEAM_ABBREVIATION']

            if not get_scores:
                team1_score = None
                team2_score = None

            # scores = [team1_score]

            #print(f"matchup = {matchup}")
            #print(f"team1_score = {team1_score}, team2_score = {team2_score}")
            #print(f"plus-minus = {game['PLUS_MINUS']}")

            team1_player_stats = self.get_player_stats_for_game(team_id, game_id, game_year, player_dict_list, team_dict_list, player_stat_names, players_per_team, testing_player_stats, from_predict)
             

            # return

            # continue if game_id is not included in a list inside game_train_list
            existing_game_ids = [g[0] for g in game_train_list]

            if game_id in existing_game_ids:
                # append the 2nd team info to the existing game entry

                game_index = existing_game_ids.index(game_id)
                game_list = game_train_list[game_index]

                game_list[1].append(team_id)
                game_list[3].append(team1_score)
                game_list[4].append(team1_player_stats)
                game_list[5].append(team1_name)

                continue

            game_list = [game_id, [team_id], game_year, [team1_score], [team1_player_stats], [team1_name]]

            game_train_list.append(game_list)

        return game_train_list
    

    

    def list_of_stat_types(self,  game_train_list):

        list_of_team_scores = []

        print(f"game_train_list[0][4][0][0]: {game_train_list[0][4][0][0]}, type: {type(game_train_list[0][4][0][0])}")
        print(f"game_train_list[0][4][0][0][0]: {game_train_list[0][4][0][0][0]}, type: {type(game_train_list[0][4][0][0][0])}")

        list_of_stat_lists = [ [] for _ in range(len(game_train_list[0][4][0][0]) )] # Create a list of empty lists for each stat type

        for i, game in enumerate(game_train_list):
            scores = game[3]  # [team1_score, team2_score]
            list_of_team_scores.extend(scores)   

            '''print(f"game[4][0]: {game[4][0]}")
            print(f"game[4][0][0]: {game[4][0][0]}")'''
            try:
                length_of_players = len(game[4][0])  # number of players per team
                length_of_stats = len(game[4][0][0])  # number of stats per player
            except IndexError:
                print(f"IndexError for game index {i}, game ID {game[0]}")

                continue

            # print(f"length_of_stats: {length_of_stats}")

            for team_player_stats in game[4]:  # For each team's player stats
                for player_stats in team_player_stats:  # For each player's stats
                    for stat_index in range(length_of_stats):
                        stat_value = player_stats[stat_index]
                        list_of_stat_lists[stat_index].append(stat_value)
        
        return list_of_team_scores, list_of_stat_lists


    

    def filter_game_data(self, game_data, players_per_team, pred_data=False): # if game does not have two full teams, remove it
        # Filters the game data to only include games with two teams

        filtered_game_data = []
        filtered_score_data = []
        filtered_name_data = []
        filtered_year_data = []

        wrong_len_counter = 0
        
        for i, game in enumerate(game_data):
            len_of_team1 = len(game[4][0])
            len_of_team2 = len(game[4][1])
            if len_of_team1 == 0 or len_of_team2 == 0:
                print(f"Removing game ID {game[0]} due to incomplete team data.")
                continue
            if len_of_team1 != players_per_team or len_of_team2 != players_per_team:
                print(f"Removing game ID {game[0]} due to insufficient players (Team 1: {len_of_team1}, Team 2: {len_of_team2}).")
                wrong_len_counter += 1
                continue
            filtered_game_data.append(game)
            filtered_score_data.append(game[3]) 
            filtered_year_data.append(game[2])
            if not pred_data:
                filtered_name_data.append(game[5])
        
        print(f"Removed {wrong_len_counter} games due to insufficient player data.")

        if not pred_data:

            return filtered_game_data, filtered_score_data, filtered_name_data, filtered_year_data
        
        
        return filtered_game_data, filtered_score_data, filtered_year_data



    def export_to_csv(self, data_list, file_path, fieldnames):
        # Exports the given data list to a CSV file at the specified file path

        # fieldnames = ['GAME_ID', 'TEAM_IDS', 'GAME_YEAR', 'SCORES', 'TEAM1_PLAYER_STATS', 'TEAM2_PLAYER_STATS']

        data_list = [fieldnames] + data_list
        # print(f"type of data_list: {type(data_list)}")
        
        with open(file_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file) #, fieldnames=fieldnames)
            # writer.writeheader()
            
            writer.writerows(data_list)

        return
    




def main():

    # [game_id, [team1_id, team2_id], game_year, [team1_score, team2_score], [  [ [player1_team1_stats], [player2_team1_stats], ... ],   [ [player1_team2_stats], [player2_team2_stats], ... ]    ]
    
    
    # Change number of players per team here !!! **********************************************************************************************************************
    # Should be 1 for team stats
    # players_per_team = 1
    # Change type of stats we are testing on here !!! **********************************************************************************************************************
    # testing_player_stats = False


    data_formatter = FormatData()

    players_per_team = data_formatter.players_per_team
    testing_player_stats = data_formatter.testing_player_stats


    data_dict_list = data_formatter.get_data()

    train_player_dict_list = data_dict_list[0]
    test_player_dict_list = data_dict_list[1]
    train_game_dict_list = data_dict_list[2]
    test_game_dict_list = data_dict_list[3]
    train_teams_list = data_dict_list[4]
    test_teams_list = data_dict_list[5]


    percentage_player_stat_names = data_formatter.percentage_player_stat_names
    average_player_stat_names = data_formatter.average_player_stat_names
    ranked_player_stat_names = data_formatter.ranked_player_stat_names

    player_stat_names = data_formatter.player_stat_names
    ext_player_stat_names = data_formatter.ext_player_stat_names

    percentage_team_stat_names = data_formatter.percentage_team_stat_names
    average_team_stat_names = data_formatter.average_team_stat_names
    ranked_team_stat_names = data_formatter.ranked_team_stat_names

    team_stat_names = data_formatter.team_stat_names
    ext_team_stat_names = data_formatter.ext_team_stat_names

    print(f"stat_names: {len(player_stat_names)}")


    game_train_list = data_formatter.game_format(train_player_dict_list, train_game_dict_list, train_teams_list, player_stat_names, players_per_team, testing_player_stats, from_predict=False)
    game_test_list = data_formatter.game_format(test_player_dict_list, test_game_dict_list, test_teams_list, player_stat_names, players_per_team, testing_player_stats, from_predict=True)
    print(f"game_train_list length: {len(game_train_list)}")

    # These two lists contain lists of 2 scores, one for each team in the game
    # The extended list of team scores now is in the right format, [[team1_score, team2_score], ...]. The only difference between this and
    # the previous two score lists is that this one has all the games scores, while the other two either have the test or training scores.
    game_train_list, alt_score_train_list, name_train_list, year_train_list = data_formatter.filter_game_data(game_train_list, players_per_team)
    game_test_list, alt_score_test_list, name_test_list, year_test_list = data_formatter.filter_game_data(game_test_list, players_per_team)

    ext_list_of_team_scores, list_of_stat_lists = data_formatter.list_of_stat_types(game_train_list)

    total_score_list = [ext_list_of_team_scores, alt_score_test_list, alt_score_train_list]
    # ext_list_of_team_scores and alt_score_train_list have the same amount of data, idk, ig ext_list of team_scores only has training data
    for item in total_score_list:
        print(f"len of score list: {len(item)}")
    print(f"len of game_train_list: {len(game_train_list)}")
    print(f"len of game_test_list: {len(game_test_list)}")
    print(f"len of name_train_list: {len(name_train_list)}")
    print(f"len of name_test_list: {len(name_test_list)}")
    print(f"len of year_train_list: {len(year_train_list)}")
    print(f"len of year_test_list: {len(year_test_list)}")
    print(f"game_train_list[0]: {game_train_list[0]}")

    
    game_ids = [game[0] for game in game_train_list]
    print(f"len of game_ids: {len(game_ids)}")

    print(f"first 3 game ids in game_ids: {game_ids[-50:-47]}")
    print(f"first 3 scores in alt_score_test_list: {alt_score_test_list[-50:-47]}")
    print(f"first 3 names in name_test_list: {name_test_list[-50:-47]}")


    data_formatter.export_to_csv(game_train_list, "csv_data_files/game_train_data.csv", ['GAME_ID', 'TEAM_IDS', 'GAME_YEAR', 'SCORES', 'TEAM1_PLAYER_STATS', 'TEAM2_PLAYER_STATS'])
    data_formatter.export_to_csv(game_test_list, "csv_data_files/game_test_data.csv", ['GAME_ID', 'TEAM_IDS', 'GAME_YEAR', 'SCORES', 'TEAM1_PLAYER_STATS', 'TEAM2_PLAYER_STATS'])
    if testing_player_stats:
        data_formatter.export_to_csv(list_of_stat_lists, "csv_data_files/game_train_stats.csv", ext_player_stat_names)
    else:
        data_formatter.export_to_csv(list_of_stat_lists, "csv_data_files/game_train_stats.csv", ext_team_stat_names)
    data_formatter.export_to_csv(total_score_list, "csv_data_files/game_train_scores.csv", []) #, ['Ext Team Scores', 'Alt Score Test', 'Alt Score Train'])

    with open('pkl_files/team_train_names.pkl', 'wb') as file:
        pickle.dump(name_train_list, file)
    with open('pkl_files/team_test_names.pkl', 'wb') as file:
        pickle.dump(name_test_list, file)
    with open('pkl_files/year_train_list.pkl', 'wb') as file:
        pickle.dump(year_train_list, file)
    with open('pkl_files/year_test_list.pkl', 'wb') as file:
        pickle.dump(year_test_list, file)
    with open('pkl_files/game_id_list.pkl', 'wb') as file:
        pickle.dump(game_ids, file)

    #data_formatter.export_to_csv(name_train_list, "csv_data_files/game_train_names.csv", ['TEAM1_NAME', 'TEAM2_NAME'])
    #data_formatter.export_to_csv(name_test_list, "csv_data_files/game_test_names.csv", ['TEAM1_NAME', 'TEAM2_NAME'])

    # print(f"Sample formatted training game data: {game_train_list[:2]}")





    return

    # THE PLAN
    # Import the data collected in collect_data.py
    # Format data in a way that is suitable for modeling
        # For each game:
        # Find players for each team
        # Normalize players' stats (find averages of ppg, apg, etc over the season)
        # Find the score of the game based on game data


    

    # !!!!!! READ GET_PREDICT_DATA.PY NOTE BEFORE RUNNING THIS !!!!!!
    # Make sure to check the structure of the data and the scalers being used.






if __name__ == "__main__":
    main()