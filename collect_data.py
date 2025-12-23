from nba_api.stats.endpoints import leaguedashplayerstats, leaguedashteamstats, leaguegamelog, boxscoretraditionalv2
import pandas as pd
import time


class CollectGameAndPlayerData:
    def __init__(self):
        pass

    def collect_teamstats_data(self, season):
        pass

    def collect_data(self, season, endpoint): #endpoint='LeagueDashPlayerStats'):

        # Get player averages for seasons between 2000-01 and 2024-25

        # Store data for all seasons for individual players in a list

        try:
            all_player_data = []

            season_str = f"{season}-{str(season + 1)[-2:]}"

            # data = leaguedashplayerstats.LeagueDashPlayerStats(season=season_str)

            # Resolve the endpoint name dynamically. First try the leaguedashplayerstats module
            # (keeps compatibility with the original call), otherwise try the top-level
            # nba_api.stats.endpoints package.
            try:
                endpoint_callable = getattr(leaguedashplayerstats, endpoint)
            except AttributeError:
                import nba_api.stats.endpoints as endpoints_module
                endpoint_callable = getattr(endpoints_module, endpoint)

            if endpoint != 'BoxScoreTraditionalV3':
                data = endpoint_callable(season=season_str)
            else:
                data = endpoint_callable(game_id=season)  # Example game ID for boxscore data

            print(f"endpoint = {endpoint}")
            df = data.get_data_frames()[0]
            # print(f"df = {df}")
        
            df['SEASON'] = season_str

            # FIX ABOVE CODE ABOUT SEASON STUFF
            all_player_data.append(df)
        except Exception as e:
            print(f"Error retrieving data for season {season_str}: {e}")
            time.sleep(1)  # wait before retrying or moving on
            self.collect_data(season, endpoint)
        
        return all_player_data
    

    def get_targeted_data(self, range_start, range_end, endpoint):

        all_data = []


        for season in range(range_start, range_end):
            print(f"In season: {season}, endpoint: {endpoint}")
            player_data = self.collect_data(season, endpoint)
            all_data.extend(player_data)

        players_df_test = pd.concat(all_data, ignore_index=True)

        return players_df_test
    
    def get_boxscore_targeted_data(self, game_ids):
        
        all_data = []


        for game_id in game_ids:
            print(f"Getting boxscore data for game_id: {game_id}")
            boxscore_data = self.collect_data(game_id, 'BoxScoreTraditionalV2')
            all_data.extend(boxscore_data)

        boxscores_df = pd.concat(all_data, ignore_index=True)

        return boxscores_df
    
            

    def get_player_data(self):

        starting_test_season = 2017


        print(f"Collecting player data...")
        test_players_df = self.get_targeted_data(starting_test_season, 2025, 'LeagueDashPlayerStats')
        test_players_df.to_csv('csv_data_files/nba_player_averages_2000-2024.csv', index=False)

        train_players_df = self.get_targeted_data(2025, 2026, 'LeagueDashPlayerStats')
        train_players_df.to_csv('csv_data_files/nba_player_averages_2025.csv', index=False)

        print(f"Collecting game log data...")
        test_games_df = self.get_targeted_data(starting_test_season, 2025, 'LeagueGameLog')
        test_games_df.to_csv('csv_data_files/nba_game_logs_2000-2024.csv', index=False)

        '''train_games_df = self.get_targeted_data(2025, 2026, 'LeagueGameLog')
        train_games_df.to_csv('csv_data_files/nba_game_logs_2025.csv', index=False)

        print(f"Collecting team data...")
        train_teams_df = self.get_targeted_data(starting_test_season, 2025, 'LeagueDashTeamStats')
        train_teams_df.to_csv('csv_data_files/nba_team_averages_2000-2024.csv', index=False)

        test_teams_df = self.get_targeted_data(2025, 2026, 'LeagueDashTeamStats')
        test_teams_df.to_csv('csv_data_files/nba_team_averages_2025.csv', index=False)'''
        
        test_games_ids = test_games_df['GAME_ID'].tolist()
        unique_test_games_ids = list(dict.fromkeys(test_games_ids))

        print(f"getting boxscore data...")
        train_games_players_df = self.get_targeted_data(2025, 2026, 'BoxScoreTraditionalV3')
        train_games_players_df.to_csv('csv_data_files/nba_boxscores_2000-2024.csv', index=False)


        return test_players_df, train_players_df, test_games_df, train_games_df, test_teams_df, train_teams_df

        # At the moment, the plan is to use only player and game log data for modeling.
        # Boxscore data will be implimented if individual player stats need to be calculated/predicted. 




def main():

    

    game_and_player_data_collector_class = CollectGameAndPlayerData()

    # team train and testing csv file data are kind of messed up somehow, like idk how the games played are correct, but we will see if the stats are correct
    test_players_df, train_players_df, test_games_df, train_games_df, test_teams_df, train_teams_df = game_and_player_data_collector_class.get_player_data()

    print("Training Data Sample:")
    print(train_players_df.head())

    print("Test Data Sample:")
    print(test_players_df.head())


if __name__ == "__main__":
    main()