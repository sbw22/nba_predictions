from nba_api.stats.endpoints import leaguedashplayerstats, leaguegamelog, boxscoretraditionalv2
import pandas as pd
import time


class Collect_game_and_player_data:
    def __init__(self):
        pass

    def collect_data(self, season, endpoint='LeagueDashPlayerStats'):

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

            data = endpoint_callable(season=season_str)

            df = data.get_data_frames()[0]
            df['SEASON'] = season_str
            all_player_data.append(df)
        except Exception as e:
            print(f"Error retrieving data for season {season_str}: {e}")
            time.sleep(1)  # wait before retrying or moving on
            self.collect_data(season)
        
        return all_player_data
    

    def get_targeted_data(self, range_start, range_end, endpoint):

        all_data = []


        for season in range(range_start, range_end):
            print(f"In season: {season}")
            player_data = self.collect_data(season, endpoint)
            all_data.extend(player_data)

        players_df_test = pd.concat(all_data, ignore_index=True)

        return players_df_test
    
        players_df_test.to_csv('nba_player_averages_2025.csv', index=False)
            

    def get_player_data(self):


        test_players_df = self.get_targeted_data(2000, 2025, 'LeagueDashPlayerStats')
        test_players_df.to_csv('nba_player_averages_2000-2024.csv', index=False)

        train_players_df = self.get_targeted_data(2025, 2026, 'LeagueDashPlayerStats')
        train_players_df.to_csv('nba_player_averages_2025.csv', index=False)

        test_games_df = self.get_targeted_data(2000, 2025, 'LeagueGameLog')
        test_games_df.to_csv('nba_game_logs_2000-2024.csv', index=False)

        train_games_df = self.get_targeted_data(2025, 2026, 'LeagueGameLog')
        train_games_df.to_csv('nba_game_logs_2025.csv', index=False)

        test_boxscores_df = self.get_targeted_data(2000, 2025, 'BoxScoreTraditionalV2')
        test_boxscores_df.to_csv('nba_boxscores_2000-2024.csv', index=False)

        train_boxscores_df = self.get_targeted_data(2025, 2026, 'BoxScoreTraditionalV2')
        train_boxscores_df.to_csv('nba_boxscores_2025.csv', index=False)

        return test_players_df, train_players_df, test_games_df, train_games_df, test_boxscores_df, train_boxscores_df





        # Get test data and store to a csv

        all_player_train_data = []

        # Loop through seasons from 2000 to 2024 for test data and store to csv
        # This data will be used to fit the min-max scaler, because it contains all the test player data.
        for season in range(2000, 2025):

            print(f"In season: {season}")
        
            player_data = self.collect_data(season)
            all_player_train_data.extend(player_data)


        players_df_train = pd.concat(all_player_train_data, ignore_index=True)
        players_df_train.to_csv('nba_player_averages_2000-2024.csv', index=False)


        #################################################################################################################################################################

        # Loop through season 2025 for test data and store to csv

        # This data might be used for the test data, but idk if it will be used yet, because I might get the data another way. 
        all_player_test_data = []
        for season in range(2025, 2026):
            print(f"In season: {season}")
            player_data = self.collect_data(season)
            all_player_test_data.extend(player_data)

        players_df_test = pd.concat(all_player_test_data, ignore_index=True)
        players_df_test.to_csv('nba_player_averages_2025.csv', index=False)



        #################################################################################################################################################################


        all_games_test_data = []

        for season in range(2000, 2025):

            print(f"Collecting game logs for season: {season}")

            try:
                season_str = f"{season}-{str(season + 1)[-2:]}"
                game_log = leaguegamelog.LeagueGameLog(season=season_str)
                games_df = game_log.get_data_frames()[0]
                all_games_test_data.append(games_df)
            except Exception as e:
                print(f"Error retrieving game logs for season {season_str}: {e}")
                time.sleep(1)  # wait before retrying or moving on
                continue
        
        games_df_test = pd.concat(all_games_test_data, ignore_index=True)
        games_df_test.to_csv('nba_game_logs_2000-2024.csv', index=False)



        ##################################################################################################################################################################




        all_games_test_data = []

        for season in range(2025, 2026):

            print(f"Collecting game logs for season: {season}")

            try:
                season_str = f"{season}-{str(season + 1)[-2:]}"
                game_log = leaguegamelog.LeagueGameLog(season=season_str)
                games_df = game_log.get_data_frames()[0]
                all_games_test_data.append(games_df)
            except Exception as e:
                print(f"Error retrieving game logs for season {season_str}: {e}")
                time.sleep(1)  # wait before retrying or moving on
                continue
        
        games_df_test = pd.concat(all_games_test_data, ignore_index=True)
        games_df_test.to_csv('nba_game_logs_2025.csv', index=False)



        return players_df_train, players_df_test



def main():

    game_and_player_data_collector_class = Collect_game_and_player_data()

    test_players_df, train_players_df, test_games_df, train_games_df, test_boxscores_df, train_boxscores_df = game_and_player_data_collector_class.get_player_data()

    print("Training Data Sample:")
    print(train_players_df.head())

    print("Test Data Sample:")
    print(test_players_df.head())


if __name__ == "__main__":
    main()