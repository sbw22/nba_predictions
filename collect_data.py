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




def main():

    game_and_player_data_collector_class = Collect_game_and_player_data()

    test_players_df, train_players_df, test_games_df, train_games_df, test_boxscores_df, train_boxscores_df = game_and_player_data_collector_class.get_player_data()

    print("Training Data Sample:")
    print(train_players_df.head())

    print("Test Data Sample:")
    print(test_players_df.head())


if __name__ == "__main__":
    main()