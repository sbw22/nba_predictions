from nba_api.stats.endpoints import leaguedashplayerstats, leaguedashteamstats, leaguegamelog, boxscoretraditionalv2, scoreboardv2
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
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

            if endpoint != 'BoxScoreTraditionalV2':
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
    

    def collect_boxscore_data(self, game_id):

        try:
            df = boxscoretraditionalv2.BoxScoreTraditionalV2(
                game_id=game_id,
                timeout=10
            ).get_data_frames()[0]

            return [df]
        except Exception as e:
            print(f"Error retrieving boxscore data for game_id {game_id}: {e}")
            return self.collect_boxscore_data(game_id)
    

    def get_targeted_data(self, range_start, range_end, endpoint):

        all_data = []


        for season in range(range_start, range_end):
            print(f"In season: {season}, endpoint: {endpoint}")
            player_data = self.collect_data(season, endpoint)
            all_data.extend(player_data)

        players_df_test = pd.concat(all_data, ignore_index=True)

        return players_df_test

    
    def get_boxscore_targeted_data(self, game_ids, max_workers=5, requests_per_minute=100):
        """
        Optimized parallel collection with rate limiting.
        
        Args:
            max_workers: number of concurrent threads (start with 5-10)
            requests_per_minute: API rate limit (adjust based on NBA API limits)
        """
        all_data = []
        delay_between_requests = 60.0 / requests_per_minute
        rate_limiter = Semaphore(max_workers)
        
        def fetch_with_rate_limit(game_id):
            with rate_limiter:
                result = self.collect_boxscore_data(game_id)
                time.sleep(delay_between_requests)
                return result
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_game = {executor.submit(fetch_with_rate_limit, gid): gid 
                            for gid in game_ids}
            
            for i, future in enumerate(as_completed(future_to_game)):
                if i % 50 == 0:
                    print(f"Completed {i}/{len(game_ids)} games")
                
                try:
                    boxscore_data = future.result()
                    all_data.extend(boxscore_data)
                except Exception as e:
                    game_id = future_to_game[future]
                    print(f"Failed for game {game_id}: {e}")
        
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    
            

    def get_player_data(self):

        starting_test_season = 2022

        # As of 12/27/2025, all data we are using is from 2000-2024 files.

        print(f"Collecting player data...")
        test_players_df = self.get_targeted_data(starting_test_season, 2025, 'LeagueDashPlayerStats')
        test_players_df.to_csv('csv_data_files/nba_player_averages_2000-2024.csv', index=False)

        train_players_df = self.get_targeted_data(2025, 2026, 'LeagueDashPlayerStats')
        train_players_df.to_csv('csv_data_files/nba_player_averages_2025.csv', index=False)

        print(f"Collecting game log data...")
        test_games_df = self.get_targeted_data(starting_test_season, 2025, 'LeagueGameLog')
        test_games_df.to_csv('csv_data_files/nba_game_logs_2000-2024.csv', index=False)

        # IDK why train_games, train teams, and test teams were commented out, but uncommenting them for now
        train_games_df = self.get_targeted_data(2025, 2026, 'LeagueGameLog')
        train_games_df.to_csv('csv_data_files/nba_game_logs_2025.csv', index=False)

        print(f"Collecting team data...")
        train_teams_df = self.get_targeted_data(starting_test_season, 2025, 'LeagueDashTeamStats')
        train_teams_df.to_csv('csv_data_files/nba_team_averages_2000-2024.csv', index=False)

        test_teams_df = self.get_targeted_data(2025, 2026, 'LeagueDashTeamStats')
        test_teams_df.to_csv('csv_data_files/nba_team_averages_2025.csv', index=False)

        
        test_games_ids = test_games_df['GAME_ID'].tolist()
        unique_test_games_ids = list(dict.fromkeys(test_games_ids))

        print(f"getting boxscore data...")
        #boxscores_df = self.get_targeted_data(2025, 2026, 'BoxScoreTraditionalV2')
        #boxscores_df = self.get_boxscore_targeted_data(unique_test_games_ids)
        #boxscores_df.to_csv('csv_data_files/nba_boxscores_2000-2024.csv', index=False)

        # Commented out above lines after adding rate limiting and multithreading to boxscore data collection


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