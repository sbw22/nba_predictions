from scale_data import DataScaler
from arrange_data import FormatData


def format_data(data_scaler, data_formatter):
    train_game_data = data_scaler.import_game_data("csv_data_files/game_train_data.csv")
    list_of_dict_lists = data_formatter.get_data() #'csv_data_files/nba_game_logs_2000-2024.csv')
    train_game_dict_list = list_of_dict_lists[2]


    # getting data for the correlations
    game_scores = [game_data[3] for game_data in train_game_data]
    all_game_stats = [game_data[4] for game_data in train_game_data]

    print(f"len of game_scores: {len(game_scores)}")
    print(f"len of game_stats: {len(all_game_stats)}")

    # Holds the game data that is arranged based off home and away teams: [[home_score, away_score], [home_team_stats, away_team_stats]]
    full_game_data = []

    for i, game in enumerate(game_scores):
        score1, score2 = game[0], game[1]

        game_stats = all_game_stats[i]
        # print(f"game_stats: {game_stats} at index {i}")
        team1_stats = game_stats[0]
        team2_stats = game_stats[1]

        team1_home = False

        # Check if team1 is the home team
        game_log_dict = train_game_dict_list[i]
        matchup_str = game_log_dict['MATCHUP']

        # If ' vs. ' is in the matchup string, team1 is home, else, if ' @ ' is in the string, team1 is away
        if ' vs. ' in matchup_str:
            team1_home = True
        else:
            team1_home = False
        
        if team1_home:
            home_team_stats = team1_stats
            away_team_stats = team2_stats
            home_score = score1
            away_score = score2
        else:
            home_team_stats = team2_stats
            away_team_stats = team1_stats
            home_score = score2
            away_score = score1
        
        game_data = [[home_score, away_score], [home_team_stats, away_team_stats]]

        full_game_data.append(game_data)
    
    return full_game_data




def find_correlations(full_game_data, stat_names):

    from scipy.stats import pearsonr
    import numpy as np

    # Assuming full_game_data is a list of games, where each game is structured as:
    # [[home_score, away_score], [home_team_stats, away_team_stats]]
    # and team_stats is a list of player stats for that team.

    num_stats = len(full_game_data[0][1][0][0])  # Number of stats per player (this is measuring the first games away team's first player's stats I think)
    home_stat_correlations = {stat_name: [] for stat_name in stat_names}
    away_stat_correlations = {stat_name: [] for stat_name in stat_names}

    # Loop through all stats and store all avg stat values for each game. Might change avg stat values to something more personable to the 
    # individual players later.
    for stat_index in range(num_stats):

        stat_name = stat_names[stat_index]

        home_stat_values = []
        away_stat_values = []
        home_score_differences = []
        away_score_differences = []

        for game in full_game_data:
            # Fix score order to always be home - away
            teams = game[1]
            scores = game[0]
            home_team = teams[0]
            away_team = teams[1]
            home_score = scores[0]
            away_score = scores[1]
            home_score_diff = home_score - away_score
            away_score_diff = away_score - home_score

            total_home_stat_value = 0
            total_away_stat_value = 0

            for i, home_player in enumerate(home_team):
                away_player = away_team[i]

                home_player_stat = home_player[stat_index]
                away_player_stat = away_player[stat_index]

                total_home_stat_value += home_player_stat
                total_away_stat_value += away_player_stat
            
            avg_home_stat_value = total_home_stat_value / len(home_team)
            avg_away_stat_value = total_away_stat_value / len(away_team)

            home_stat_values.append(avg_home_stat_value)
            away_stat_values.append(avg_away_stat_value)
            home_score_differences.append(home_score_diff)
            away_score_differences.append(away_score_diff)
        
        # Append correlations to dictionaries
        home_correlation, _ = pearsonr(home_stat_values, home_score_differences)
        away_correlation, _ = pearsonr(away_stat_values, away_score_differences)
        home_stat_correlations[stat_name].append(home_correlation)
        away_stat_correlations[stat_name].append(away_correlation)

        print(f"Stat: {stat_name} | Home Correlation: {home_correlation} | Away Correlation: {away_correlation}")
    



    # Print out the correlations
    for stat_index, correlation in home_stat_correlations.items():
        print(f"Home Stat Index {stat_index}: Correlation with Score Difference = {correlation}")
    
    return home_stat_correlations, away_stat_correlations


def sort_corrs(stat_correlations, title):
    print(f"\n--- {title} ---")
    # Sort the correlations by absolute value in descending order
    sorted_stats = sorted(stat_correlations.items(), key=lambda item: abs(round(item[1][0], 7)), reverse=True)

    for stat_name, correlation in sorted_stats:
        print(f"Stat: {stat_name} | Correlation: {correlation[0]}")
    
    print(f"sorted_stats: {sorted_stats}")

    return sorted_stats


def graph_correlations(stat_correlations, title):
    import matplotlib.pyplot as plt

    stat_names = [item[0] for item in stat_correlations]
    correlations = [item[1][0] for item in stat_correlations]
    
    # Split data into two halves
    mid_point = len(stat_names) // 2
    first_half_names = stat_names[:mid_point]
    first_half_corrs = correlations[:mid_point]
    second_half_names = stat_names[mid_point:]
    second_half_corrs = correlations[mid_point:]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 4))

    # First half (top subplot)
    bars1 = ax1.bar(first_half_names, first_half_corrs, color='skyblue', width=0.4)
    ax1.set_ylabel('Pearson Correlation with Score Difference')
    ax1.set_title(f'{title} - Part 1')
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_ylim(-1, 1)
    
    # Add correlation values on top of bars for first half
    for bar, corr in zip(bars1, first_half_corrs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height,
                 f'{corr:.2f}', ha='center', va='bottom')

    # Second half (bottom subplot)
    bars2 = ax2.bar(second_half_names, second_half_corrs, color='skyblue', width=0.4)
    ax2.set_xlabel('Statistics')
    ax2.set_ylabel('Pearson Correlation with Score Difference')
    ax2.set_title(f'{title} - Part 2')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylim(-1, 1)
    
    # Add correlation values on top of bars for second half
    for bar, corr in zip(bars2, second_half_corrs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, height,
                 f'{corr:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    


def find_trends(home_stat_correlations, away_stat_correlations):
    import matplotlib.pyplot as plt

    sorted_home_corrs = sort_corrs(home_stat_correlations, "Home Team Stat Correlations")
    sorted_away_corrs = sort_corrs(away_stat_correlations, "Away Team Stat Correlations")

    # Create both graphs
    graph_correlations(sorted_home_corrs, "Home Team Stat Correlations with Score Difference")
    graph_correlations(sorted_away_corrs, "Away Team Stat Correlations with Score Difference")
    
    # Show all figures at once
    plt.show()





def main():
    data_scaler = DataScaler()
    data_formatter = FormatData()

    full_game_data = format_data(data_scaler, data_formatter)

    stat_names = data_formatter.ext_team_stat_names
    testing_player_stats = data_formatter.testing_player_stats
    if testing_player_stats:
        stat_names = data_formatter.ext_player_stat_names


    home_stat_correlations, away_stat_correlations = find_correlations(full_game_data, stat_names)

    find_trends(home_stat_correlations, away_stat_correlations)

    # FIX AND VERIFY THAT FIND_CORRELATIONS WORKS PROPERLY


        
        
    
    # team1_stats = [game[0] for game in game_stats]      # Would like to maybe find a way to see which of these two teams are home and away
    # team2_stats = [game[1] for game in game_stats]


    








if __name__ == "__main__":
    main()