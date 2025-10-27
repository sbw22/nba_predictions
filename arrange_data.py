import pandas as pd


def import_data(file_path):
    return pd.read_csv(file_path)











def main():
    train_data = import_data('nba_player_averages_2000-2024.csv')
    print(train_data.head())
    test_data = import_data('nba_player_averages_2025.csv')
    print(test_data.head())






if __name__ == "__main__":
    main()