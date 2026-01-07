




class GetUnavailablePlayers:
    def __init__(self):
        pass

    def get_unavailable_players(self):
        # Placeholder for actual implementation to get unavailable players
        unavailable_players = []  # Example: ['Player A', 'Player B']
        return unavailable_players
    

    clev = ["Chris Livingston", "Donovan Mitchell", "Larry Nance Jr.", "Max Strus", "Luke Travers"]
    ind = ["Tyrese Haliburton", "Isaiah Jackson", "Bennedict Mathurin", "Taelon Peter", "Obi Toppin"]
    orl = ["Colin Castleton", "Jalen Suggs", "Franz Wagner", "Moritz Wagner"]
    was = ["Kyshawn George", "Corey Kispert", "Cam Whitmore"]
    lal = ["Rui Hachimura", "Austin Reaves", "Adou Thiero", "Gabe Vincent"]
    nol = ["Trey Alexander", "Saddiq Bey", "Hunter Dickinson", "Dejounte Murray", "Herbert Jones"]
    brk = ["Tyson Etienne", "Haywood Highsmith", "Chaney Johnson", "E.J. Liddell", "Ben Saraf"]
    mia = ["Jaime Jaquez Jr.", "Terry Rozier"]
    min = ["Terrence Shannon Jr.", "Joan Beringer", "Enrique Freeman"]
    sas = ["Harrison Ingram", "David Jones Garcia", "Stanley Umude", "Devin Vassell"]
    mem = ["Brandon Clarke", "Cedric Coward", "Zach Edey", "Ty Jerome", "John Konchar", "Jahmai Mashack", "Ja Morant", "Scotty Pippen Jr."]
    okc = ["Jaylin Williams", "Isaiah Hartenstein", "Ousmane Dieng", "Nikola Topic", "Thomas Sorber"] # (?)
    cha = ["Grant Williams", "Mason Plumlee", "Ryan Kalkbrenner"] #(?)
    chi = ["Josh Giddey", "Zach Collins", "Jalen Smith", "Trentyn Flowers", "Noa Essengue", "Emanuel Miller"]
    bos = ["Jayson Tatum", "Ron Harper Jr.", "Josh Minott", "Max Shulga", "Amari Williams"]
    gsw = ["Seth Curry"] # (?)
    lac = ["Bradley Beal", "Bogdan Bogdanovic", "Derrick Jones Jr.", "Chris Paul"]
    den = ["Nikola Jokic", "Tamar Bates", "Cameron Johnson", "Jonas Valanciunas"]
    phi = ["Trendon Watford", "Kelly Oubre Jr.", "Johni Broome", "Marjon Beauchamp"]
    hou = ["Fred VanVleet", "Tristen Newton", "Isiah Crawford", "Alperen Sengun"]
    det = ["Tobias Harris", "Jalen Duren", "Isaac Jones", "Bobi Clinton", "Wendell Moore Jr."]
    atl = ["N'Faly Dante", "Trae Young"]
    dal = ["Kyrie Irving", "Dereck Lively II", "Dante Exum", "Moussa Cisse", "Miles Kelly"] 

    all_teams = [clev, ind, orl, was, lal, nol, brk, mia, min, sas, mem, okc, cha, chi, bos, gsw, lac, den, phi, hou, det, atl, dal]
    
    comb_teams = [player for team in all_teams for player in team]

    



