"""
Configuration file for ProphitBet Football Prediction Application
Contains league URLs, fixture URLs, and other constants
"""

import os

# File paths and directories
AVAILABLE_LEAGUES_FILEPATH = 'database/storage/leagues/available_leagues.csv'
SAVED_LEAGUES_DIRECTORY = 'database/storage/leagues/saved/'
MODELS_CHECKPOINT_DIRECTORY = 'database/storage/checkpoints/'
CACHE_DIRECTORY = 'cache/'
TEAM_MAPPINGS_FILE = 'team_mappings.json'

# Constants
RANDOM_SEED = 0
DELAY_SECONDS = 10
MAX_RETRIES = 3

# Chrome driver options
CHROME_OPTIONS = [
    '--headless',
    '--no-sandbox',
    '--disable-dev-shm-usage',
    '--disable-gpu',
    '--disable-extensions',
    '--disable-logging',
    '--silent'
]

# Define URLs for historical match data from football-data.co.uk (1993–2025 seasons)
LEAGUE_URLS = {
    'England': {
        'England-Premier League': [(f'https://www.football-data.co.uk/mmz4281/{y-1:02d}{y%100:02d}/E0.csv', y) for y in range(1993, 2025)],
        'England-Championship': [(f'https://www.football-data.co.uk/mmz4281/{y-1:02d}{y%100:02d}/E1.csv', y) for y in range(1993, 2025)],
        'England-League 1': [(f'https://www.football-data.co.uk/mmz4281/{y-1:02d}{y%100:02d}/E2.csv', y) for y in range(1993, 2025)],
        'England-League 2': [(f'https://www.football-data.co.uk/mmz4281/{y-1:02d}{y%100:02d}/E3.csv', y) for y in range(1993, 2025)],
        'England-National League': [(f'https://www.football-data.co.uk/mmz4281/{y-1:02d}{y%100:02d}/EC.csv', y) for y in range(2005, 2025)]
    },
    'Scotland': {
        'Scotland-Premiership': [(f'https://www.football-data.co.uk/mmz4281/{y-1:02d}{y%100:02d}/SC0.csv', y) for y in range(1994, 2025)],
        'Scotland-Championship': [(f'https://www.football-data.co.uk/mmz4281/{y-1:02d}{y%100:02d}/SC1.csv', y) for y in range(1994, 2025)],
    },
    'Germany': {
        'Germany-Bundesliga': [(f'https://www.football-data.co.uk/mmz4281/{y-1:02d}{y%100:02d}/D1.csv', y) for y in range(1994, 2025)],
        'Germany-Bundesliga 2': [(f'https://www.football-data.co.uk/mmz4281/{y-1:02d}{y%100:02d}/D2.csv', y) for y in range(1994, 2025)]
    },
    'Italy': {
        'Italy-Serie A': [(f'https://www.football-data.co.uk/mmz4281/{y-1:02d}{y%100:02d}/I1.csv', y) for y in range(1994, 2025)],
        'Italy-Serie B': [(f'https://www.football-data.co.uk/mmz4281/{y-1:02d}{y%100:02d}/I2.csv', y) for y in range(1994, 2025)]
    },
    'Spain': {
        'Spain-La Liga': [(f'https://www.football-data.co.uk/mmz4281/{y-1:02d}{y%100:02d}/SP1.csv', y) for y in range(1994, 2025)],
        'Spain-La Liga 2': [(f'https://www.football-data.co.uk/mmz4281/{y-1:02d}{y%100:02d}/SP2.csv', y) for y in range(1994, 2025)]
    },
    'France': {
        'France-Ligue 1': [(f'https://www.football-data.co.uk/mmz4281/{y-1:02d}{y%100:02d}/F1.csv', y) for y in range(1994, 2025)],
        'France-Ligue 2': [(f'https://www.football-data.co.uk/mmz4281/{y-1:02d}{y%100:02d}/F2.csv', y) for y in range(1994, 2025)]
    },
    'Netherlands': {
        'Netherlands-Eredivisie': [(f'https://www.football-data.co.uk/mmz4281/{y-1:02d}{y%100:02d}/N1.csv', y) for y in range(1993, 2025)]
    },
    'Belgium': {
        'Belgium-Jupiler League': [(f'https://www.football-data.co.uk/mmz4281/{y-1:02d}{y%100:02d}/B1.csv', y) for y in range(1995, 2025)]
    },
    'Greece': {
        'Greece-Ethniki Katigoria': [(f'https://www.football-data.co.uk/mmz4281/{y-1:02d}{y%100:02d}/G1.csv', y) for y in range(1994, 2025)]
    },
    'Turkey': {
        'Turkey-Futbol Ligi 1': [(f'https://www.football-data.co.uk/mmz4281/{y-1:02d}{y%100:02d}/T1.csv', y) for y in range(1994, 2025)]
    },
    'Portugal': {
        'Portugal-Liga NOS': [(f'https://www.football-data.co.uk/mmz4281/{y-1:02d}{y%100:02d}/P1.csv', y) for y in range(1994, 2025)]
    },
    'Brazil': {
        'Brazil-Serie A': [('https://www.football-data.co.uk/new/BRA.csv', y) for y in range(2012, 2025)]
    },
    'Argentina': {
        'Argentina-Primera Division': [('https://www.football-data.co.uk/new/ARG.csv', y) for y in range(2012, 2025)]
    },
    'Austria': {
        'Austria-Bundesliga': [('https://www.football-data.co.uk/new/AUT.csv', y) for y in range(2012, 2025)]
    },
    'China': {
        'China-Super League': [('https://www.football-data.co.uk/new/CHN.csv', y) for y in range(2012, 2025)]
    },
    'Denmark': {
        'Denmark-Superliga': [('https://www.football-data.co.uk/new/DNK.csv', y) for y in range(2012, 2025)]
    },
    'Finland': {
        'Finland-Veikkausliiga': [('https://www.football-data.co.uk/new/FIN.csv', y) for y in range(2012, 2025)]
    },
    'Ireland': {
        'Ireland-Premier Division': [('https://www.football-data.co.uk/new/IRL.csv', y) for y in range(2012, 2025)]
    },
    'Japan': {
        'Japan-J1 League': [('https://www.football-data.co.uk/new/JPN.csv', y) for y in range(2012, 2025)]
    },
    'Mexico': {
        'Mexico-Liga MX': [('https://www.football-data.co.uk/new/MEX.csv', y) for y in range(2012, 2025)]
    },
    'Norway': {
        'Norway-Eliteserien': [('https://www.football-data.co.uk/new/NOR.csv', y) for y in range(2012, 2025)]
    },
    'Poland': {
        'Poland-Ekstraklasa': [('https://www.football-data.co.uk/new/POL.csv', y) for y in range(2012, 2025)]
    },
    'Romania': {
        'Romania-Liga I': [(f'https://www.football-data.co.uk/new/ROU.csv', y) for y in range(2012, 2025)]
    },
    'Russia': {
        'Russia-Premier League': [('https://www.football-data.co.uk/new/RUS.csv', y) for y in range(2012, 2025)]
    },
    'Sweden': {
        'Sweden-Allsvenskan': [('https://www.football-data.co.uk/new/SWE.csv', y) for y in range(2012, 2025)]
    },
    'Switzerland': {
        'Switzerland-Super League': [('https://www.football-data.co.uk/new/SWZ.csv', y) for y in range(2012, 2025)]
    },
    'USA': {
        'USA-MLS': [('https://www.football-data.co.uk/new/USA.csv', y) for y in range(2012, 2025)]
    },
}

# Define URLs for scraping upcoming match fixtures from footystats.org
FIXTURE_URLS = {
    'England-Premier League': 'https://footystats.org/england/premier-league/fixtures',
    'England-Championship': 'https://footystats.org/england/championship/fixtures',
    'England-League 1': 'https://footystats.org/england/efl-league-one/fixtures',
    'England-League 2': 'https://footystats.org/england/efl-league-two/fixtures',
    'England-National League': 'https://footystats.org/england/national-league',
    'Scotland-Premiership': 'https://footystats.org/scotland/premiership/fixtures',
    'Scotland-Championship': 'https://footystats.org/scotland/championship/fixtures',
    'Germany-Bundesliga': 'https://footystats.org/germany/bundesliga/fixtures',
    'Germany-Bundesliga 2': 'https://footystats.org/germany/2-bundesliga/fixtures',
    'Italy-Serie A': 'https://footystats.org/italy/serie-a/fixtures',
    'Italy-Serie B': 'https://footystats.org/italy/serie-b/fixtures',
    'Spain-La Liga': 'https://footystats.org/spain/la-liga/fixtures',
    'Spain-La Liga 2': 'https://footystats.org/spain/segunda-division/fixtures',
    'France-Ligue 1': 'https://footystats.org/france/ligue-1/fixtures',
    'France-Ligue 2': 'https://footystats.org/france/ligue-2/fixtures',
    'Netherlands-Eredivisie': 'https://footystats.org/netherlands/eredivisie/fixtures',
    'Belgium-Jupiler League': 'https://footystats.org/belgium/pro-league/fixtures',
    'Greece-Ethniki Katigoria': 'https://footystats.org/greece/super-league/fixtures',
    'Turkey-Futbol Ligi 1': 'https://footystats.org/turkey/super-lig/fixtures',
    'Portugal-Liga NOS': 'https://footystats.org/portugal/liga-nos/fixtures',
    'Brazil-Serie A': 'https://footystats.org/brazil/serie-a/fixtures',
    'Argentina-Primera Division': 'https://footystats.org/argentina/primera-division/fixtures',
    'Austria-Bundesliga': 'https://footystats.org/austria/bundesliga/fixtures',
    'China-Super League': 'https://footystats.org/china/chinese-super-league/fixtures',
    'Denmark-Superliga': 'https://footystats.org/denmark/superliga/fixtures',
    'Finland-Veikkausliiga': 'https://footystats.org/finland/veikkausliiga/fixtures',
    'Ireland-Premier Division': 'https://footystats.org/republic-of-ireland/premier-division/fixtures',
    'Japan-J1 League': 'https://footystats.org/japan/j1-league/fixtures',
    'Mexico-Liga MX': 'https://footystats.org/mexico/liga-mx/fixtures',
    'Norway-Eliteserien': 'https://footystats.org/norway/eliteserien/fixtures',
    'Poland-Ekstraklasa': 'https://footystats.org/poland/ekstraklasa/fixtures',
    'Romania-Liga I': 'https://footystats.org/romania/liga-1/fixtures',
    'Russia-Premier League': 'https://footystats.org/russia/premier-league/fixtures',
    'Sweden-Allsvenskan': 'https://footystats.org/sweden/allsvenskan/fixtures',
    'Switzerland-Super League': 'https://footystats.org/switzerland/super-league/fixtures',
    'USA-MLS': 'https://footystats.org/usa/major-league-soccer/fixtures'
}