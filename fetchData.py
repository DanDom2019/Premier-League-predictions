import os
from typing import Any, Dict, Optional

import requests

BASE_URL = "https://api.football-data.org/v4"


class FootballDataConfigurationError(RuntimeError):
    """Raised when an API-backed operation is used without provider credentials."""


def get_football_api_token() -> str:
    """Return the provider token without requiring it during application startup."""
    token = os.environ.get("FOOTBALL_API_TOKEN", "").strip()
    if not token:
        raise FootballDataConfigurationError(
            "Football data provider credentials are not configured."
        )
    return token


def fetch(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    url = BASE_URL + path
    headers = {
        "X-Auth-Token": get_football_api_token(),
        "Accept": "application/json",
    }
    resp = requests.get(url, headers=headers, params=params or {})
    resp.raise_for_status()
    return resp.json()

def fetch_leagues_teams():
    data = fetch("/competitions")
    return data

def fetch_all_teams_matches_in_season(leagueId, season=None):
    """
    Fetches all matches for a specific league and season.
    :param leagueId: The ID of the league to fetch matches for.
    :param season: Optional season to filter matches.
    :return: A list of matches in the specified league and season.
    """
    params = {
        "status": "FINISHED",
        "season": season
    }
    
    data = fetch(f"/competitions/{leagueId}/matches", params=params)
    return data

def fetch_teams():
    data = fetch("/teams")
    return data

def load_team_data(teamId : int):
    """
    Fetches teams for a specific league.
    :param leagueid: The ID of the league to fetch teams for.
    :return: A list of teams in the specified league.
    """
    data = fetch(f"/teams/{teamId}")
    return data

def load_team_match_upcoming_match(teamId : int):
    """
    Fetches match history for a specific team.
    :param teamId: The ID of the team to fetch matches for.
    :return: A list of matches played by the specified team.
    """
    data = fetch(f"/teams/{teamId}/matches")
    return data

def load_last_10_match(teamId : int):
    """
    Fetches the last 10 matches for a specific team.
    :param teamId: The ID of the team to fetch matches for.
    :return: A list of the last 10 matches played by the specified team.
    """
    data = fetch(f"/teams/{teamId}/matches?status=FINISHED&limit=10")
    return data


def load_seasons_matches_history(leagueId, season=None):
    """
    Fetches all Premier League matches for a given season.
    Optionally filters for a specific team.
    :param season: The year of the season (e.g., 2024 for 2024-25).
    :param team_id: Optional team ID to filter matches for a specific team.
    :return: A list of matches.
    """
    params = {
        "status": "FINISHED",
        "season": season
    }
    data = fetch(f"/competitions/{leagueId}/matches", params=params)
    if not data:
        print(f"Failed to fetch matches for season {season}")
        return 
    if data['resultSet']['count'] == 0:
        print(f"No matches found for season {season}...")
        return 

    matches = data['matches']
    return matches

def calculate_league_averages(matches):
    """
    Calculates the average home and away goals for a whole season.
    :param matches: A list of match data dictionaries.
    :return: A dictionary containing the calculated averages.
    """
    total_home_goals = 0
    total_away_goals = 0
    # valid_matches_count replaces len(matches) to ensure we only count games with scores
    valid_matches_count = 0 

    if not matches:
        return {"avg_home_goals": 0, "avg_away_goals": 0}

    for match in matches:
        # Ensure the score and fullTime keys exist and are not None
        if match.get('score') and match['score'].get('fullTime'):
            home_goals = match['score']['fullTime'].get('home')
            away_goals = match['score']['fullTime'].get('away')
            
            # Only count this match if scores are actually present (not None)
            if home_goals is not None and away_goals is not None:
                total_home_goals += home_goals
                total_away_goals += away_goals
                valid_matches_count += 1

    if valid_matches_count == 0:
        return {"avg_home_goals": 0, "avg_away_goals": 0}

    avg_home_goals = total_home_goals / valid_matches_count
    avg_away_goals = total_away_goals / valid_matches_count

    return {
        "avg_home_goals": round(avg_home_goals, 3),
        "avg_away_goals": round(avg_away_goals, 3)
    }
#fetch the matches based on the team id and season and numbers of matches as requested
def retrieve_matches_for_team(leagueId,season, team_id, numsMatches=None,noMatch=False):
    start_season=season
    matches = load_seasons_matches_history(leagueId=leagueId, season=season)
    if noMatch:
            print(f"No matches found for season {season}!")
            return None
    # check if we need to include the previous seasons as well
    #if there is no matches for the current season, we can fetch the previous seasons
    combo = False
    noShow = False
    if matches is None:
        
        noShow=True
        season = season - 1
        # Fetch matches for the previous season
        matches = load_seasons_matches_history(leagueId=leagueId, season=season)
        if matches is None:
            matches = []
      #if current season matches dont have enough matches, we can fetch the previous seasons
    if team_id:
        # Filter matches involving the specified team
        matches = filter_matches_by_team_id(team_id, matches)
        current_matches_count = len(matches)
        if not numsMatches:
            print(f"Found {len(matches)} matches for team ID {team_id} in season {start_season}")
            return matches
        if current_matches_count < numsMatches:
            # Fetch matches from previous seasons until we have enough matches
            while current_matches_count < numsMatches:
                combo=True
                season -= 1
                previous_matches = load_seasons_matches_history(leagueId=leagueId, season=season)
                if not previous_matches:
                    print(f"No more matches found for season {season}")
                    break
                previous_matches = filter_matches_by_team_id(team_id, previous_matches)
                matches.extend(previous_matches)
                current_matches_count = len(matches)

    
    # Sort matches by date in descending order to get the most recent ones first.
    matches.sort(key=lambda match: match['utcDate'], reverse=True)
    
    # If numsMatches is specified, return only that number of matches.
    if numsMatches:
        final_matches = matches[:numsMatches]
        
        # print an accurate message based on the flags and stored season
        if combo is False and noShow is False:
            print(f" Found {len(final_matches)} matches for team ID {team_id} in season {start_season}")
        elif noShow is True:
            print(f"Found {len(final_matches)} matches for team ID {team_id} from season {start_season - 1} and earlier.")
        else: # This means 'combo' is True
            print(f"Found {len(final_matches)} matches for team ID {team_id} by combining season {start_season} with previous seasons.")

        return final_matches

    return matches


def filter_matches_by_team_id(team_id, matches):
    matches = [match for match in matches if match['homeTeam']['id'] == team_id or match['awayTeam']['id'] == team_id]
    return matches

#test and example usage
if __name__ == "__main__":

    data=retrieve_matches_for_team(leagueId=2021, season=2025, team_id=65, numsMatches=None, noMatch=True)
    print(data)
