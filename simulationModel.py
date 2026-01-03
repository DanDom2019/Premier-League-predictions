from scipy.stats import poisson
import json
import os
from datetime import datetime
from prosessData import calculate_final_team_stats 
from fetchData import retrieve_matches_for_team, calculate_league_averages, filter_matches_by_team_id

def get_current_season():
    """
    Determines the current football season based on the current date.
    Season starts on August 1st of each year.
    
    Examples:
    - If today is July 15, 2025 -> returns 2024 (previous season)
    - If today is August 15, 2025 -> returns 2025 (current season)
    - If today is March 10, 2025 -> returns 2024 (current season)
    """
    now = datetime.now()
    current_year = now.year
    current_month = now.month
    
    # If we're before August, we're still in the previous season
    if current_month < 8:
        return current_year - 1
    else:
        return current_year

def predict_match(home_team_id, away_team_id, league_id):
    """
    Simulates a match by dynamically calculating team stats based on current season performance,
    using the previous season's data as a baseline during transitional periods.

    :param home_team_id: The ID for the home team.
    :param away_team_id: The ID for the away team.
    :param league_id: The ID for the league the match is in.
    :return: A dictionary containing the prediction probabilities and details.
    """
    print("running")
    current_season = get_current_season()  # Dynamically determine current season

    # 1. Fetch ALL league matches ONCE (this is the only API call needed!)
    all_matches_current_season = retrieve_matches_for_team(league_id, current_season, team_id=None) or []
    
    # 2. Calculate league averages from the fetched data
    league_averages = calculate_league_averages(all_matches_current_season)
    
    # If current season has no valid matches (all zeros), fall back to previous season
    if league_averages['avg_home_goals'] == 0 and league_averages['avg_away_goals'] == 0:
        print(f"No valid matches found for season {current_season}, falling back to previous season")
        previous_season = current_season - 1
        all_matches_previous_season = retrieve_matches_for_team(league_id, previous_season, team_id=None) or []
        if all_matches_previous_season:
            league_averages = calculate_league_averages(all_matches_previous_season)
            # Also update the matches to use previous season for team stats
            all_matches_current_season = all_matches_previous_season
            current_season = previous_season
        else:
            return {"error": f"No match data found for the league in season {current_season} or {previous_season} to calculate averages."}
    
    if not all_matches_current_season:
        return {"error": f"No match data found for the league in season {current_season} to calculate averages."}
    
    print(f"got league averages: {league_averages}")
    
    # 3. Filter matches for each team from the already-fetched data (no additional API calls!)
    home_team_matches = filter_matches_by_team_id(home_team_id, all_matches_current_season)
    away_team_matches = filter_matches_by_team_id(away_team_id, all_matches_current_season)
    
    # 4. Calculate dynamic stats for both teams using the filtered data
    try:
        home_team_stats = calculate_final_team_stats(current_season, league_id, home_team_id, league_averages, team_matches=home_team_matches)
        # Validate that stats dict has all required keys
        required_keys = ["attack_strength_home", "defense_strength_home", "attack_strength_away", "defense_strength_away"]
        if not home_team_stats or not isinstance(home_team_stats, dict) or not all(key in home_team_stats for key in required_keys):
            print(f"Warning: home_team_stats is invalid or incomplete: {home_team_stats}")
            home_team_stats = {
                "attack_strength_home": 0,
                "defense_strength_home": 0,
                "attack_strength_away": 0,
                "defense_strength_away": 0
            }
    except Exception as e:
        print(f"Error calculating home team stats: {e}")
        import traceback
        traceback.print_exc()
        home_team_stats = {
            "attack_strength_home": 0,
            "defense_strength_home": 0,
            "attack_strength_away": 0,
            "defense_strength_away": 0
        }
    
    try:
        away_team_stats = calculate_final_team_stats(current_season, league_id, away_team_id, league_averages, team_matches=away_team_matches)
        # Validate that stats dict has all required keys
        required_keys = ["attack_strength_home", "defense_strength_home", "attack_strength_away", "defense_strength_away"]
        if not away_team_stats or not isinstance(away_team_stats, dict) or not all(key in away_team_stats for key in required_keys):
            print(f"Warning: away_team_stats is invalid or incomplete: {away_team_stats}")
            away_team_stats = {
                "attack_strength_home": 0,
                "defense_strength_home": 0,
                "attack_strength_away": 0,
                "defense_strength_away": 0
            }
    except Exception as e:
        print(f"Error calculating away team stats: {e}")
        import traceback
        traceback.print_exc()
        away_team_stats = {
            "attack_strength_home": 0,
            "defense_strength_home": 0,
            "attack_strength_away": 0,
            "defense_strength_away": 0
        }
    
    print("successfully calculated team stats")
    # 3. Calculate expected goals (lambda) using dynamic strengths
    lambda_home = (home_team_stats['attack_strength_home'] * away_team_stats['defense_strength_away'] * league_averages['avg_home_goals'])
                   
    lambda_away = (away_team_stats['attack_strength_away'] * home_team_stats['defense_strength_home'] * league_averages['avg_away_goals'])
    print("calculated expected goals")
    # 4. Simulate outcomes using the Poisson distribution
    max_goals = 7
    home_win_prob, away_win_prob, draw_prob = 0, 0, 0
    score_probabilities = {}

    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            prob = poisson.pmf(home_goals, lambda_home) * poisson.pmf(away_goals, lambda_away)
            scoreline = f"{home_goals}-{away_goals}"
            score_probabilities[scoreline] = prob
            
            if home_goals > away_goals:
                home_win_prob += prob
            elif away_goals > home_goals:
                away_win_prob += prob
            else:
                draw_prob += prob

    total_calculated_prob = home_win_prob + away_win_prob + draw_prob
    
    if total_calculated_prob > 0:
        home_win_prob /= total_calculated_prob
        away_win_prob /= total_calculated_prob
        draw_prob /= total_calculated_prob
    # Sort the score probabilities to find the top 5
    sorted_scores = sorted(score_probabilities.items(), key=lambda item: item[1], reverse=True)
    top_five_scores = [
        {"score": score, "probability": float(round(prob * 100, 2))}
        for score, prob in sorted_scores[:5]
    ]

    return {
        "home_team_win_probability": float(round(home_win_prob * 100, 2)),
        "away_team_win_probability": float(round(away_win_prob * 100, 2)),
        "draw_probability": float(round(draw_prob * 100, 2)),
        "predicted_goals_home": float(round(lambda_home, 2)),
        "predicted_goals_away": float(round(lambda_away, 2)),
        "top_five_scores": top_five_scores,  # New field with top 5 scores
        "league_averages": league_averages,
        "home_team_stats": home_team_stats,
        "away_team_stats": away_team_stats,
        "home_expected_goals": lambda_home,
        "away_expected_goals": lambda_away,


    }

# Example of how to run it directly
if __name__ == '__main__':
    prediction = predict_match(home_team_id=57, away_team_id=66, league_id=2021)
    print("--- Dynamic Match Simulation Result ---")
    print(json.dumps(prediction, indent=4))