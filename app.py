from flask import Flask, jsonify, send_from_directory, request
import json
import os
from datetime import datetime
from prosessData import process_last_X_games
from fetchData import load_team_data, load_team_match_upcoming_match
# Import your new prediction function
from simulationModel import predict_match
from flask_cors import CORS

app = Flask(__name__)

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

# Enhanced CORS configuration
CORS(app, 
     origins=['*'],
     methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization'],
     supports_credentials=True)

# Add manual CORS headers as backup
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
    return response

# --- Frontend Route ---
@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

# --- API Info Route ---
@app.route('/api')
def api_info():
    return jsonify({
        "app": "KickCast Match Predictions",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "test": "/test",
            "team_info": "/api/team/{teamId}",
            "last_matches": "/app/team/{teamId}/last10matches?leagueId={leagueId}",
            "next_match": "/api/team/{teamId}/next_match",
            "prediction": "/simulation/predict?home={homeId}&away={awayId}&leagueId={leagueId}"
        }
    })

@app.route('/test')
def test():
    return jsonify({
        "status": "running",
        "message": "Backend is working!",
        "cors": "enabled"
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})



# --- API Routes ---

@app.route('/api/team/<int:teamId>')
def get_team_by_id(teamId):
    """Fetches detailed data for a specific team using its ID."""
    try:
        team_data = load_team_data(teamId)
        if not team_data:
            return jsonify({"error": "Team not found"}), 404
        return jsonify(team_data)
    except Exception as e:
        print(f"Error in get_team_by_id: {e}")
        return jsonify({"error": "An error occurred on the server"}), 500

@app.route('/app/team/<int:teamId>/last10matches')
def get_last_10_matches(teamId):
    """Get the last 10 processed matches for a given team."""
    leagueId = request.args.get('leagueId', type=int)
    season = get_current_season()  # Dynamically determine current season
    if not leagueId:
        return jsonify({"error": "leagueId is required"}), 400
    try:
        matches = process_last_X_games(leagueId=leagueId, teamId=teamId, season=season)
        if not matches:
            return jsonify({"error": "No matches found for this team."}), 404
        return jsonify(matches)
    except Exception as e:
        print(f"Error in get_last_10_matches: {e}")
        return jsonify({"error": "An error occurred while fetching match data."}), 500

@app.route('/api/team/<int:teamId>/next_match')
def get_next_match(teamId):
    """
    Returns one of:
      { "status": "upcoming",     "match":      <match> }  200
      { "status": "season_ended", "last_match": <match> }  200
      { "error": "..." }                                   404 (genuinely no matches)
      { "error": "..." }                                   500 (upstream / API failure)
    """
    try:
        upcoming_matches = load_team_match_upcoming_match(teamId)

        # Malformed/empty upstream payload = API failure, not "no matches".
        if not upcoming_matches or 'matches' not in upcoming_matches:
            return jsonify({"error": "Upstream API returned no match data."}), 500

        matches = upcoming_matches['matches']

        scheduled_matches = [m for m in matches if m.get('status') in ('SCHEDULED', 'TIMED')]
        if scheduled_matches:
            scheduled_matches.sort(key=lambda x: x['utcDate'])
            return jsonify({"status": "upcoming", "match": scheduled_matches[0]})

        finished_matches = [m for m in matches if m.get('status') == 'FINISHED']
        if finished_matches:
            finished_matches.sort(key=lambda x: x['utcDate'], reverse=True)
            return jsonify({"status": "season_ended", "last_match": finished_matches[0]})

        return jsonify({"error": "No matches found for this team."}), 404
    except Exception as e:
        print(f"Error fetching next match: {e}")
        return jsonify({"error": "An error occurred while fetching the next match."}), 500
        

# --- NEW SIMULATION ROUTE ---
@app.route('/simulation/predict')
def run_prediction():
    """
    Runs the simulation based on home and away team IDs from the request.
    """
    home_id = request.args.get('home', type=int)
    away_id = request.args.get('away', type=int)
    league_id = request.args.get('leagueId', type=int)

    if not all([home_id, away_id, league_id]):
        return jsonify({"error": "Missing home, away, or leagueId parameters"}), 400

    try:
        prediction_result = predict_match(home_team_id=home_id, away_team_id=away_id, league_id=league_id)
        if "error" in prediction_result:
            return jsonify(prediction_result), 500
        return jsonify(prediction_result)
    except Exception as e:
        print(f"Error during simulation: {e}")
        return jsonify({"error": "An internal error occurred during the simulation."}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)