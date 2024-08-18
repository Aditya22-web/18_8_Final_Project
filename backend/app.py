from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
import pandas as pd
from ml_model import preprocess_data, train_model, predict_best_players

app = Flask(__name__)
CORS(app)

RAPIDAPI_KEY = '9673d73c4emsh2baa879c5f2b5fdp1f952bjsnd0d83d1664ee'
RAPIDAPI_HOST = 'cricbuzz-cricket.p.rapidapi.com'

@app.route('/player_stats', methods=['GET'])
def get_player_stats():
    player_id = request.args.get('id')
    if not player_id:
        return jsonify({"error": "Player ID is required"}), 400

    batting_url = f"https://cricbuzz-cricket.p.rapidapi.com/stats/v1/player/{player_id}/batting"
    bowling_url = f"https://cricbuzz-cricket.p.rapidapi.com/stats/v1/player/{player_id}/bowling"

    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": RAPIDAPI_HOST
    }

    try:
        batting_response = requests.get(batting_url, headers=headers)
        bowling_response = requests.get(bowling_url, headers=headers)

        batting_data = batting_response.json() if batting_response.status_code == 200 else None
        bowling_data = bowling_response.json() if bowling_response.status_code == 200 else None

        player_stats = {
            "batting": batting_data,
            "bowling": bowling_data
        }

        return jsonify(player_stats)
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 500

@app.route('/recommend_players', methods=['POST'])
def recommend_players():
    try:
        data = request.get_json()
        app.logger.info(f"Received data: {data}")

        if not data or 'player_stats' not in data or 'pitch_conditions' not in data:
            return jsonify({"error": "Invalid input data. Both player_stats and pitch_conditions are required."}), 400

        # Convert player_stats to DataFrame, ensuring it's a list of dictionaries
        if isinstance(data['player_stats'], dict):
            player_stats = pd.DataFrame([data['player_stats']], index=[0])
        elif isinstance(data['player_stats'], list):
            player_stats = pd.DataFrame(data['player_stats'], index=range(len(data['player_stats'])))
        else:
            return jsonify({"error": "Invalid player_stats format. Expected a dictionary or a list of dictionaries."}), 400

        if player_stats.empty:
            return jsonify({"error": "Player stats data is empty"}), 400

        # Convert pitch_conditions to DataFrame, ensuring it's a single row
        if isinstance(data['pitch_conditions'], dict):
            pitch_conditions = pd.DataFrame([data['pitch_conditions']], index=[0])
        else:
            return jsonify({"error": "Invalid pitch_conditions format. Expected a dictionary."}), 400

        app.logger.info(f"Player stats shape: {player_stats.shape}")
        app.logger.info(f"Pitch conditions shape: {pitch_conditions.shape}")
        app.logger.info(f"Player stats columns: {player_stats.columns}")
        app.logger.info(f"Pitch conditions columns: {pitch_conditions.columns}")

        # Preprocess data
        try:
            X = preprocess_data(player_stats, pitch_conditions)
        except ValueError as e:
            app.logger.error(f"Error in preprocessing: {str(e)}")
            return jsonify({"error": f"Error in preprocessing: {str(e)}"}), 400
        except Exception as e:
            app.logger.error(f"Unexpected error in preprocessing: {str(e)}")
            return jsonify({"error": "An unexpected error occurred during preprocessing"}), 500

        app.logger.info(f"Preprocessed X shape: {X.shape}")

        # Train the model
        try:
            model = train_model(X)
        except Exception as e:
            app.logger.error(f"Error in model training: {str(e)}")
            return jsonify({"error": "An error occurred during model training"}), 500

        # Make predictions
        try:
            predictions = predict_best_players(model, X)
        except Exception as e:
            app.logger.error(f"Error in making predictions: {str(e)}")
            return jsonify({"error": "An error occurred while making predictions"}), 500

        app.logger.info(f"Predictions shape: {predictions.shape}")

        # Sort players by their prediction scores and create a response
        try:
            player_names = player_stats['name'].tolist()
            player_scores = predictions.tolist()
            sorted_players = sorted(zip(player_names, player_scores), key=lambda x: x[1], reverse=True)

            response = {
                "recommended_players": [
                    {"name": name, "score": score} for name, score in sorted_players
                ]
            }

            return jsonify(response)
        except KeyError as e:
            app.logger.error(f"KeyError: {str(e)}. 'name' column not found in player_stats.")
            return jsonify({"error": "Invalid player data structure. 'name' field is missing."}), 400
    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500

if __name__ == '__main__':
    app.run(debug=True)
