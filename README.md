# KickCast - Premier League Predictor

**KickCast** (https://kickcast.live) is a full-stack web application that predicts soccer match outcomes using statistical modeling. By leveraging historical match data and the **Poisson Distribution**, it calculates the probabilities of wins, draws, and losses, alongside the most likely specific scorelines.

The application features a Python Flask backend for data processing and simulation, served to a responsive Bootstrap frontend with Chart.js visualizations.

---

## How It Works: The Analysis Logic

KickCast does not rely on random guessing. It uses a mathematical approach to determine the "Attack Strength" and "Defense Strength" of every team to calculate **Expected Goals (xG)**.

### 1\. The Mathematical Model: Poisson Distribution

Soccer scores are "rare events" that occur independently within a fixed timeframe. This makes the **Poisson Distribution** the ideal statistical model for predicting scores.

The probability of a team scoring $k$ goals is calculated as:

$$P(k) = \frac{e^{-\lambda} \lambda^k}{k!}$$

Where $\lambda$ (Lambda) represents the **Expected Goals** for that specific match.

### 2\. Calculating Team Strengths (The Inputs)

To find $\lambda$, the system first calculates relative strengths compared to the league average:

- **Attack Strength:** (Average goals scored by Team A) ÷ (Average goals scored by an average league team).
- **Defense Strength:** (Average goals conceded by Team A) ÷ (Average goals conceded by an average league team).

These calculations are split into **Home** and **Away** contexts, as home-field advantage is statistically significant.

### 3\. The "Dynamic Season" Algorithm

One common issue with statistical models is early-season volatility (e.g., a team wins their first game 5-0 and looks invincible).

KickCast solves this using a **Weighted Transition Period** (found in `prosessData.py`):

- **Early Season (\< 8 matches):** The model calculates a weighted average of **Last Season's Stats** and **Current Season's Stats**. As the season progresses, the weight shifts linearly toward the current form.
- **Mid-to-Late Season (8+ matches):** The model relies entirely on the current season's data.

### 4\. The Simulation Loop

Once the inputs are ready, the simulation (in `simulationModel.py`) runs as follows:

1.  **Calculate Expected Goals (xG):**
    ```python
    Home_xG = Home_Attack * Away_Defense * Avg_League_Home_Goals
    Away_xG = Away_Attack * Home_Defense * Avg_League_Away_Goals
    ```
2.  **Probability Matrix:** The system iterates through a matrix of scorelines (0-0 up to 7-7).
3.  **Outcome Summation:**
    - **Home Win %:** Sum of probabilities where Home Goals \> Away Goals.
    - **Draw %:** Sum of probabilities where Home Goals == Away Goals.
    - **Away Win %:** Sum of probabilities where Away Goals \> Home Goals.

---

## Features

- **Real-Time Data:** Fetches the latest match results and fixtures via `football-data.org` API.
- **Visualizations:** Interactive Doughnut charts (Chart.js) displaying win probabilities.
- **Detailed Stats:** View specific Attack and Defense ratings compared to the league average.
- **Match History:** Displays the last 10 matches for selected teams to analyze form.
- **Multi-League Support:** Designed to scale for Premier League, La Liga, Bundesliga, Serie A, and Ligue 1.

---

## Tech Stack

- **Backend:** Python 3.9+, Flask, SciPy (for Poisson `pmf`), Pandas.
- **Frontend:** HTML5, CSS3, Bootstrap 5, JavaScript (Fetch API), Chart.js.
- **Deployment:** Docker, Google Cloud Run (configured via `app.yaml`).
- **Data Source:** [football-data.org](https://www.football-data.org/).

---

## Installation & Setup

### Prerequisites

- Python 3.x
- pip

### 1\. Clone the Repository

```bash
git clone https://github.com/yourusername/kickcast.git
cd kickcast
```

### 2\. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3\. Set Environment Variables

You need an API token from [football-data.org](https://www.football-data.org/).
The application reads it lazily, so health and static routes work without provider
configuration. API-backed routes return a controlled unavailable response until the
variable is set.

```bash
# On Mac/Linux
export FOOTBALL_API_TOKEN="your_api_token_here"

# On Windows (Powershell)
$env:FOOTBALL_API_TOKEN="your_api_token_here"
```

Never put a real token in source files, `app.yaml`, Docker build arguments, or a
committed `.env` file. `.env.example` lists the required variable without a value.

For Cloud Run, store the replacement credential in Google Secret Manager and expose
one pinned secret version as `FOOTBALL_API_TOKEN` on the service revision. Grant the
Cloud Run service identity Secret Manager Secret Accessor permission, then bind the
secret without placing its value on the command line:

```bash
gcloud run services update SERVICE_NAME \
  --region REGION \
  --update-secrets=FOOTBALL_API_TOKEN=football-api-token:SECRET_VERSION
```

Confirm the new revision is healthy before directing traffic to it. See
[`SECURITY.md`](SECURITY.md) for the credential-incident and rotation checklist.

### 4\. Run the Application

```bash
python app.py
```

Visit `http://127.0.0.1:5000` (or the port specified in your terminal) to view the app.

---

## Project Structure

```text
├── app.py                 # Main Flask application entry point
├── simulationModel.py     # Core Poisson prediction logic
├── prosessData.py         # Statistical processing & weighting logic
├── fetchData.py           # API wrappers for football-data.org
├── static/
│   ├── foundationData/    # Cached JSON data for leagues/teams
│   ├── styles.css         # Custom styling
│   └── script.js          # Frontend logic (DOM manipulation)
├── templates/
│   └── index.html         # Main UI
└── requirements.txt       # Python dependencies
```

---

## Future Improvements

- **Player-Level Analysis:** incorporating key player injuries into the weighting system.
- **Live Odds Comparison:** Fetching betting odds to compare model value vs. bookmaker value.
- **Head-to-Head History:** Adding a weight factor for historical matchups between specific teams.

---

## License

This project is open-source. Please ensure you comply with the data usage policies of the `football-data.org` API.
