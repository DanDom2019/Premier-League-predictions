// --- Global State Management ---
let firstTeam = null;
let firstTeamData = null; // Store first team's full data
let opponentTeam = null;
let opponentTeamData = null; // Store opponent team's full data
let nextMatchDetails = null;
let isReplay = false;
let predictionChart = null;

// --- Friendly error popup (debounced so back-to-back failures only alert once) ---
let lastApiUnavailableAt = 0;
function showApiUnavailableMessage() {
  const now = Date.now();
  if (now - lastApiUnavailableAt < 3000) return;
  lastApiUnavailableAt = now;
  alert(
    "Match data is currently unavailable. The new season may not have started yet, or our data provider has temporarily reached its usage limit. Please try again in a few minutes."
  );
}
// Detect if we're running locally or deployed
const isLocal =
  window.location.hostname === "localhost" ||
  window.location.hostname === "127.0.0.1";
const API_BASE = isLocal
  ? "http://127.0.0.1:5000"
  : "https://kickcast-1031094153243.us-central1.run.app"; // Cloud Run URL

const USE_STATIC_DATA = !isLocal;

// --- League Configuration ---
// Change this constant to switch between leagues (2021 = Premier League)
const CURRENT_LEAGUE_ID = 2021;

// ==================================================================
//  Functions for Populating Dropdowns
// ==================================================================

/**
 * Loads teams for a given league ID and populates the appropriate dropdown
 * @param {number|string} leagueId - The league ID to load teams for
 * @param {string} selectionType - Either "firstTeam" or "opponent"
 */
function loadLeagueTeams(leagueId, selectionType) {
  fetch("./static/foundationData/mainLeaguesTeams.json")
    .then((response) => response.json())
    .then((allTeamsData) => {
      // Convert leagueId to string for JSON key lookup
      const leagueKey = String(leagueId);
      const teams = allTeamsData[leagueKey];
      if (!teams) {
        console.error(`No teams found for league ID: ${leagueId}`);
        return;
      }

      const teamListId =
        selectionType === "firstTeam"
          ? "first-team-list"
          : "opponent-team-list";
      const teamList = document.getElementById(teamListId);
      teamList.innerHTML = "";

      teams.forEach((team) => {
        const li = document.createElement("li");
        const a = document.createElement("a");
        a.className = "dropdown-item";
        a.href = "#";
        a.textContent = team.Name;
        a.onclick = (event) => {
          event.preventDefault();
          if (selectionType === "firstTeam") {
            selectFirstTeam(team.TeamID, team.Name, leagueId);
          } else {
            nextMatchDetails = null;
            isReplay = false;
            selectOpponentTeam(team.TeamID, team.Name, leagueId);
          }
        };
        li.appendChild(a);
        teamList.appendChild(li);
      });
    })
    .catch((error) => console.error("Error fetching teams:", error));
}

// ==================================================================
// User Workflow Functions
// ==================================================================

function selectFirstTeam(teamId, teamName, leagueId) {
  // Clear any previous opponent data when selecting a new first team
  opponentTeam = null;
  opponentTeamData = null;
  nextMatchDetails = null;
  isReplay = false;

  // Use CURRENT_LEAGUE_ID if leagueId is not provided
  const effectiveLeagueId = leagueId || CURRENT_LEAGUE_ID;
  
  firstTeam = { id: teamId, name: teamName, leagueId: effectiveLeagueId };
  document.getElementById("first-team-dropdown-btn").textContent = teamName;

  // **CORRECTED ANIMATION LOGIC**
  // 1. Animate the main container upwards
  document.getElementById("main-container").classList.add("content-moved-up");

  // 2. Show the details container that holds the team info
  document
    .getElementById("first-team-details-container")
    .classList.remove("d-none");

  // 3. Populate the details and store the data
  apiCall(`/api/team/${teamId}`, "first-team-info", (data, elementId) => {
    displayTeamInfo(data, elementId);
    firstTeamData = data; // Store the team data for later use
  });
  displayLast10Matches(teamId, effectiveLeagueId, "last10");

  // 4. Show the opponent selection step and load opponent teams
  document.getElementById("step2-container").classList.remove("d-none");
  // Load opponent teams when step 2 is shown
  loadLeagueTeams(CURRENT_LEAGUE_ID, "opponent");

  // 5. Hide other sections until needed
  document.getElementById("opponent-team-info-wrapper").classList.add("d-none");
  document.getElementById("step3-simulation-result").classList.add("d-none");
}

function selectOpponentTeam(teamId, teamName, leagueId) {
  // Use CURRENT_LEAGUE_ID if leagueId is not provided
  const effectiveLeagueId = leagueId || CURRENT_LEAGUE_ID;
  
  opponentTeam = { id: teamId, name: teamName, leagueId: effectiveLeagueId };

  document.getElementById("opponent-team-dropdown-btn").textContent = teamName;
  document
    .getElementById("opponent-team-info-wrapper")
    .classList.remove("d-none");

  // Fetch opponent team data directly and then display match-up
  fetchOpponentTeamData(teamId);

  displayLast10Matches(teamId, effectiveLeagueId, "opponent-last10");
  document.getElementById("start-simulation-btn").classList.remove("d-none");
  document.getElementById("step3-simulation-result").classList.add("d-none");
}

/**
 * Resets the application to the first stage (team selection).
 * Clears all state and restores the initial UI.
 */
function resetToFirstStage() {
  // Clear all global state variables
  firstTeam = null;
  firstTeamData = null;
  opponentTeam = null;
  opponentTeamData = null;
  nextMatchDetails = null;
  isReplay = false;

  // Destroy the prediction chart if it exists
  if (predictionChart) {
    predictionChart.destroy();
    predictionChart = null;
  }

  // Hide all step containers except step 1
  document.getElementById("step2-container").classList.add("d-none");
  document.getElementById("step3-simulation-result").classList.add("d-none");
  document.getElementById("first-team-details-container").classList.add("d-none");
  document.getElementById("opponent-team-info-wrapper").classList.add("d-none");
  document.getElementById("start-simulation-btn").classList.add("d-none");

  // Restore initial layout by removing the moved-up class
  document.getElementById("main-container").classList.remove("content-moved-up");

  // Reset dropdown button texts
  document.getElementById("first-team-dropdown-btn").textContent = "Select Team";
  document.getElementById("opponent-team-dropdown-btn").textContent = "Select Opponent";

  // Clear displayed content in result containers
  document.getElementById("first-team-info").innerHTML = "";
  document.getElementById("opponent-team-info").innerHTML = "";
  document.getElementById("match-result-display").innerHTML = "";
  document.getElementById("last10-table-body").innerHTML = "";
  document.getElementById("opponent-last10-table-body").innerHTML = "";
}

async function simulateNextOfficialMatch() {
  if (!firstTeam) {
    alert("Please select a team first.");
    return;
  }
  const endpoint = `/api/team/${firstTeam.id}/next_match`;

  try {
    const response = await fetch(API_BASE + endpoint);
    const matchData = await response.json();

    if (response.ok) {
      if (matchData.status === "upcoming") {
        nextMatchDetails = matchData.match;
        isReplay = false;
      } else if (matchData.status === "season_ended") {
        nextMatchDetails = matchData.last_match;
        isReplay = true;
      } else {
        alert("Unexpected response from server.");
        return;
      }

      const opponent =
        nextMatchDetails.awayTeam.id === firstTeam.id
          ? nextMatchDetails.homeTeam
          : nextMatchDetails.awayTeam;
      const opponentLeagueId = nextMatchDetails.competition.id;

      // Set opponent team info and fetch their data
      opponentTeam = {
        id: opponent.id,
        name: opponent.name,
        leagueId: opponentLeagueId,
      };
      document.getElementById("opponent-team-dropdown-btn").textContent =
        opponent.name;
      document
        .getElementById("opponent-team-info-wrapper")
        .classList.remove("d-none");

      // Fetch opponent team data and then display match-up
      fetchOpponentTeamData(opponent.id);

      displayLast10Matches(opponent.id, opponentLeagueId, "opponent-last10");
      document
        .getElementById("start-simulation-btn")
        .classList.remove("d-none");
      document
        .getElementById("step3-simulation-result")
        .classList.add("d-none");
    } else {
      showApiUnavailableMessage();
    }
  } catch (error) {
    showApiUnavailableMessage();
  }
}

async function fetchOpponentTeamData(teamId) {
  try {
    const response = await fetch(API_BASE + `/api/team/${teamId}`);
    const data = await response.json();

    if (response.ok) {
      opponentTeamData = data; // Store the opponent team data
      displayMatchUp(); // Now display the match-up with stored data
    } else {
      document.getElementById("opponent-team-info").innerHTML = "";
      showApiUnavailableMessage();
    }
  } catch (error) {
    document.getElementById("opponent-team-info").innerHTML = "";
    showApiUnavailableMessage();
  }
}

function displayMatchUp() {
  const container = document.getElementById("opponent-team-info");

  // Check if we have the required team data
  if (!firstTeamData || !opponentTeamData) {
    container.innerHTML = `<p class="text-danger">Team data not available. Please try again.</p>`;
    return;
  }

  // Determine home and away teams
  let homeTeamData, awayTeamData;

  if (nextMatchDetails) {
    // If we have official match details, use the actual home/away designation
    if (nextMatchDetails.homeTeam.id === firstTeam.id) {
      homeTeamData = firstTeamData;
      awayTeamData = opponentTeamData;
    } else {
      homeTeamData = opponentTeamData;
      awayTeamData = firstTeamData;
    }
  } else {
    // For manual selection, first team is on the left, opponent on the right
    homeTeamData = firstTeamData;
    awayTeamData = opponentTeamData;
  }

  const matchUpHtml = `
        <div class="row align-items-center">
            <!-- Home Team (Left) -->
            <div class="col-md-4 text-center">
                <div class="team-card">
                    <img src="${homeTeamData.crest}" alt="${
    homeTeamData.name
  } logo" class="team-logo mb-2" style="width: 80px; height: 80px;">
                    <h5 class="mb-1">${homeTeamData.name}</h5>
                    <img src="${homeTeamData.area.flag}" alt="${
    homeTeamData.area.name
  } flag" style="width: 30px; height: auto; margin-bottom: 8px;">
                    <p class="mb-1 text-muted small"><strong>Venue:</strong> ${
                      homeTeamData.venue
                    }</p>
                    <p class="mb-0 text-muted small"><strong>Founded:</strong> ${
                      homeTeamData.founded || "N/A"
                    }</p>
                </div>
            </div>
            
            <!-- Match Details (Center) -->
            <div class="col-md-4 text-center">
                <div class="match-details">
                    <h4 class="mb-3">VS</h4>
                    ${
                      nextMatchDetails && isReplay
                        ? `
                        <div class="match-info">
                            <span class="badge bg-secondary mb-2">🔁 Season Ended — Replay of last match</span>
                            <p class="mb-1"><strong>Date:</strong></p>
                            <p class="mb-2">${new Date(
                              nextMatchDetails.utcDate
                            ).toLocaleDateString()}</p>
                            <p class="mb-1"><strong>Final Score:</strong></p>
                            <p class="mb-0">${nextMatchDetails.score?.fullTime?.home ?? "-"} – ${nextMatchDetails.score?.fullTime?.away ?? "-"}</p>
                        </div>
                    `
                        : nextMatchDetails
                        ? `
                        <div class="match-info">
                            <p class="mb-1"><strong>Date & Time:</strong></p>
                            <p class="mb-2">${new Date(
                              nextMatchDetails.utcDate
                            ).toLocaleString()}</p>
                            <p class="mb-1"><strong>Venue:</strong></p>
                            <p class="mb-0">${nextMatchDetails.venue || homeTeamData.venue || "N/A"}</p>
                        </div>
                    `
                        : `
                        <div class="match-info">
                            <p class="mb-0 text-muted">Match details will be available when scheduled</p>
                        </div>
                    `
                    }
                </div>
            </div>
            
            <!-- Away Team (Right) -->
            <div class="col-md-4 text-center">
                <div class="team-card">
                    <img src="${awayTeamData.crest}" alt="${
    awayTeamData.name
  } logo" class="team-logo mb-2" style="width: 80px; height: 80px;">
                    <h5 class="mb-1">${awayTeamData.name}</h5>
                    <img src="${awayTeamData.area.flag}" alt="${
    awayTeamData.area.name
  } flag" style="width: 30px; height: auto; margin-bottom: 8px;">
                    <p class="mb-1 text-muted small"><strong>Venue:</strong> ${
                      awayTeamData.venue
                    }</p>
                    <p class="mb-0 text-muted small"><strong>Founded:</strong> ${
                      awayTeamData.founded || "N/A"
                    }</p>
                </div>
            </div>
        </div>
    `;

  container.innerHTML = matchUpHtml;
}

// ==================================================================
// Display and Simulation Functions
// ==================================================================

function displayTeamInfo(teamData, elementId) {
  const container = document.getElementById(elementId);
  if (
    !teamData ||
    !teamData.crest ||
    !teamData.name ||
    !teamData.venue ||
    !teamData.area
  ) {
    container.innerHTML = `<p class="text-danger">Could not display team info.</p>`;
    return;
  }
  const teamCardHtml = `
        <div class="d-flex align-items-center">
            <img src="${teamData.crest}" alt="${
    teamData.name
  } logo" style="width: 75px; height: 75px; margin-right: 15px;">
            <div>
                <h4 class="mb-1">${teamData.name}
                <img src="${teamData.area.flag}" alt="${
    teamData.area.name
  } flag" style="width: 40px; height: auto; margin-left: 10px;">
                </h4>  
                <p class="mb-0 text-muted"><strong>Venue:</strong> ${
                  teamData.venue
                }</p>
                <p class="mb-0 text-muted"><strong>Founded:</strong> ${
                  teamData.founded || "N/A"
                }</p>
            </div>
        </div>
    `;
  container.innerHTML = teamCardHtml;
}

// displayMatchDetails function removed - match details are now integrated into displayMatchUp

function runSimulation() {
  if (!firstTeam || !opponentTeam) {
    alert("Error: Both teams must be selected.");
    return;
  }

  let homeId, awayId, leagueId;
  if (nextMatchDetails) {
    homeId = nextMatchDetails.homeTeam.id;
    awayId = nextMatchDetails.awayTeam.id;
    leagueId = nextMatchDetails.competition?.id ?? firstTeam.leagueId;
  } else {
    homeId = firstTeam.id;
    awayId = opponentTeam.id;
    leagueId = firstTeam.leagueId;
  }

  const endpoint = `/simulation/predict?home=${homeId}&away=${awayId}&leagueId=${leagueId}`;
  document.getElementById("step3-simulation-result").classList.remove("d-none");
  apiCall(endpoint, "match-result-display", displayPredictionResult);
}
function displayPredictionResult(data, elementId) {
  const container = document.getElementById(elementId);
  console.log("data", data);
  if (!data || data.error) {
      container.innerHTML = `<p class="text-danger">Failed to get a valid simulation result: ${
      data.error || "Unknown error"
    }</p>`;
      return;
  }

  // 1. Format Top 5 Scores
  let topScoresHtml = "";
  if (data.top_five_scores && Array.isArray(data.top_five_scores)) {
    data.top_five_scores.forEach((item) => {
        topScoresHtml += `
            <li class="list-group-item d-flex justify-content-between align-items-center">
                Score: <strong>${item.score}</strong>
                <span class="badge bg-info rounded-pill">${item.probability}%</span>
            </li>
        `;
    });
  }

  // 2. Format Team Stats (Attack / Defense) with safe access
  // Check if data exists before accessing properties
  const homeStats = data.home_team_stats || {};
  const awayStats = data.away_team_stats || {};
  const leagueAvgs = data.league_averages || {};
  
  // ========== DEBUG LOGGING ==========
  console.group("🔍 Simulation Data Debug");
  console.log("Full API Response:", data);
  console.log("Response Keys:", Object.keys(data));
  console.log("---");
  console.log("homeStats:", homeStats);
  console.log("homeStats Type:", typeof homeStats);
  console.log("homeStats Keys:", Object.keys(homeStats));
  console.log("homeStats Empty?", Object.keys(homeStats).length === 0);
  console.log("---");
  console.log("awayStats:", awayStats);
  console.log("awayStats Type:", typeof awayStats);
  console.log("awayStats Keys:", Object.keys(awayStats));
  console.log("awayStats Empty?", Object.keys(awayStats).length === 0);
  console.log("---");
  console.log("leagueAvgs:", leagueAvgs);
  console.log("leagueAvgs Type:", typeof leagueAvgs);
  console.log("leagueAvgs Keys:", Object.keys(leagueAvgs));
  console.log("leagueAvgs Empty?", Object.keys(leagueAvgs).length === 0);
  console.log("---");
  
  // Check if stats are empty objects and log warnings
  if (Object.keys(homeStats).length === 0) {
    console.warn("⚠️ WARNING: home_team_stats is empty object! This indicates a backend issue.");
    console.warn("Expected keys: attack_strength_home, defense_strength_home, attack_strength_away, defense_strength_away");
  }
  if (Object.keys(awayStats).length === 0) {
    console.warn("⚠️ WARNING: away_team_stats is empty object! This indicates a backend issue.");
    console.warn("Expected keys: attack_strength_home, defense_strength_home, attack_strength_away, defense_strength_away");
  }
  if (Object.keys(leagueAvgs).length === 0) {
    console.warn("⚠️ WARNING: league_averages is empty object! This indicates a backend issue.");
    console.warn("Expected keys: avg_home_goals, avg_away_goals");
  }
  
  // Validate specific keys exist
  const requiredHomeKeys = ["attack_strength_home", "defense_strength_home", "attack_strength_away", "defense_strength_away"];
  const requiredAwayKeys = ["attack_strength_home", "defense_strength_home", "attack_strength_away", "defense_strength_away"];
  const requiredLeagueKeys = ["avg_home_goals", "avg_away_goals"];
  
  const missingHomeKeys = requiredHomeKeys.filter(key => !(key in homeStats));
  const missingAwayKeys = requiredAwayKeys.filter(key => !(key in awayStats));
  const missingLeagueKeys = requiredLeagueKeys.filter(key => !(key in leagueAvgs));
  
  if (missingHomeKeys.length > 0) {
    console.error("❌ Missing keys in homeStats:", missingHomeKeys);
  }
  if (missingAwayKeys.length > 0) {
    console.error("❌ Missing keys in awayStats:", missingAwayKeys);
  }
  if (missingLeagueKeys.length > 0) {
    console.error("❌ Missing keys in leagueAvgs:", missingLeagueKeys);
  }
  
  console.groupEnd();
  // ========== END DEBUG LOGGING ==========
  
  // Get numeric values first (for calculations), then format for display
  const homeAttackNum = parseFloat(homeStats.attack_strength_home) || 0;
  const homeDefNum = parseFloat(homeStats.defense_strength_home) || 0;
  const awayAttackNum = parseFloat(awayStats.attack_strength_away) || 0;
  const awayDefNum = parseFloat(awayStats.defense_strength_away) || 0;
  
  console.log("Parsed Values:", {
    homeAttackNum,
    homeDefNum,
    awayAttackNum,
    awayDefNum
  });
  // Format for display (2 decimal places)
  const homeAttack = homeAttackNum.toFixed(2);
  const homeDef = homeDefNum.toFixed(2);
  const awayAttack = awayAttackNum.toFixed(2);
  const awayDef = awayDefNum.toFixed(2);

  // 3. Format League Averages with safe access
  const leagueHomeNum = parseFloat(leagueAvgs.avg_home_goals) || 0;
  const leagueAwayNum = parseFloat(leagueAvgs.avg_away_goals) || 0;
  const leagueHome = leagueHomeNum.toFixed(2);
  const leagueAway = leagueAwayNum.toFixed(2);

  // 4. Format predicted goals with safe access
  const predictedGoalsHome = (data.predicted_goals_home !== undefined && data.predicted_goals_home !== null)
    ? parseFloat(data.predicted_goals_home).toFixed(2)
    : "0.00";
  const predictedGoalsAway = (data.predicted_goals_away !== undefined && data.predicted_goals_away !== null)
    ? parseFloat(data.predicted_goals_away).toFixed(2)
    : "0.00";

  container.innerHTML = `
      <div class="row align-items-center mb-4">
          <div class="col-md-5 text-center">
              <h5>${firstTeam.name}</h5>
              <h1 class="display-4 text-success">${data.home_team_win_probability}%</h1>
              <p class="text-muted">Win Probability</p>
          </div>
          <div class="col-md-2 text-center">
              <h5 class="text-muted">Draw</h5>
              <h2 class="display-5">${data.draw_probability}%</h2>
          </div>
          <div class="col-md-5 text-center">
              <h5>${opponentTeam.name}</h5>
              <h1 class="display-4 text-danger">${data.away_team_win_probability}%</h1>
              <p class="text-muted">Win Probability</p>
          </div>
      </div>

      <div class="card mb-4 bg-light border-0">
          <div class="card-body">
              <div class="row text-center">
                  <div class="col-md-4">
                      <h6 class="text-muted">Attack Strength</h6>
                      <div class="d-flex justify-content-around align-items-center">
                          <div><strong>${homeAttack}</strong></div>
                          <small class="text-muted">vs League Avg (1.0)</small>
                          <div><strong>${awayAttack}</strong></div>
                      </div>
                  </div>

                  <div class="col-md-4 border-start border-end">
                      <h6 class="text-muted">League Averages</h6>
                      <p class="mb-1 small">Avg Home Goals: <strong>${leagueHome}</strong></p>
                      <p class="mb-0 small">Avg Away Goals: <strong>${leagueAway}</strong></p>
                  </div>

                  <div class="col-md-4">
                      <h6 class="text-muted">Defense Strength</h6>
                       <div class="d-flex justify-content-around align-items-center">
                          <div><strong>${homeDef}</strong></div>
                          <small class="text-muted">(Lower is better)</small>
                          <div><strong>${awayDef}</strong></div>
                      </div>
                  </div>
              </div>
          </div>
      </div>

      <hr>

      <div class="row mt-3">
          <div class="col-md-7">
              <h5>Outcome Visualization</h5>
              <canvas id="predictionChart"></canvas>
          </div>
          <div class="col-md-5">
              <h5>Most Likely Scorelines</h5>
              <ul class="list-group mb-3"> ${topScoresHtml} </ul>
              
              <h5>Expected Goals (xG)</h5>
              <ul class="list-group">
                  <li class="list-group-item d-flex justify-content-between align-items-center">
                      ${firstTeam.name}:
                      <span class="badge bg-primary rounded-pill">${predictedGoalsHome}</span>
                  </li>
                  <li class="list-group-item d-flex justify-content-between align-items-center">
                      ${opponentTeam.name}:
                      <span class="badge bg-primary rounded-pill">${predictedGoalsAway}</span>
                  </li>
              </ul>
          </div>
      </div>
  `;

  // Render the Chart
  const ctx = document.getElementById("predictionChart").getContext("2d");
  if (predictionChart) {
      predictionChart.destroy();
  }
  predictionChart = new Chart(ctx, {
      type: "doughnut",
      data: {
          labels: [`${firstTeam.name} Win`, "Draw", `${opponentTeam.name} Win`],
          datasets: [{
              label: "Match Outcome Probability",
              data: [
                  data.home_team_win_probability,
                  data.draw_probability,
                  data.away_team_win_probability,
              ],
              backgroundColor: [
                  "rgba(5, 242, 108, 0.7)", // Premier League Green #05F26C
                  "rgba(234, 242, 5, 0.7)", // Premier League Yellow #EAF205
                  "rgba(242, 5, 92, 0.7)", // Premier League Pink #F2055C
              ],
              borderColor: [
                  "#05F26C",
                  "#EAF205",
                  "#F2055C",
              ],
              borderWidth: 1,
          }, ],
      },
      options: {
          responsive: true,
          plugins: {
              legend: {
                  position: "top",
                  labels: {
                      color: "#ffffff"
                  },
              },
              title: {
                  display: true,
                  text: "Match Outcome Prediction",
                  color: "#ffffff",
              },
          },
      },
  });
}
async function displayLast10Matches(teamId, leagueId, tableType) {
  const tableBody = document.getElementById(`${tableType}-table-body`);
  const loadingState = document.getElementById(`${tableType}-loading-state`);
  const endpoint = `/app/team/${teamId}/last10matches?leagueId=${leagueId}`;

  tableBody.innerHTML = "";
  loadingState.classList.remove("d-none");

  try {
    const response = await fetch(API_BASE + endpoint);
    const matches = await response.json();
    loadingState.classList.add("d-none");

    if (!response.ok || matches.error) {
      tableBody.innerHTML = "";
      showApiUnavailableMessage();
      return;
    }

    const getResultBadge = (result) => {
      if (result === "Win")
        return `<span class="badge bg-success">${result}</span>`;
      if (result === "Loss")
        return `<span class="badge bg-danger">${result}</span>`;
      return `<span class="badge bg-warning text-dark">${result}</span>`;
    };

    matches.forEach((match) => {
      const row = `
                <tr>
                    <td>${match.matchDay}</td>
                    <td>${match.date}</td>
                    <td>${match.opponent}</td>
                    <td>${match.score}</td>
                    <td>${getResultBadge(match.result)}</td>
                </tr>
            `;
      tableBody.innerHTML += row;
    });
  } catch (error) {
    loadingState.classList.add("d-none");
    tableBody.innerHTML = "";
    showApiUnavailableMessage();
  }
}

// ==================================================================
//  Utility Function for API Calls
// ==================================================================

async function apiCall(endpoint, resultId, displayFunction) {
  const resultDiv = document.getElementById(resultId);
  resultDiv.innerHTML =
    '<div class="text-center"><div class="spinner-border" role="status"><span class="visually-hidden">Loading...</span></div></div>';
  try {
    const response = await fetch(API_BASE + endpoint);
    const data = await response.json();

    if (response.ok) {
      if (displayFunction) {
        displayFunction(data, resultId);
      } else {
        resultDiv.innerHTML =
          "<pre>" + JSON.stringify(data, null, 2) + "</pre>";
      }
    } else {
      resultDiv.innerHTML = "";
      showApiUnavailableMessage();
    }
  } catch (error) {
    resultDiv.innerHTML = "";
    showApiUnavailableMessage();
  }
}

// ==================================================================
//  Initialize on Page Load
// ==================================================================

// Load teams when the page loads
document.addEventListener("DOMContentLoaded", function() {
  loadLeagueTeams(CURRENT_LEAGUE_ID, "firstTeam");
});
