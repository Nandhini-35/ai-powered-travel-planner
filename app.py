from flask import Flask, render_template, request
import pandas as pd
from sklearn.cluster import KMeans

app = Flask(__name__)

# ---------------- LOAD DATASET ----------------
df = pd.read_csv("data/travel_data.csv")

# ---------------- MACHINE LEARNING (K-MEANS) ----------------
# Use ML only for grouping / ranking (NOT hard filtering)
X = df[
    ["Entrance Fee in INR", "time needed to visit in hrs", "ratings"]
]

kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(X)

# ---------------- FLASK ROUTE ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    places = []

    if request.method == "POST":
        city = request.form["city"].strip()
        interest = request.form["interest"].strip()
        budget = int(request.form["budget"])

        # -------- RULE-BASED FILTERING --------
        filtered = df[
            (df["City"].str.lower() == city.lower()) &
            (df["type"].str.lower() == interest.lower()) &
            (df["Entrance Fee in INR"] <= budget)
        ]

        # -------- ML-ASSISTED RANKING --------
        # Lower cluster = generally cheaper & shorter
        filtered = filtered.sort_values(
            by=["cluster", "ratings"],
            ascending=[True, False]
        )

        places = filtered["place"].tolist()

    return render_template("index.html", places=places)

if __name__ == "__main__":
    app.run(debug=True)
