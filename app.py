from flask import Flask, render_template, request
import pandas as pd
from flask import Response
import os
from surprise import dump


# Path to dump files and name
dumpfile_knn = os.path.join("./data/dump/dump_knn_dump_file")
beer_pickel_path = os.path.join("./data/dump/beer_final.pkl")

# Load dump files
predictions_knn, algo_knn = dump.load(dumpfile_knn)
beers_df = pd.read_pickle(beer_pickel_path)
beers_df["beer_brewery"] = beers_df["beer_brewery"].replace("/", "-", regex=True)

# Create the trainset from the knn_algorithm in order to get the inner_ids
trainset_knn = algo_knn.trainset


def get_beer_brewery(beer_raw_id):
    beer_brewery = beers_df.loc[beers_df.beer_id == beer_raw_id, "beer_brewery"].values[0]
    return beer_brewery


def get_beer_raw_id(beer_name):
    beer_raw_id = beers_df.loc[beers_df.beer_brewery == beer_name, "beer_id"].values[0]
    return beer_raw_id


def get_beer_style(beer_raw_id):
    beer_style = beers_df.loc[beers_df.beer_id == beer_raw_id, "style"].values[0]
    return beer_style


def get_beer_score_mean(beer_raw_id):
    score_mean = beers_df.loc[beers_df.beer_id == beer_raw_id, "score"].values[0]
    return score_mean


def get_beer_neighbors(beer_raw_id):
    beer_inner_id = algo_knn.trainset.to_inner_iid(beer_raw_id)
    beer_neighbors = algo_knn.get_neighbors(beer_inner_id, k=10)
    beer_neighbors = (
        algo_knn.trainset.to_raw_iid(inner_id) for inner_id in beer_neighbors
    )
    return beer_neighbors


def get_beer_recc_df(beer_raw_id):
    beer_inner_id = algo_knn.trainset.to_inner_iid(beer_raw_id)
    beer_neighbors = algo_knn.get_neighbors(beer_inner_id, k=10)
    beer_neighbors = (
        algo_knn.trainset.to_raw_iid(inner_id) for inner_id in beer_neighbors
    )
    beers_id_recc = []
    beer_brewery_recc = []
    beer_style_recc = []
    beer_score_mean = []
    for beer in beer_neighbors:
        beers_id_recc.append(beer)
        beer_brewery_recc.append(get_beer_brewery(beer))
        beer_style_recc.append(get_beer_style(beer))
        beer_score_mean.append(get_beer_score_mean(beer))
    beer_reccomendations_df = pd.DataFrame(
        list(zip(beers_id_recc, beer_brewery_recc, beer_style_recc, beer_score_mean)),
        columns=["beer_id", "name", "style", "score_mean"],
    )
    return beer_reccomendations_df


################################################################
#                        Flask Setup                           #
################################################################

app = Flask(__name__)


################################################################
#                        Flask Routes                          #
################################################################

@app.route("/")
def home():
    return render_template("verification.html")


@app.route("/index.html")
def index():
    return render_template("index.html")


@app.errorhandler(404)
def invalid_route(e):
    return render_template("404.html")

# --------------------------------------------------------------#
#                       Recommender model routes                #
# --------------------------------------------------------------#

# Route returns the beer;brewery to populate the dropdown
@app.route("/knnrecommender.html")
def recommender_selector():
    beers = beers_df["beer_brewery"].tolist()
    beers.sort()
    beers.append("Choose a Beer")
    return render_template("knnrecommender.html", beers=beers)

# Beer_name is beer;brewery format to match the search route
@app.route("/neighbors/<beer_name>")  
def nearest_neighbors(beer_name):
    beer_raw_id = get_beer_raw_id(beer_name)
    df = get_beer_recc_df(beer_raw_id)
    df["score_mean"] = df["score_mean"].apply(lambda x: round(x, 2))

    # return json of the dataframe
    return Response(df.to_json(orient="records"), mimetype="application/json")


# Beer_name is beer;brewery format
@app.route("/predict", methods=["POST"])
def predict():
    data_dict = request.get_json()
    username = data_dict["username"]
    beer_name = data_dict["beer"]  
    beer_raw_id = get_beer_raw_id(beer_name)
    predict = algo_knn.predict(username, beer_raw_id)
    df_predict = pd.DataFrame(
        [predict], columns=["username", "beer_id", "r_ui", "prediction", "details"]
    )
    return Response(df_predict.to_json(orient="records"), mimetype="application/json")

# Route takes the username and returns the top10 and bottom 10 predicted ratings
@app.route("/userpredict/<username>")
def userpredict(username):
    beers = beers_df["beer_brewery"].tolist()
    predict_df = pd.DataFrame([])
    for beer in beers:
        beer_raw_id = get_beer_raw_id(beer)
        predict = algo_knn.predict(username, beer_raw_id)
        predict_df = predict_df.append(
            pd.DataFrame(
                [predict],
                columns=["username", "beer_id", "r_ui", "prediction", "details"],
            )
        )
    picks = pd.merge(predict_df, beers_df, on="beer_id")
    picks = picks.round({"prediction": 2, "score": 2})
    top_10picks = picks.sort_values(by=["prediction"], ascending=False)[:10]
    top_10picks["pick"] = "Top10"
    bot_10picks = picks.sort_values(by=["prediction"], ascending=False)[-10:]
    bot_10picks["pick"] = "Bottom10"
    user_picks = pd.concat([top_10picks, bot_10picks])
    
    return Response(user_picks.to_json(orient="records"), mimetype="application/json")


# Route will call /userpredict/<username> to render predictions for user with table
@app.route("/userpredict.html")
def predict_user_rating():
    return render_template("userpredict.html")


################################################################
#                           Main                               #
################################################################
if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))

  
